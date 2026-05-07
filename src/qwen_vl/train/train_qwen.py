# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
)
from qwen_vl.data.data_qwen import make_supervised_data_module

from qwen_vl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer, AutoConfig, set_seed, enable_full_determinism, TrainerCallback, GenerationConfig
import gc

local_rank = None


class ClearCacheCallback(TrainerCallback):
    """Callback to clear CUDA cache after each training step to avoid memory fragmentation."""

    def on_step_end(self, args, state, control, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    # Get visual module (same for both Qwen2.5-VL and Qwen3-VL)
    visual_module = model.visual if hasattr(model, 'visual') else model.model.visual

    if model_args.tune_mm_vision:
        for n, p in visual_module.named_parameters():
            p.requires_grad = True
    else:
        for n, p in visual_module.named_parameters():
            p.requires_grad = False

    # Visual merger - controlled by tune_mm_mlp (only visual_module.merger)
    if model_args.tune_mm_mlp:
        for n, p in visual_module.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in visual_module.merger.named_parameters():
            p.requires_grad = False

    # Feature projector - DEPRECATED
    if getattr(model_args, 'tune_feature_projector', False):
        raise DeprecationWarning(
            "tune_feature_projector is no longer supported. "
            "feature_projector has been removed from the model."
        )

    # Memory feature projector - DEPRECATED
    if getattr(model_args, 'tune_memory_feature_projector', False):
        raise DeprecationWarning(
            "tune_memory_feature_projector is no longer supported. "
            "memory_feature_projector has been removed from the model."
        )

    # Memory feature fusion - independent control
    if hasattr(model, 'memory_feature_fusion'):
        tune_mff = getattr(model_args, 'tune_memory_feature_fusion', False)
        for n, p in model.memory_feature_fusion.named_parameters():
            p.requires_grad = tune_mff

    # Pointer position encoder - independent control
    if hasattr(model, 'pointer_position_encoder'):
        tune_ppe = getattr(model_args, 'tune_pointer_position_encoder', True)
        for n, p in model.pointer_position_encoder.named_parameters():
            p.requires_grad = tune_ppe

    # Get LLM module - Qwen3-VL uses model.model.language_model, Qwen2.5-VL uses model.model
    if hasattr(model.model, 'language_model'):
        # Qwen3-VL structure
        llm_module = model.model.language_model
    else:
        # Qwen2.5-VL structure
        llm_module = model.model

    if model_args.tune_mm_llm:
        for n, p in llm_module.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in llm_module.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

    if model_args.use_pointer_memory and not model_args.use_preprocessed_input:
        # point3r memory is frozen
        for n, p in model.point3r_model.named_parameters():
            p.requires_grad = False

def train(attn_implementation="sdpa"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    # enable_full_determinism(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    print('='*70)
    if "qwen3.5" in model_args.model_name_or_path.lower():
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path
            ).image_processor
        data_args.model_type = "qwen3.5"

    elif "qwen3" in model_args.model_name_or_path.lower():
        if getattr(model_args, "use_pointer_memory", False):
            from qwen_vl.model.qwen3_vl.modeling_qwen3_point3r import Qwen3VLForConditionalGenerationWithPoint3R
            from qwen_vl.model.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessorWithPoint3R
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            if hasattr(config, "use_pointer_memory") and config.use_pointer_memory != model_args.use_pointer_memory:
                raise ValueError(
                    "The use_pointer_memory in config and model_args are not consistent. "
                    "Please check the model config."
                )
            for k in [
                "use_pointer_memory",
                "use_preprocessed_input",
                "point3r_model_path",
                "pointer_memory_size",
                # Memory feature fusion parameters (Point3R)
                "merge_memory_feat",
                "memory_fusion_method",
                "memory_fusion_attention_heads",
                "memory_fusion_dropout",
                "memory_fusion_num_layers",
                "memory_merger_hidden_dim",
                "memory_merger_type",
                # Pointer position encoding parameters
                "use_pointer_position_encoding",
                "pointer_pos_hidden_dim",
                # RoPE ablation parameters
                "rope_mode",
                "rope_position_range",
                "tune_rope3d_continuous",
            ]:
                setattr(config, k, getattr(model_args, k))

            # Add missing vision config attributes for Qwen2.5VL compatibility
            if hasattr(config, "vision_config"):
                if not hasattr(config.vision_config, "fullatt_block_indexes"):
                    depth = getattr(config.vision_config, "depth", 27)
                    config.vision_config.fullatt_block_indexes = list(range(depth))
                if not hasattr(config.vision_config, "window_size"):
                    config.vision_config.window_size = 112

            # Add missing text config attributes for Qwen2.5VL compatibility
            if hasattr(config, "text_config"):
                num_hidden_layers = getattr(config.text_config, "num_hidden_layers", 32)
                if not hasattr(config.text_config, "use_sliding_window"):
                    config.text_config.use_sliding_window = False
                if not hasattr(config.text_config, "sliding_window"):
                    config.text_config.sliding_window = None
                if not hasattr(config.text_config, "max_window_layers"):
                    config.text_config.max_window_layers = num_hidden_layers
                if not hasattr(config.text_config, "layer_types"):
                    config.text_config.layer_types = ["full_attention"] * num_hidden_layers

            assert model_args.use_preprocessed_input or model_args.point3r_model_path is not None, \
                "When use_pointer_memory is True, use_preprocessed_input must be True or point3r_model_path must be set in the config."
            model = Qwen3VLForConditionalGenerationWithPoint3R.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                # Don't load Point3R model when using preprocessed inputs
                point3r_model_path=None if model_args.use_preprocessed_input else model_args.point3r_model_path
            )

            base_processor = AutoProcessor.from_pretrained(
                model_args.model_name_or_path,
                use_fast=True,
                cache_dir=training_args.cache_dir,
                min_pixels=data_args.min_pixels,
                max_pixels=data_args.max_pixels
            )

            # Create Point3R processor with pointer token support for Qwen3-VL
            processor = Qwen3VLProcessorWithPoint3R(
                image_processor=base_processor.image_processor,
                tokenizer=base_processor.tokenizer,
                video_processor=base_processor.video_processor if hasattr(base_processor, 'video_processor') else None,
                chat_template=base_processor.chat_template if hasattr(base_processor, 'chat_template') else None,
                pointer_format=getattr(model_args, "pointer_format", "video"),
                add_frame_id=getattr(model_args, "add_frame_id", False),
            )

            # Store pointer token ID in model config for proper processing
            model.config.pointer_token_id = processor.pointer_token_id
            model.pointer_token_id = processor.pointer_token_id

            # Resize token embeddings to accommodate new pointer token
            model.resize_token_embeddings(len(processor.tokenizer))

            data_args.image_processor = processor.image_processor

            # Determine model type based on rope_mode
            rope_mode = getattr(model_args, "rope_mode", "none")
            if rope_mode == "discrete":
                data_args.model_type = "qwen3vl-rope-discrete"
            elif rope_mode == "continuous":
                data_args.model_type = "qwen3vl-rope-continuous"
            elif rope_mode == "pointer_timestamp":
                data_args.model_type = "qwen3vl-rope-pointer"
            else:
                data_args.model_type = "qwen3vl"

            # Pass RoPE config to model
            model.config.rope_mode = rope_mode
            model.config.rope_position_range = getattr(model_args, "rope_position_range", 128)

            # Pass pointer format to data pipeline
            data_args.pointer_format = getattr(model_args, "pointer_format", "video")
            # Pass pointer directory name override to data pipeline
            data_args.pointer_dir_name = getattr(model_args, "pointer_dir_name", None)
            # Pass frame ID label option to data pipeline
            data_args.add_frame_id = getattr(model_args, "add_frame_id", False)
        else:
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
            data_args.image_processor = AutoProcessor.from_pretrained(
                model_args.model_name_or_path
                ).image_processor
            data_args.model_type = "qwen3vl"

    elif "qwen2.5" in model_args.model_name_or_path.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path
            ).image_processor
        data_args.model_type = "qwen2.5vl"

    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        pass
        # replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if model_args.use_pointer_memory:
        tokenizer = processor.tokenizer
        assert tokenizer.padding_side == "right", "Padding side must be right"
        # Q. does use_fast make any difference?
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        # Get visual module - same for both Qwen2.5-VL and Qwen3-VL
        visual_module = model.visual if hasattr(model, 'visual') else model.model.visual
        if hasattr(visual_module, 'print_trainable_parameters'):
            visual_module.print_trainable_parameters()

        # Get LLM module - Qwen3-VL uses model.model.language_model, Qwen2.5-VL uses model.model
        if hasattr(model.model, 'language_model'):
            llm_module = model.model.language_model
        else:
            llm_module = model.model
        if hasattr(llm_module, 'print_trainable_parameters'):
            llm_module.print_trainable_parameters()

    # Set generation config for instruct (non-thinking) mode
    model.generation_config = GenerationConfig(
        presence_penalty=1.5,
        repetition_penalty=1.0,
        do_sample=False,
    )

    # Pass processor if using Point3R, otherwise pass None
    processor_to_pass = processor if model_args.use_pointer_memory else None
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, processor=processor_to_pass)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args,
        callbacks=[ClearCacheCallback()],
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    # Try to copy chat_template.json if it exists
    source_path = os.path.join(model_args.model_name_or_path, "chat_template.json")
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    if os.path.exists(source_path):
        shutil.copy2(source_path, template_path)
    else:
        # Try to find it in the cache directory
        cache_pattern = os.path.join(training_args.cache_dir or "./cache", f"models--{model_args.model_name_or_path.replace('/', '--')}", "**", "chat_template.json")
        import glob
        cached_files = glob.glob(cache_pattern, recursive=True)
        if cached_files:
            shutil.copy2(cached_files[0], template_path)
        else:
            rank0_print(f"Warning: chat_template.json not found at {source_path} or in cache, skipping copy")

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="sdpa")
