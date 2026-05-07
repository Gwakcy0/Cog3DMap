"""
Point3R-LLM Model Wrapper for lmms-eval

This module implements a wrapper for Qwen2.5-VL enhanced with Point3R pointer memory,
enabling 3D scene understanding through pointer tokens.

Key Features:
1. **Pre-computed Pointer Memory Support**: Load pre-processed .pt files containing
   pointer_memory_embeds and pointer_positions to avoid on-the-fly computation.
2. **On-the-fly Extraction**: Automatically extracts pointer memory if pre-computed
   data is not available.
3. **Two-stage Processing** (following demo_point3r.py):
   - Stage 1: Extract image embeddings via model.visual()
   - Stage 2: Pass to Point3R for pointer memory extraction
4. **Batch Processing**: Configurable batch size for memory-efficient extraction.
5. **Caching**: In-memory cache for loaded pointer data to speed up evaluation.

Usage Examples:

1. With pre-computed pointer data:
   ```bash
   lmms-eval \\
       --model point3r_llm \\
       --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct,use_preprocessed_input=True,base_dir=data/media \\
       --tasks <task_name> \\
       --batch_size 1
   ```

2. With on-the-fly extraction:
   ```bash
   lmms-eval \\
       --model point3r_llm \\
       --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct,use_pointer_memory=True,point3r_model_path=./cache/point3r_512.pth \\
       --tasks <task_name> \\
       --batch_size 1
   ```

Pre-computed Pointer Data Format:
The .pt files should contain a dictionary with:
- 'pointer_memory_embeds': torch.Tensor of shape (num_pointers, hidden_dim)
- 'pointer_positions': torch.Tensor of shape (num_pointers, 3) with (height, width, depth)

Path Resolution:
Task annotations contain relative paths in the 'pointer_data' field.
These are joined with base_dir (default: "data/media") to get the full path:
- Example: base_dir="data/media" + pointer_data="scannet/pointer_memory/scene0000_00.pt"
  -> "data/media/scannet/pointer_memory/scene0000_00.pt"
"""

import base64
import os
from io import BytesIO
from typing import List, Optional, Tuple, Union

import copy
import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoConfig,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

from qwen_vl.model.qwen3_vl.modeling_qwen3_point3r import Qwen3VLForConditionalGenerationWithPoint3R
from qwen_vl.model.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessorWithPoint3R
from qwen_vl.model.point3r.point3r import Point3R
from qwen_vl.model.point3r.extract_memory import extract_pointer_memory
from qwen_vl.utils.config_validation import validate_parameters
from qwen_vl.data.pointer_data import load_pointer_data

try:
    from qwen_vl_utils import extract_vision_info, process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("point3r_llm")
class Point3RLLMv2(lmms):
    """
    Point3R-LLM Model

    Qwen2.5-VL enhanced with Point3R pointer memory for 3D understanding.

    This model supports both pre-computed pointer memory (recommended for faster evaluation)
    and on-the-fly extraction using the Point3R model.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        max_length: Optional[int] = None,
        add_frame_index: bool = False,
        point3r_model_path: str = "./cache/point3r_512.pth",
        pointer_memory_size: int = 512,
        use_pointer_memory: bool = True,
        use_preprocessed_input: bool = True,
        base_dir: Optional[str] = "data/media",
        extract_batch_size: int = 2,
        # Memory feature fusion parameters (Point3R)
        merge_memory_feat: bool = False,
        memory_fusion_method: str = "add",
        memory_merger_hidden_dim: int = 4096,
        memory_merger_type: str = "mlp",
        tune_feature_projector: bool = False,
        # Pointer position encoding parameters
        use_pointer_position_encoding: bool = False,
        pointer_pos_hidden_dim: int = 256,
        tune_pointer_position_encoder: bool = True,
        # RoPE ablation parameters
        rope_mode: str = "none",
        rope_position_range: int = 128,
        # Pointer token format
        pointer_format: str = "video",
        # Pointer data directory override
        pointer_dir_name: Optional[str] = None,
        # Frame ID labels for pointer tokens
        add_frame_id: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate evaluation arguments against training config
        eval_args = {
            "pointer_memory_size": pointer_memory_size,
            "use_pointer_position_encoding": use_pointer_position_encoding,
            "pointer_pos_hidden_dim": pointer_pos_hidden_dim,
            "merge_memory_feat": merge_memory_feat,
            "memory_fusion_method": memory_fusion_method,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "memory_merger_hidden_dim": memory_merger_hidden_dim,
            "memory_merger_type": memory_merger_type,
            "rope_mode": rope_mode,
            "rope_position_range": rope_position_range,
        }
        try:
            validate_parameters(
                model_path=pretrained,
                eval_args=eval_args,
                fail_on_critical=True,
                warn_on_high=True,
            )
        except ValueError as e:
            eval_logger.error(str(e))
            raise

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.add_frame_index = add_frame_index
        self.max_image_size = max_image_size
        self.use_pointer_memory = use_pointer_memory
        self.pointer_memory_size = pointer_memory_size
        self.use_preprocessed_input = use_preprocessed_input
        self.extract_batch_size = extract_batch_size
        self.base_dir = base_dir
        self.merge_memory_feat = merge_memory_feat
        self.pointer_dir_name = pointer_dir_name

        # Cache for pre-loaded pointer data
        self.pointer_data_cache = {}

        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        config = AutoConfig.from_pretrained(pretrained)

        # Memory fusion config (must be set before model loading)
        if merge_memory_feat:
            config.merge_memory_feat = merge_memory_feat
            config.memory_fusion_method = memory_fusion_method
            config.memory_merger_hidden_dim = memory_merger_hidden_dim
            config.memory_merger_type = memory_merger_type
            eval_logger.info(f"Memory fusion enabled: method={memory_fusion_method}")

        # Feature projector config (must be set before model loading)
        if tune_feature_projector:
            config.tune_feature_projector = tune_feature_projector
            eval_logger.info("Feature projector enabled")

        # Pointer position encoding config (must be set before model loading)
        self.use_pointer_position_encoding = use_pointer_position_encoding
        if use_pointer_position_encoding:
            config.use_pointer_position_encoding = use_pointer_position_encoding
            config.pointer_pos_hidden_dim = pointer_pos_hidden_dim
            eval_logger.info(f"Pointer position encoding enabled: hidden_dim={pointer_pos_hidden_dim}, tune={tune_pointer_position_encoder}")

        # RoPE ablation config (must be set before model loading)
        self.rope_mode = rope_mode
        config.rope_mode = rope_mode
        config.rope_position_range = rope_position_range
        if rope_mode != "none":
            eval_logger.info(f"RoPE mode for pointers: {rope_mode}, position_range={rope_position_range}")

        # Pointer format config
        self.pointer_format = pointer_format
        self.add_frame_id = add_frame_id

        # Load Point3R-enhanced model
        eval_logger.info("Using Qwen3VLForConditionalGenerationWithPoint3R")

        if use_flash_attention_2:
            self._model = Qwen3VLForConditionalGenerationWithPoint3R.from_pretrained(
                pretrained,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = Qwen3VLForConditionalGenerationWithPoint3R.from_pretrained(
                pretrained,
                config=config,
                torch_dtype="auto",
                device_map=self.device_map
            ).eval()

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        # Load base processor
        base_processor = AutoProcessor.from_pretrained(
            pretrained,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            padding_side="left"
        )

        # Create Point3R processor with pointer token support
        self.processor = Qwen3VLProcessorWithPoint3R(
            image_processor=base_processor.image_processor,
            tokenizer=base_processor.tokenizer,
            video_processor=base_processor.video_processor,
            chat_template=base_processor.chat_template if hasattr(base_processor, 'chat_template') else None,
            pointer_format=self.pointer_format,
            add_frame_id=self.add_frame_id,
        )
        self._tokenizer = self.processor.tokenizer

        # Store pointer token ID in model config
        self._model.config.pointer_token_id = self.processor.pointer_token_id
        self._model.pointer_token_id = self.processor.pointer_token_id

        # Resize token embeddings to accommodate new pointer token
        self._model.resize_token_embeddings(len(self.processor.tokenizer))

        eval_logger.info(f"Pointer token: {self.processor.pointer_token}")
        eval_logger.info(f"Pointer token ID: {self.processor.pointer_token_id}")

        if max_length is not None:
            eval_logger.warning(f"Setting max_length to {max_length}")
            setattr(self.processor.tokenizer, "model_max_length", max_length)
            setattr(self._tokenizer, "model_max_length", max_length)

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        # Load Point3R model for memory extraction if enabled
        if self.use_pointer_memory and not self.use_preprocessed_input:
            eval_logger.info(f"Loading Point3R model from {point3r_model_path}...")
            self.point3r_model = Point3R.from_pretrained(point3r_model_path)
            point3r_device = torch.device("cuda" if self.device_map == "auto" else self.device_map)
            self.point3r_model = self.point3r_model.to(point3r_device)
            self.point3r_model.eval()
            eval_logger.info("Point3R model loaded successfully")
        else:
            self.point3r_model = None

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
            self._model = self.model.to("cuda").to(torch.bfloat16)

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Point3R-LLM")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def extract_pointer_memory_from_images(self, image_inputs):
        """
        Extract pointer memory from images using Point3R model.
        This follows the two-stage process from demo_point3r.py:
        Stage 1: Extract image embeddings via model.visual()
        Stage 2: Pass to Point3R for pointer memory extraction
        """
        if not self.use_pointer_memory or self.point3r_model is None:
            return None

        # Process images in batches to extract embeddings (Stage 1)
        image_embeds_list = []
        deepstack_image_embeds = []  # Track deepstack embeddings per layer
        grid_thw_list = []

        for i in range(0, len(image_inputs), self.extract_batch_size):
            batch_images = image_inputs[i:i+self.extract_batch_size]

            # Process batch
            processed_batch = self.processor.image_processor(
                images=batch_images,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels
            )

            with torch.inference_mode():
                # Get model device
                model_device = next(self.model.visual.parameters()).device

                # Move tensors to same device as model
                pixel_values = processed_batch.pixel_values.type(self.model.visual.dtype)
                pixel_values = pixel_values.to(model_device)
                grid_thw = processed_batch.image_grid_thw

                # Handle tuple return from visual encoder (embeds, deepstack_embeds)
                visual_output = self.model.visual(pixel_values, grid_thw=grid_thw)
                batch_deepstack_image_embeds = None
                if isinstance(visual_output, tuple):
                    batch_embeds, batch_deepstack_image_embeds = visual_output
                else:
                    batch_embeds = visual_output

                image_embeds_list.append(batch_embeds)

                # Aggregate deepstack embeddings
                if batch_deepstack_image_embeds is not None:
                    if len(deepstack_image_embeds) == 0:
                        deepstack_image_embeds = [[] for _ in range(len(batch_deepstack_image_embeds))]
                    for layer_idx, layer_embeds in enumerate(batch_deepstack_image_embeds):
                        deepstack_image_embeds[layer_idx].append(layer_embeds)

                grid_thw_list.append(grid_thw)

        # Concatenate all batches
        image_embeds = torch.cat(image_embeds_list, dim=0)
        grid_thw = torch.cat(grid_thw_list, dim=0)

        # Concatenate deepstack embeddings per layer
        if deepstack_image_embeds:
            deepstack_image_embeds = [
                torch.cat(layer_embeds_list, dim=0) for layer_embeds_list in deepstack_image_embeds
            ]

        # Get Point3R device
        point3r_device = next(self.point3r_model.parameters()).device

        # Move to Point3R device
        image_embeds = image_embeds.to(point3r_device)
        grid_thw = grid_thw.to(point3r_device)

        # Move deepstack embeddings to Point3R device
        if deepstack_image_embeds:
            deepstack_image_embeds = [layer.to(point3r_device) for layer in deepstack_image_embeds]

        # Extract pointer memory (Stage 2)
        pointer_data = extract_pointer_memory(
            image_inputs=image_inputs,
            point3r_model=self.point3r_model,
            image_embeds=image_embeds,
            grid_thw=grid_thw,
            deepstack_image_embeds=deepstack_image_embeds if deepstack_image_embeds else None,
            device=point3r_device,
            no_crop=False,
            size=self.pointer_memory_size,
            verbose=False,
        )

        return pointer_data

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            messages = []

            # Try to load or extract pointer memory if using pointer mode
            pointer_data = None
            if self.use_pointer_memory:
                # If pre-processed input exists, load from the document
                if self.use_preprocessed_input:
                    # IMPORTANT: With preprocessed input, we can only handle one sample at a time
                    # because each sample has its own pointer_data file
                    if len(contexts) > 1:
                        raise ValueError(
                            f"Preprocessed pointer input only supports batch_size=1, got {len(contexts)} contexts. "
                            "Please set --batch_size 1 in your evaluation command."
                        )

                    # Get the document and load pointer_data directly
                    doc = self.task_dict[task][split][doc_id[0]]

                    if "pointer_data" in doc:
                        pointer_data_path = doc["pointer_data"]
                        try:
                            pointer_data = load_pointer_data(
                                pointer_data_path=pointer_data_path,
                                base_dir=self.base_dir,
                                max_pointer_tokens=10000,
                                pointer_dir_name=self.pointer_dir_name,
                            )
                            eval_logger.debug(f"Loaded pointer data: {pointer_data['num_pointers']} tokens")
                        except FileNotFoundError as e:
                            eval_logger.warning(str(e))
                            pointer_data = None
                    else:
                        eval_logger.warning(f"No 'pointer_data' field found in document for task {task}")
                
                # If we have visuals, compute on-the-fly
                elif len(visuals) > 0:
                    # Collect all image inputs
                    image_inputs_for_pointer = [] # TODO: Disable this with use_preprocessed_input = True
                    for i, context in enumerate(contexts):
                        visual = visuals[i] if i < len(visuals) else None
                        if isinstance(visual, Image.Image):
                            image_inputs_for_pointer.append(visual)
                        elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
                            image_inputs_for_pointer.extend(visual)

                    # Extract pointer memory from all images
                    if image_inputs_for_pointer:
                        pointer_data = self.extract_pointer_memory_from_images(image_inputs_for_pointer)

            # Build messages with pointer tokens if available
            for i, context in enumerate(contexts):
                message = [{"role": "system", "content": "You are a helpful assistant."}]

                if self.use_pointer_memory and pointer_data is not None:
                    # Use pointer token instead of images
                    message.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": context}
                        ]
                    })
                elif len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        vr = decord.VideoReader(visual)
                        image_num = len(vr)
                        if image_num < self.max_num_frames:
                            frame_indices = np.arange(image_num)
                        else:
                            frame_indices = np.linspace(0, image_num - 1, self.max_num_frames).astype(int)
                        frames = [vr[i].asnumpy() for i in frame_indices]
                        visual_content = []
                        for frame in frames:
                            image = Image.fromarray(frame).convert("RGB")
                            visual_content.append({"type": "image", "image": image})
                        message.append({"role": "user", "content": visual_content + [{"type": "text", "text": context}]})
                    elif isinstance(visual, Image.Image):  # Single image
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"}, {"type": "text", "text": context}]})
                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                        image_content = []
                        image_count = 0
                        for v in visual:
                            base64_image = v.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            if self.add_frame_index:
                                image_content.append({"type": "text", "text": "Frame-{}: ".format(image_count)})
                            image_content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
                            image_count += 1
                        message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
                    else:
                        message.append({"role": "user", "content": [{"type": "text", "text": context}]})
                else:
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})

                messages.append(message)

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Prepare inputs
            if self.use_pointer_memory and pointer_data is not None:
                # Fallback: create zero timestamps if missing from preprocessed data
                pointer_timestamps = pointer_data.get('pointer_timestamps')
                if pointer_timestamps is None:
                    num_pointers = pointer_data['pointer_memory_embeds'].shape[0]
                    pointer_timestamps = torch.zeros(num_pointers, dtype=torch.long)
                    eval_logger.warning(
                        f"pointer_timestamps missing from preprocessed data, "
                        f"using zero tensor of length {num_pointers}"
                    )

                # Use pointer processing
                inputs = self.processor(
                    text=text,
                    pointer_timestamps=pointer_timestamps,
                    frames_indices=pointer_data.get('frames_indices'),
                    padding=True,
                    return_tensors="pt",
                )
            else:
                # Standard image processing (fallback when pointer memory is disabled)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=text,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

            device = "cuda" if self.device_map == "auto" else self.device
            inputs = inputs.to(device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            # Add pointer memory to generation if available
            generation_kwargs = {
                **inputs,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": pad_token_id,
                "do_sample": True if gen_kwargs["temperature"] > 0 else False,
                "temperature": gen_kwargs["temperature"],
                "top_p": gen_kwargs["top_p"],
                "num_beams": gen_kwargs["num_beams"],
                "max_new_tokens": gen_kwargs["max_new_tokens"],
                "use_cache": self.use_cache,
            }

            if self.use_pointer_memory and pointer_data is not None:
                generation_kwargs["pointer_memory_embeds"] = pointer_data['pointer_memory_embeds'].to(device)
                generation_kwargs["pointer_positions"] = pointer_data['pointer_positions'].to(device)

                # Add deepstack_pointer_embeds if available
                if 'deepstack_pointer_embeds' in pointer_data and pointer_data['deepstack_pointer_embeds'] is not None:
                    generation_kwargs["deepstack_pointer_embeds"] = [
                        layer.to(device) for layer in pointer_data['deepstack_pointer_embeds']
                    ]
                else:
                    generation_kwargs["deepstack_pointer_embeds"] = None

                # Add memory_feat if available AND merge_memory_feat is enabled
                if self.merge_memory_feat and 'memory_feat' in pointer_data and pointer_data['memory_feat'] is not None:
                    generation_kwargs["memory_feat"] = pointer_data['memory_feat'].to(device)

            cont = self.model.generate(**generation_kwargs)

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
