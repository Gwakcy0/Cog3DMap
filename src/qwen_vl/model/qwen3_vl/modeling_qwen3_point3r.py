import torch
import torch.nn as nn
from typing import Optional, Any, Tuple, List, Dict, Union

from .modeling_qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLCausalLMOutputWithPast
from .configuration_qwen3_vl import Qwen3VLConfig
from ..point3r.point3r import Point3R, Point3RConfig
from ..feature_fusion import FeatureFusionModule, FeatureFusionConfig, FeatureProjector
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.cache_utils import Cache


class Qwen3VLForConditionalGenerationWithPoint3R(Qwen3VLForConditionalGeneration):
    """
    Qwen3-VL model with Point3R 3D memory support.

    This class extends Qwen3VLForConditionalGeneration to support pointer memory embeddings
    from Point3R, enabling 3D scene understanding capabilities.

    The pointer memory embeddings are injected into the model via the same mechanism as
    image/video embeddings (masked_scatter), and position encoding is extended to 4D
    (temporal, height, width, depth) for pointer tokens.
    """

    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)

        # Pointer token ID obtained from tokenizer (set after tokenizer.add_special_tokens)
        self.pointer_token_id = getattr(config, 'pointer_token_id', None)

        # Initialize Point3R model if needed (for on-the-fly feature extraction)
        if getattr(config, 'use_pointer_memory', False) and not getattr(config, 'use_preprocessed_input', False):
            self._init_point3r_memory(config)

        # Initialize memory feature fusion modules if enabled
        if getattr(config, 'merge_memory_feat', False):
            self._init_memory_fusion(config)

        # Initialize pointer position encoder if enabled
        if getattr(config, 'use_pointer_position_encoding', False):
            self._init_pointer_position_encoder(config)

        # Initialize RoPE3D-Continuous encoder if enabled
        if getattr(config, 'rope_mode', 'none') == 'continuous':
            self._init_rope3d_continuous(config)

        self.post_init()

        # Apply custom weight initialization AFTER post_init() to prevent overwriting
        self._init_custom_weights()

    def _init_point3r_memory(self, config):
        """Initialize Point3R model for on-the-fly 3D feature extraction."""
        inf = float('inf')
        point3r_config = Point3RConfig(
            freeze='encoder',
            pos_embed='RoPE100',
            pos_embed_3d='RoPE3D100',
            pose_head=True,
            patch_embed_cls='ManyAR_PatchEmbed',
            img_size=(512, 512),
            head_type='dpt',
            output_mode='pts3d+pose',
            depth_mode=('exp', -inf, inf),
            conf_mode=('exp', 1, inf),
            pose_mode=('exp', -inf, inf),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            landscape_only=False
        )
        self.point3r_model = Point3R(point3r_config)
        self.point3r_model.eval()

    def _init_pointer_position_encoder(self, config):
        """Initialize learnable position encoder for pointer memory tokens.

        This encoder projects continuous 3D coordinates (xyz) to embedding space
        and adds the position encoding to pointer embeddings before LLM injection.
        """
        from ..feature_fusion import PointerPositionEncoder

        self.pointer_position_encoder = PointerPositionEncoder(
            coord_dim=3,  # xyz coordinates
            hidden_dim=getattr(config, 'pointer_pos_hidden_dim', 256),
            out_dim=config.text_config.hidden_size,  # Match LLM hidden size (3584 for Qwen3-VL)
        )

    def _init_rope3d_continuous(self, config):
        """Initialize RoPE3DContinuous encoder with output projector (like memory_feat fusion).

        This encoder applies continuous 3D rotary position encoding to pointer embeddings
        using the RoPE3DContinuous module from Point3R.

        RoPE3DContinuous preserves input dimension - it splits the embedding into 3 parts
        for x, y, z encoding and concatenates them back. No input projection needed.
        """
        from ..croco.pos_embed_con import RoPE3DContinuous

        hidden_size = config.text_config.hidden_size  # 3584 for Qwen3-VL

        # RoPE3DContinuous handles any dimension - it splits D into 3 parts:
        # x_token: [..., :D//3], y_token: [..., D//3:2*D//3], z_token: [..., 2*D//3:]
        # The remainder (D % 3) goes to z_token, so output size equals input size.
        self.rope3d_continuous = RoPE3DContinuous(freq=100.0, F0=1.0)

        # Output projector for learnable residual adaptation
        # Note: Custom weight init (small weights) is done in _init_custom_weights() AFTER post_init()
        self.rope3d_output_projector = nn.Linear(hidden_size, hidden_size)

    def _init_custom_weights(self):
        """Apply custom weight initialization for residual learning modules.

        Must be called AFTER post_init() to prevent overwriting by _init_weights().
        This ensures that modules using additive fusion start with small weights,
        allowing gradual learning without disrupting pretrained features.
        """
        with torch.no_grad():
            # RoPE3D continuous projector - small weights for residual learning
            if hasattr(self, 'rope3d_output_projector'):
                self.rope3d_output_projector.weight.data.mul_(0.001)
                self.rope3d_output_projector.bias.data.zero_()

            # Memory feature fusion module - custom initialization based on fusion method
            if hasattr(self, 'memory_feature_fusion'):
                if hasattr(self.memory_feature_fusion, 'fusion_method'):
                    if self.memory_feature_fusion.fusion_method == "weighted":
                        # Initialize weights to 0.5, 0.5 (balanced)
                        if hasattr(self.memory_feature_fusion, 'weight_2d'):
                            self.memory_feature_fusion.weight_2d.fill_(0.5)
                        if hasattr(self.memory_feature_fusion, 'weight_3d'):
                            self.memory_feature_fusion.weight_3d.fill_(0.5)
                    elif self.memory_feature_fusion.fusion_method in ["cross_attention", "self_attention"]:
                        # Scale down attention layers
                        for module in self.memory_feature_fusion.modules():
                            if isinstance(module, nn.Linear):
                                module.weight.data.mul_(0.01)
                                if module.bias is not None:
                                    module.bias.data.zero_()

    def _apply_continuous_rope(self, pointer_embeds, positions):
        """Apply RoPE3DContinuous and add to embeddings (like memory_feat fusion).

        Args:
            pointer_embeds: Pointer embeddings (num_pointers, hidden_size)
            positions: 3D positions for each pointer (num_pointers, 3) as [h, w, d]

        Returns:
            Pointer embeddings with continuous RoPE position encoding added
        """
        # Reshape for RoPE3DContinuous: expects (B, heads, ntokens, D)
        embeds = pointer_embeds.unsqueeze(0).unsqueeze(0)  # (1, 1, N, hidden_size)
        pos = positions.unsqueeze(0)  # (1, N, 3)

        # Apply continuous 3D RoPE directly - it preserves input dimension
        # RoPE3DContinuous splits embeds into x, y, z parts, applies RoPE1D to each,
        # and concatenates them back to the original dimension
        encoded = self.rope3d_continuous(embeds, pos)  # (1, 1, N, hidden_size)

        # Project for learnable adaptation and add (like memory_feat fusion)
        rope_features = self.rope3d_output_projector(encoded)
        rope_features = rope_features.squeeze(0).squeeze(0)  # (N, hidden_size)

        return pointer_embeds + rope_features  # Additive fusion

    def _init_memory_fusion(self, config):
        """Initialize memory feature fusion modules for Point3R memory_feat integration.

        This mirrors the pattern used in Qwen2_5_VLForConditionalGenerationWithPoint3R's
        _init_memory_fusion method.
        """
        memory_dim = 768  # Point3R dec_embed_dim
        output_dim = config.text_config.hidden_size

        self.memory_feature_projector = FeatureProjector(
            input_dim=memory_dim,
            output_dim=output_dim,
            hidden_dim=getattr(config, "memory_merger_hidden_dim", 4096),
        )

        # Create feature fusion module to combine memory_feat with pointer_memory_embeds
        fusion_config = FeatureFusionConfig(
            fusion_method=getattr(config, "memory_fusion_method", "add"),
            hidden_size=output_dim,
            num_heads=getattr(config, "memory_fusion_attention_heads", 8),
            dropout=getattr(config, "memory_fusion_dropout", 0.1),
            num_layers=getattr(config, "memory_fusion_num_layers", 1)
        )
        self.memory_feature_fusion = FeatureFusionModule(fusion_config)

    def _init_memory_fusion_as_residual(self):
        """DEPRECATED: Use _init_custom_weights() instead.

        This method is no longer called from __init__ because post_init() would
        overwrite the custom weights. The functionality has been moved to
        _init_custom_weights() which is called AFTER post_init().

        Original purpose: Initialize memory fusion modules to preserve input features initially,
        ensuring memory_feat fusion doesn't corrupt pointer embeddings when using pretrained checkpoints.
        """
        if not hasattr(self, 'memory_feature_projector') or not hasattr(self, 'memory_feature_fusion'):
            return

        with torch.no_grad():
            # Initialize memory feature merger with small weights
            for module in self.memory_feature_projector.modules():
                if isinstance(module, torch.nn.Linear):
                    # Scale down linear layer weights significantly
                    module.weight.data.mul_(0.001)
                    if module.bias is not None:
                        module.bias.data.zero_()

            # Initialize fusion module based on fusion method
            if hasattr(self.memory_feature_fusion, 'fusion_method'):
                if self.memory_feature_fusion.fusion_method == "weighted":
                    # Initialize weights to 0.5, 0.5 (balanced)
                    if hasattr(self.memory_feature_fusion, 'weight_2d'):
                        self.memory_feature_fusion.weight_2d.fill_(0.5)
                    if hasattr(self.memory_feature_fusion, 'weight_3d'):
                        self.memory_feature_fusion.weight_3d.fill_(0.5)
                elif self.memory_feature_fusion.fusion_method in ["cross_attention", "self_attention"]:
                    # Scale down attention layers
                    for module in self.memory_feature_fusion.modules():
                        if isinstance(module, torch.nn.Linear):
                            module.weight.data.mul_(0.01)
                            if module.bias is not None:
                                module.bias.data.zero_()

    def _process_memory_features(self, pointer_memory_embeds, memory_feat):
        """Process Point3R memory features and fuse with pointer embeddings.

        This mirrors the pattern used in Qwen2_5_VLForConditionalGenerationWithPoint3R's
        _process_memory_features method.

        Args:
            pointer_memory_embeds: Tensor of shape (num_pointers, text_hidden_size)
                Qwen-aligned image embeddings from Point3R
            memory_feat: Tensor of shape (num_pointers, 768)
                Point3R internal decoder features

        Returns:
            Tensor of shape (num_pointers, text_hidden_size)
                Fused features combining pointer embeddings with memory features
        """
        if memory_feat is None:
            return pointer_memory_embeds

        # Ensure memory_feat has correct dtype
        memory_feat = memory_feat.to(pointer_memory_embeds.dtype)

        # Step 1: Reshape memory_feat to add spatial dimensions for GeometryFeatureMerger
        # memory_feat shape: (num_pointers, 768)
        # GeometryFeatureMerger expects: (n_image, h_patch, w_patch, dim)
        # Since memory_feat is already token-level, we treat each token as a 1x1 spatial patch
        num_pointers, memory_dim = memory_feat.shape
        memory_feat_spatial = memory_feat.view(num_pointers, 1, 1, memory_dim)

        # Step 2: Apply merger to match dimensions (768 → text_hidden_size)
        # Output: (num_pointers, 1, 1, text_hidden_size)
        merged_memory = self.memory_feature_projector(memory_feat_spatial)

        # Step 3: Flatten back to token-level
        # (num_pointers, 1, 1, text_hidden_size) → (num_pointers, text_hidden_size)
        merged_memory = merged_memory.view(num_pointers, -1)

        # Step 4: Fuse with pointer_memory_embeds
        # Both tensors now have shape (num_pointers, text_hidden_size)
        fused_embeds = self.memory_feature_fusion(pointer_memory_embeds, merged_memory)

        return fused_embeds

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load model with optional Point3R checkpoint."""
        point3r_model_path = kwargs.pop("point3r_model_path", None)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if point3r_model_path:
            model.point3r_model = Point3R.from_pretrained(point3r_model_path)
            model.point3r_model.eval()
        return model

    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pointer_memory_embeds: Optional[torch.Tensor] = None,
        pointer_positions: Optional[torch.Tensor] = None,
        pointer_timestamps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate RoPE indices based on rope_mode configuration.

        rope_mode options:
        - "none": No special handling for pointers (parent behavior, 3D RoPE)
        - "discrete": Discretized 4D positions for pointer tokens
        - "continuous": Base 3D positions; continuous RoPE applied in forward
        - "pointer_timestamp": 3D RoPE treating pointer groups as visual tokens
        """
        rope_mode = getattr(self.config, 'rope_mode', 'none')

        if rope_mode == "discrete" and pointer_positions is not None:
            from ...data.rope2d import get_rope_index_qwen3vl_discrete
            return get_rope_index_qwen3vl_discrete(
                spatial_merge_size=2,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                pointer_memory_embeds=pointer_memory_embeds,
                pointer_positions=pointer_positions,
                position_range=getattr(self.config, 'rope_position_range', 128),
            )
        elif rope_mode == "pointer_timestamp" and pointer_timestamps is not None:
            from ...data.rope2d import get_rope_index_qwen3vl_pointer
            return get_rope_index_qwen3vl_pointer(
                spatial_merge_size=2,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                pointer_timestamps=pointer_timestamps,
            )
        else:
            # For "none" and "continuous" modes, use parent's get_rope_index (3D RoPE)
            # Continuous RoPE is applied separately in forward pass
            return self.model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        pointer_memory_embeds: Optional[torch.Tensor] = None,
        pointer_positions: Optional[torch.Tensor] = None,
        deepstack_pointer_embeds: Optional[List[torch.Tensor]] = None,
        memory_feat: Optional[torch.FloatTensor] = None,
        pointer_timestamps: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple | Qwen3VLCausalLMOutputWithPast:
        """
        Forward pass with pointer memory support.

        Args:
            pointer_memory_embeds: Point3R memory embeddings (num_pointers, hidden_size)
            pointer_positions: 3D positions for each pointer (num_pointers, 3) as [h, w, d]
            deepstack_pointer_embeds: List of intermediate layer embeddings for pointers
            memory_feat: Point3R internal decoder features (num_pointers, 768) for fusion
            pointer_timestamps: Per-token frame indices (num_pointers,) for timestamp-grouped RoPE
            ... (other args same as parent)
        """
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Handle image embeddings (same as parent but we need to do it here since we modified inputs_embeds)
        image_mask = None
        video_mask = None
        pointer_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None
        # deepstack_pointer_embeds is now a parameter

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.model.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if pointer_memory_embeds is not None:
            pointer_memory_embeds = pointer_memory_embeds.type(self.model.visual.dtype)
            n_pointer_tokens = (input_ids == self.pointer_token_id).sum().item()
            n_pointer_features = pointer_memory_embeds.shape[0]

            if n_pointer_tokens != n_pointer_features:
                raise ValueError(
                    f"Pointer memory features and pointer tokens do not match: "
                    f"tokens: {n_pointer_tokens}, features {n_pointer_features}"
                )

            # Fuse with memory_feat if enabled (before masked_scatter)
            if getattr(self.config, 'merge_memory_feat', False) and memory_feat is not None:
                pointer_memory_embeds = self._process_memory_features(pointer_memory_embeds, memory_feat)
            # Apply position encoding if enabled and positions available
            if hasattr(self, 'pointer_position_encoder') and pointer_positions is not None:
                pointer_memory_embeds = self.pointer_position_encoder(
                    pointer_memory_embeds, pointer_positions
                )

            # Apply RoPE3D-Continuous if enabled
            if hasattr(self, 'rope3d_continuous') and pointer_positions is not None:
                pointer_memory_embeds = self._apply_continuous_rope(
                    pointer_memory_embeds, pointer_positions
                )

            mask = input_ids == self.pointer_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            pointer_mask = mask_expanded.to(inputs_embeds.device)

            pointer_memory_embeds = pointer_memory_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(pointer_mask, pointer_memory_embeds)

        # Aggregate visual position masks and deepstack embeddings
        visual_pos_masks = None
        deepstack_visual_embeds = None

        # Reduce masks to 2D (batch, seq_len)
        if image_mask is not None:
            image_mask = image_mask[..., 0]
        if video_mask is not None:
            video_mask = video_mask[..., 0]
        if pointer_mask is not None:
            pointer_mask = pointer_mask[..., 0]

        # Combine all visual masks
        mask_list = [m for m in [image_mask, video_mask, pointer_mask] if m is not None]
        if mask_list:
            visual_pos_masks = mask_list[0]
            for m in mask_list[1:]:
                visual_pos_masks = visual_pos_masks | m

        # Aggregate deepstack embeddings from all sources
        deepstack_sources = [
            (image_mask, deepstack_image_embeds),
            (video_mask, deepstack_video_embeds),
            (pointer_mask, deepstack_pointer_embeds),
        ]
        active_sources = [(mask, embeds) for mask, embeds in deepstack_sources if mask is not None and embeds is not None]

        if visual_pos_masks is not None and active_sources:
            # Get number of layers from first available source
            num_layers = len(active_sources[0][1])
            hidden_dim = active_sources[0][1][0].shape[-1]
            device = active_sources[0][1][0].device
            dtype = active_sources[0][1][0].dtype

            deepstack_visual_embeds = []
            for layer_idx in range(num_layers):
                embed_joint = torch.zeros(visual_pos_masks.sum(), hidden_dim, device=device, dtype=dtype)

                for mask, embeds in active_sources:
                    mask_joint = mask[visual_pos_masks]
                    embed_joint[mask_joint] = embeds[layer_idx].to(device, dtype)

                deepstack_visual_embeds.append(embed_joint)

        # Calculate position IDs if not provided
        if position_ids is None:
            past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            if self.model.rope_deltas is None or past_key_values_length == 0:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask,
                    pointer_memory_embeds=pointer_memory_embeds,
                    pointer_positions=pointer_positions,
                    pointer_timestamps=pointer_timestamps,
                )
                self.model.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (past_key_values_length + self.model.rope_deltas).to(inputs_embeds.device)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                # Expand to match expected dimensions (always 3D for MRoPE)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # Forward through language model
        outputs = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Compute logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.model.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        pointer_memory_embeds=None,
        pointer_positions=None,
        deepstack_pointer_embeds=None,
        memory_feat=None,
        pointer_timestamps=None,
        **kwargs,
    ):
        """Prepare inputs for generation with pointer memory support."""
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            pointer_memory_embeds=pointer_memory_embeds,
            pointer_positions=pointer_positions,
            deepstack_pointer_embeds=deepstack_pointer_embeds,
            memory_feat=memory_feat,
            pointer_timestamps=pointer_timestamps,
            use_cache=use_cache,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        assert "pointer_memory_embeds" in model_inputs.keys(), "No pointer_memory_embeds"
        assert "pointer_positions" in model_inputs.keys(), "No pointer_positions"
        assert "deepstack_pointer_embeds" in model_inputs.keys(), "No deepstack_pointer_embeds"

        if cache_position[0] != 0:
            # After the prefill phase, vision and pointer inputs should not be forwarded
            # as they have already been processed and are stored in the KV cache
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["pointer_memory_embeds"] = None
            model_inputs["pointer_positions"] = None
            model_inputs["deepstack_pointer_embeds"] = None
            model_inputs["memory_feat"] = None
            model_inputs["pointer_timestamps"] = None
        return model_inputs

    def _get_pointer_nums(
        self,
        input_ids: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        """
        Get the number of pointer tokens for each sample in the batch.

        Args:
            input_ids: Input token IDs (batch_size, seq_length)

        Returns:
            pointer_nums: Number of pointer tokens per sample (batch_size,)
        """
        pointer_token_id = self.pointer_token_id
        if pointer_token_id is None:
            return torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        # Count all pointer tokens in each sample (not just those after vision_start)
        pointer_mask = input_ids == pointer_token_id
        pointer_nums = torch.sum(pointer_mask, dim=1)

        return pointer_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expand inputs for beam search with pointer memory support."""
        if expand_size == 1:
            return input_ids, model_kwargs

        # Add pointer keys to visual keys
        visual_keys = [
            "pixel_values", "image_grid_thw", "pixel_values_videos",
            "video_grid_thw", "pointer_memory_embeds", "pointer_positions",
            "deepstack_pointer_embeds", "memory_feat", "pointer_timestamps"
        ]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )
            pointer_nums = self._get_pointer_nums(input_ids)

            if video_grid_thw is not None:
                cumulative_frame_counts = torch.cumsum(video_grid_thw[:, 0], dim=0)
                cumulative_token_video_counts = torch.cumsum(video_nums, dim=0)
                video_boundary_indices = torch.searchsorted(cumulative_frame_counts, cumulative_token_video_counts)
                video_nums = torch.diff(torch.cat([-video_boundary_indices.new_ones(1), video_boundary_indices]))

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values" and dict_to_expand[key] is not None and image_grid_thw is not None:
                    samples = torch.split(image_grid_thw, list(image_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw" and dict_to_expand[key] is not None:
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos" and dict_to_expand[key] is not None and video_grid_thw is not None:
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw" and dict_to_expand[key] is not None:
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pointer_memory_embeds" and dict_to_expand[key] is not None:
                    lengths = list(pointer_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pointer_positions" and dict_to_expand[key] is not None:
                    lengths = list(pointer_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "deepstack_pointer_embeds" and dict_to_expand[key] is not None:
                    # deepstack_pointer_embeds is a list of tensors, expand each layer
                    lengths = list(pointer_nums)
                    dict_to_expand[key] = [
                        _repeat_interleave_samples(layer_embeds, lengths=lengths, repeat_times=expand_size)
                        for layer_embeds in dict_to_expand[key]
                    ]
                elif key == "memory_feat" and dict_to_expand[key] is not None:
                    # memory_feat has same structure as pointer_memory_embeds
                    lengths = list(pointer_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pointer_timestamps" and dict_to_expand[key] is not None:
                    # pointer_timestamps has same structure as pointer_memory_embeds
                    lengths = list(pointer_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )

            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        # Expand visual inputs (including pointer memory)
        if input_ids is not None and input_ids.numel() != 0:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

__all__ = ["Qwen3VLForConditionalGenerationWithPoint3R"]