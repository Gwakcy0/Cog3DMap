"""Feature fusion modules for combining 2D and 3D features."""

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass


@dataclass
class FeatureFusionConfig:
    """Configuration for feature fusion."""
    fusion_method: str = "add"  # "add", "concat", "gated", "weighted", "cross_attention"
    hidden_size: int = 3584
    num_heads: int = 8
    dropout: float = 0.1
    num_layers: int = 1


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block with position encoding, MLP and residual connections."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Layer norms
        self.norm1_query = nn.LayerNorm(hidden_size)
        self.norm1_key = nn.LayerNorm(hidden_size)
        self.norm1_value = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention block without position embeddings.

        Args:
            query: Query tensor (batch, seq_len, hidden_size)
            key_value: Key/value tensor (batch, seq_len, hidden_size)
        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        # Normalize
        query_norm = self.norm1_query(query)
        key_norm = self.norm1_key(key_value)
        value_norm = self.norm1_value(key_value)

        # Cross-attention (no position embeddings)
        attn_output, _ = self.cross_attention(query_norm, key_norm, value_norm)

        # Residual + MLP
        x = query + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class FeatureFusionModule(nn.Module):
    """Enhanced feature fusion module with multiple fusion strategies."""
    
    def __init__(self, config: FeatureFusionConfig):
        super().__init__()
        self.config = config
        self.fusion_method = config.fusion_method
        self.hidden_size = config.hidden_size
        
        self._build_fusion_layers()
    
    def _build_fusion_layers(self):
        """Build fusion layers based on method."""
        if self.config.fusion_method == "concat":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
            
        elif self.config.fusion_method == "cross_attention":
            self.cross_attn_blocks = nn.ModuleList([
                CrossAttentionBlock(
                    self.hidden_size, 
                    self.config.num_heads, 
                    self.config.dropout
                ) 
                for _ in range(self.config.num_layers)
            ])

        elif self.config.fusion_method == "gated":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.gate_projection = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.Sigmoid()
            )
            
        elif self.config.fusion_method == "weighted":
            self.weight_2d = nn.Parameter(torch.tensor(0.5))
            self.weight_3d = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, pointer_embeds: torch.Tensor, memory_embeds: torch.Tensor) -> torch.Tensor:
        """
        Fuse pointer embeddings and memory embeddings.

        Args:
            pointer_embeds: Pointer memory embeddings (N, D)
            memory_embeds: Merged memory features (N, D)
        Returns:
            Fused embeddings (N, D)
        """
        if self.fusion_method == "add":
            return pointer_embeds + memory_embeds

        elif self.fusion_method == "concat":
            pointer_embeds_norm = self.norm1(pointer_embeds)
            memory_embeds_norm = self.norm2(memory_embeds)
            concat_features = torch.cat([pointer_embeds_norm, memory_embeds_norm], dim=-1)
            return self.projection(concat_features)

        elif self.fusion_method == "cross_attention":
            # Add batch dimension for attention: (N, D) -> (1, N, D)
            pointer_embeds_batched = pointer_embeds.unsqueeze(0)
            memory_embeds_batched = memory_embeds.unsqueeze(0)
            x = pointer_embeds_batched
            for block in self.cross_attn_blocks:
                x = block(x, memory_embeds_batched)
            return x.squeeze(0)  # (1, N, D) -> (N, D)

        elif self.fusion_method == "gated":
            pointer_embeds_norm = self.norm1(pointer_embeds)
            memory_embeds_norm = self.norm2(memory_embeds)
            concat_features = torch.cat([pointer_embeds_norm, memory_embeds_norm], dim=-1)
            gate = self.gate_projection(concat_features)
            return gate * pointer_embeds_norm + (1 - gate) * memory_embeds_norm

        elif self.fusion_method == "weighted":
            weight_sum = self.weight_2d + self.weight_3d
            norm_weight_ptr = self.weight_2d / weight_sum
            norm_weight_mem = self.weight_3d / weight_sum
            return norm_weight_ptr * pointer_embeds + norm_weight_mem * memory_embeds

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


class FeatureProjector(nn.Module):
    """Simple feature projector similar to Qwen3VLVisionPatchMerger.

    Projects features from input_dim to output_dim using:
    LayerNorm -> Linear -> GELU -> Linear

    Args:
        input_dim: Input feature dimension (d_1)
        output_dim: Output feature dimension (d_2)
        hidden_dim: Optional hidden dimension for the MLP. Defaults to input_dim.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim

        self.norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features from (N, d_1) to (N, d_2).

        Args:
            x: Input tensor of shape (N, input_dim)

        Returns:
            Output tensor of shape (N, output_dim)
        """
        x = self.norm(x)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class GeometryFeatureMerger(nn.Module):
    """Unified merger for geometry features from different encoders.
    
    Supports different merger types:
    - "mlp": MLP-based feature transformation with spatial merging
    - "avg": Average pooling across spatial merge dimensions
    - "attention": Attention-based merger (not implemented yet)
    """
    
    def __init__(self, output_dim: int, hidden_dim: int, context_dim: int, 
                 spatial_merge_size: int = 2, merger_type: str = "mlp"):
        super().__init__()
        self.merger_type = merger_type
        self.input_dim = context_dim * (spatial_merge_size ** 2)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.merge_size = spatial_merge_size
        
        if merger_type == "mlp":
            # Import here to avoid circular import
            try:
                from .modeling_qwen2_5_vl import Qwen2RMSNorm
            except ImportError:
                # Fallback to standard LayerNorm if Qwen2RMSNorm not available
                Qwen2RMSNorm = nn.LayerNorm
                
            self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        elif merger_type == "avg":
            self.mlp = nn.Sequential(
                nn.Linear(context_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        elif merger_type == "attention":
            # Add attention-based merger for future extensibility
            raise NotImplementedError("Attention merger not implemented yet")
        else:
            raise ValueError(f"Unknown merger type: {merger_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the merger."""

        n_image, h_patch, w_patch, dim = x.shape
        x = x[:, :h_patch // self.merge_size * self.merge_size, :w_patch // self.merge_size*self.merge_size , :]
        x = x.reshape(n_image, h_patch // self.merge_size, self.merge_size, w_patch // self.merge_size, self.merge_size, dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        if self.merger_type == "mlp":
            x = self.mlp(self.ln_q(x).view(-1, self.input_dim))
        elif self.merger_type == "avg":
            # Average pooling across spatial merge dimensions
            x = x.mean(dim=(3, 4))  # Average over the merge_size dimensions
            x = x.view(-1, dim)  # Flatten for projection
            x = self.mlp(x)
        else:
            raise NotImplementedError(f"Merger type {self.merger_type} not implemented")
        x = x.reshape(n_image, h_patch // self.merge_size, w_patch // self.merge_size, -1)
        return x


class PointerPositionEncoder(nn.Module):
    """
    Learnable position encoding for pointer memory tokens.
    Projects continuous 3D coordinates to embedding space.

    This encoder adds spatial position information to pointer embeddings
    by projecting their 3D world coordinates through a learnable MLP.
    The output is added to the pointer embeddings before injection into the LLM.

    Args:
        coord_dim: Dimension of input coordinates (default: 3 for xyz)
        hidden_dim: Hidden dimension of the MLP (default: 256)
        out_dim: Output dimension matching LLM hidden size (default: 3584 for Qwen3-VL)
    """

    def __init__(
        self,
        coord_dim: int = 3,
        hidden_dim: int = 256,
        out_dim: int = 3584,
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.pos_projector = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, pointer_embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Add position encoding to pointer embeddings.

        Args:
            pointer_embeds: Pointer memory embeddings [num_pointers, hidden_size]
            positions: Continuous 3D coordinates [num_pointers, 3] (xyz)

        Returns:
            encoded_embeds: Pointer embeddings with position info added [num_pointers, hidden_size]
        """
        # Project positions to embedding space (match dtype of weights)
        pos_encoding = self.pos_projector(positions.to(self.pos_projector[0].weight.dtype))

        # Add to pointer embeddings
        encoded_embeds = pointer_embeds + pos_encoding

        return encoded_embeds
