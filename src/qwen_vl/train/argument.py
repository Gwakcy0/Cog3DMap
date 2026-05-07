import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    use_pointer_memory: bool = field(default=False)
    use_preprocessed_input: bool = field(default=False)
    point3r_model_path: str = field(default="./cache/point3r_512.pth")
    pointer_memory_size: int = field(default=512)

    # Memory feature fusion parameters (Point3R)
    merge_memory_feat: bool = field(default=False)
    memory_fusion_method: str = field(default="add")
    memory_fusion_attention_heads: int = field(default=8)
    memory_fusion_dropout: float = field(default=0.1)
    memory_fusion_num_layers: int = field(default=1)
    memory_merger_hidden_dim: int = field(default=4096)
    memory_merger_type: str = field(default="mlp")

    # Independent training control for memory modules
    tune_feature_projector: bool = field(default=False)
    tune_memory_feature_projector: bool = field(default=False)
    tune_memory_feature_fusion: bool = field(default=False)

    # Pointer position encoding parameters
    use_pointer_position_encoding: bool = field(
        default=False,
        metadata={"help": "Enable learnable position encoding for pointer tokens using 3D coordinates"}
    )
    pointer_pos_hidden_dim: int = field(
        default=256,
        metadata={"help": "Hidden dimension for pointer position encoder MLP"}
    )
    tune_pointer_position_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to train the pointer position encoder"}
    )

    # Pointer token format
    pointer_format: str = field(
        default="video",
        metadata={"help": "Format for pointer tokens: 'image' (flat, no vision markers) or 'video' (grouped by timestamp with vision markers)"}
    )

    # Pointer data directory override
    pointer_dir_name: Optional[str] = field(
        default=None,
        metadata={"help": "Override 'pointer_memory' directory name in pointer data paths "
                          "(e.g., 'pointer_memory_qwen3vl_lambda0.0')"}
    )

    # Frame ID labels for pointer tokens
    add_frame_id: bool = field(
        default=False,
        metadata={"help": "Use <frame-N> labels instead of <X.X seconds> timestamps for pointer token groups"}
    )

    # RoPE ablation parameters for pointer tokens
    rope_mode: str = field(
        default="none",
        metadata={"help": "RoPE mode for pointer tokens: 'none', 'discrete', 'continuous', 'pointer_timestamp'"}
    )
    rope_position_range: int = field(
        default=128,
        metadata={"help": "Discretization range [0, range-1] for discrete RoPE mode"}
    )
    tune_rope3d_continuous: bool = field(
        default=True,
        metadata={"help": "Whether to train the RoPE3D continuous projectors"}
    )

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    max_samples: int = field(default=-1)
    shuffle: bool = field(default=True)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
