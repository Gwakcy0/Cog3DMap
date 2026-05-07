"""Unified pointer data processing utilities for Point3R-LLM.

This module provides shared functions for loading and processing pointer memory data
that can be used by both training (LazySupervisedDataset) and evaluation (Point3RLLMv2).
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class PointerDataConfig:
    """Configuration for pointer data processing."""

    max_pointer_tokens: int = 8000
    pointer_token: str = "<|pointer_pad|>"


def load_pointer_data(
    pointer_data_path: str,
    base_dir: Optional[str] = None,
    max_pointer_tokens: int = 8000,
    pointer_dir_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Load pointer memory data from a .pt file.

    This is the canonical function for loading pointer data. Both training and
    evaluation should use this to ensure consistent handling.

    Args:
        pointer_data_path: Path to .pt file (can be relative or absolute)
        base_dir: Base directory to prepend if path is relative
        max_pointer_tokens: Maximum number of pointer tokens (truncate if exceeded)
        pointer_dir_name: If set, replace "pointer_memory" in the path with this value.
            Allows switching to alternative pre-computed pointer data directories
            (e.g., "pointer_memory_qwen3vl_lambda0.0") without modifying annotation files.

    Returns:
        Dictionary with keys:
        - 'pointer_memory_embeds': torch.Tensor (N, D)
        - 'pointer_positions': torch.Tensor (N, 3)
        - 'deepstack_pointer_embeds': Optional[List[torch.Tensor]]
        - 'memory_feat': Optional[torch.Tensor]
        - 'pointer_timestamps': Optional[torch.Tensor]
        - 'num_pointers': int

    Raises:
        FileNotFoundError: If pointer data file doesn't exist
    """
    # Resolve path
    if base_dir is not None and not os.path.isabs(pointer_data_path):
        full_path = os.path.join(base_dir, pointer_data_path)
    else:
        full_path = pointer_data_path

    # Substitute pointer directory name if specified
    if pointer_dir_name is not None:
        full_path = full_path.replace("pointer_memory", pointer_dir_name)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Pointer data file not found: {full_path}")

    # Load data
    pointer_data = torch.load(full_path, weights_only=True)

    # Extract required fields
    pointer_memory_embeds = pointer_data["pointer_memory_embeds"]
    pointer_positions = pointer_data["pointer_positions"]

    # Get number of pointers
    num_pointers = pointer_memory_embeds.shape[0]

    # Extract optional fields
    deepstack_pointer_embeds = pointer_data.get("deepstack_image_embeds")
    memory_feat = pointer_data.get("memory_feat")
    pointer_timestamps = pointer_data.get("pointer_timestamps")
    frames_indices = pointer_data.get("frames_indices")

    # Randomly sample if exceeding max tokens
    if num_pointers > max_pointer_tokens:
        logger.debug(
            f"Randomly sampling pointer data from {num_pointers} to {max_pointer_tokens} tokens"
        )
        indices = torch.randperm(num_pointers)[:max_pointer_tokens].sort().values
        num_pointers = max_pointer_tokens
        pointer_memory_embeds = pointer_memory_embeds[indices].clone()
        pointer_positions = pointer_positions[indices].clone()

        if deepstack_pointer_embeds is not None:
            deepstack_pointer_embeds = [
                d[indices].clone() for d in deepstack_pointer_embeds
            ]

        if memory_feat is not None:
            memory_feat = memory_feat[indices].clone()

        if pointer_timestamps is not None:
            pointer_timestamps = pointer_timestamps[indices].clone()

    # Free original data to save memory
    del pointer_data

    return {
        "pointer_memory_embeds": pointer_memory_embeds,
        "pointer_positions": pointer_positions,
        "deepstack_pointer_embeds": deepstack_pointer_embeds,
        "memory_feat": memory_feat,
        "pointer_timestamps": pointer_timestamps,
        "frames_indices": frames_indices,
        "num_pointers": num_pointers,
    }


def expand_pointer_tokens(
    content: str,
    num_pointer_tokens: int,
    pointer_token: str = "<|pointer_pad|>",
) -> str:
    """Expand a single pointer placeholder to multiple pointer tokens.

    In annotations, we use a single <|pointer_pad|> as a placeholder.
    This function expands it to N tokens based on the actual pointer count.

    Args:
        content: Text content containing pointer token placeholder
        num_pointer_tokens: Number of tokens to expand to
        pointer_token: The pointer token string

    Returns:
        Content with expanded pointer tokens

    Example:
        >>> expand_pointer_tokens("Here is <|pointer_pad|> the scene", 512)
        "Here is <|pointer_pad|><|pointer_pad|>...(512 times) the scene"
    """
    if pointer_token in content:
        return content.replace(pointer_token, pointer_token * num_pointer_tokens)
    return content


def expand_pointer_tokens_grouped(
    content: str,
    pointer_timestamps: "torch.Tensor",
    frames_indices: Optional[List[int]] = None,
    pointer_fps: Optional[float] = None,
    pointer_token: str = "<|pointer_pad|>",
    vision_start_token: str = "<|vision_start|>",
    vision_end_token: str = "<|vision_end|>",
    add_frame_id: bool = False,
) -> str:
    """Expand a single pointer placeholder to timestamp-grouped pointer tokens.

    Groups pointer tokens by their timestamp and wraps each group with
    timestamp headers and vision markers, matching the video token format.

    Args:
        content: Text content containing pointer token placeholder
        pointer_timestamps: Per-token frame indices (N,) sorted ascending
        frames_indices: Original image indices that were sampled (optional)
        pointer_fps: Frames per second for timestamp conversion.
            If None, defaults to 24.0 when frames_indices is provided
            (matching Qwen3VL video fallback), or 1.0 otherwise.
        pointer_token: The pointer token string
        vision_start_token: Vision start marker
        vision_end_token: Vision end marker
        add_frame_id: If True, use <frame-N> (1-indexed) instead of <X.X seconds>

    Returns:
        Content with grouped pointer tokens

    Example:
        >>> ts = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
        >>> expand_pointer_tokens_grouped("Here <|pointer_pad|> scene", ts)
        "Here <0.0 seconds><|vision_start|><|pointer_pad|>*3<|vision_end|><1.0 seconds>... scene"
        >>> expand_pointer_tokens_grouped("Here <|pointer_pad|> scene", ts, add_frame_id=True)
        "Here <frame-1><|vision_start|><|pointer_pad|>*3<|vision_end|><frame-2>... scene"
    """
    if pointer_token not in content:
        return content

    # Auto-select FPS: use 24.0 when frames_indices maps to original frame
    # numbers (like video), or 1.0 when using raw sampled-frame indices
    if pointer_fps is None:
        pointer_fps = 24.0 if frames_indices is not None else 1.0

    # Group tokens by timestamp
    unique_timestamps = pointer_timestamps.unique(sorted=True)
    grouped_placeholder = ""
    for frame_idx, ts in enumerate(unique_timestamps, start=1):
        count = (pointer_timestamps == ts).sum().item()

        if add_frame_id:
            grouped_placeholder += f"<frame-{frame_idx}>"
        else:
            ts_val = ts.item()

            # Convert frame index to seconds
            if frames_indices is not None and int(ts_val) < len(frames_indices):
                time_seconds = frames_indices[int(ts_val)] / pointer_fps
            else:
                time_seconds = ts_val / pointer_fps

            grouped_placeholder += f"<{time_seconds:.1f} seconds>"

        grouped_placeholder += (
            vision_start_token + pointer_token * count + vision_end_token
        )

    # Handle both wrapped and bare pointer token patterns
    wrapped = f"{vision_start_token}{pointer_token}{vision_end_token}"
    if wrapped in content:
        return content.replace(wrapped, grouped_placeholder, 1)
    return content.replace(pointer_token, grouped_placeholder, 1)


def prepare_pointer_batch(
    instances: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Collate pointer data from multiple instances into a batch.

    This is the shared collation logic for pointer memory data.

    Args:
        instances: List of data dictionaries, each containing pointer fields

    Returns:
        Batched tensor dictionary with concatenated pointer data
    """
    batch: Dict[str, Any] = {}

    # Check if any instance has pointer data
    if not any("pointer_memory_embeds" in inst for inst in instances):
        return batch

    # Filter instances that have pointer data
    pointer_instances = [inst for inst in instances if "pointer_memory_embeds" in inst]

    if not pointer_instances:
        return batch

    # Concatenate pointer memory embeds
    if "pointer_memory_embeds" in pointer_instances[0]:
        pointer_embeds = [inst["pointer_memory_embeds"] for inst in pointer_instances]
        batch["pointer_memory_embeds"] = torch.cat(pointer_embeds, dim=0)

    # Concatenate pointer positions
    if "pointer_positions" in pointer_instances[0]:
        positions = [inst["pointer_positions"] for inst in pointer_instances]
        batch["pointer_positions"] = torch.cat(positions, dim=0)

    # Concatenate deepstack embeddings (per-layer)
    if (
        "deepstack_pointer_embeds" in pointer_instances[0]
        and pointer_instances[0]["deepstack_pointer_embeds"] is not None
    ):
        num_layers = len(pointer_instances[0]["deepstack_pointer_embeds"])
        batch["deepstack_pointer_embeds"] = [
            torch.cat(
                [inst["deepstack_pointer_embeds"][i] for inst in pointer_instances],
                dim=0,
            )
            for i in range(num_layers)
        ]

    # Concatenate memory features
    if (
        "memory_feat" in pointer_instances[0]
        and pointer_instances[0]["memory_feat"] is not None
    ):
        batch["memory_feat"] = torch.cat(
            [inst["memory_feat"] for inst in pointer_instances], dim=0
        )

    # Concatenate pointer timestamps
    if (
        "pointer_timestamps" in pointer_instances[0]
        and pointer_instances[0]["pointer_timestamps"] is not None
    ):
        batch["pointer_timestamps"] = torch.cat(
            [inst["pointer_timestamps"] for inst in pointer_instances], dim=0
        )

    return batch
