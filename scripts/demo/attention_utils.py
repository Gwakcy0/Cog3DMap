"""
Utility functions for extracting and visualizing attention maps
from Qwen3VL + Point3R models.

Adapted from VLM-Visualizer (sample_codes/VLM-Visualizer/utils.py).
"""

import torch
import numpy as np
import cv2


def identify_pointer_indices(input_ids, pointer_token_id):
    """Find positions of pointer tokens in the input sequence.

    Args:
        input_ids: (seq_len,) tensor of token IDs
        pointer_token_id: the ID of the pointer token

    Returns:
        pointer_indices: (num_pointers,) tensor of positions
    """
    return torch.where(input_ids == pointer_token_id)[0]


def aggregate_qwen3vl_attention(step_attentions):
    """Average attention across layers and heads for one generation step.

    Follows VLM-Visualizer's null-attention practice: zeroes the attention
    to the first token (BOS) since it acts as an attention sink.

    Args:
        step_attentions: tuple of num_layers tensors, each (1, num_heads, q_len, kv_len)

    Returns:
        avg_attn: (kv_len,) averaged attention vector for the last query token
    """
    per_layer = []
    for layer_attn in step_attentions:
        # (1, num_heads, q_len, kv_len) -> (num_heads, q_len, kv_len)
        layer_attn = layer_attn.squeeze(0)
        # Average across heads, take last query position
        avg_over_heads = layer_attn.mean(dim=0)[-1]  # (kv_len,)
        per_layer.append(avg_over_heads.cpu())

    avg = torch.stack(per_layer).mean(dim=0)  # (kv_len,)
    # Zero out BOS attention (null attention pattern)
    avg[0] = 0.0
    # Re-normalize
    total = avg.sum()
    if total > 0:
        avg = avg / total
    return avg


def extract_pointer_attention(outputs, pointer_indices, input_len):
    """Extract attention from each generated token to pointer tokens.

    Args:
        outputs: GenerateDecoderOnlyOutput with attentions
        pointer_indices: (num_pointers,) tensor of pointer positions in input
        input_len: length of the input sequence (before generation)

    Returns:
        attention_matrix: (num_generated_tokens, num_pointer_tokens) tensor
    """
    num_generated = len(outputs.attentions)
    num_pointers = len(pointer_indices)
    attention_matrix = torch.zeros(num_generated, num_pointers)

    for gen_step, step_attns in enumerate(outputs.attentions):
        avg_attn = aggregate_qwen3vl_attention(step_attns)
        # Extract attention to pointer token positions only
        pointer_attn = avg_attn[pointer_indices.cpu()]
        # Re-normalize to show relative attention distribution over pointers
        ptr_sum = pointer_attn.sum()
        if ptr_sum > 0:
            pointer_attn = pointer_attn / ptr_sum
        attention_matrix[gen_step] = pointer_attn

    return attention_matrix


def build_pointer_grid(pointer_timestamps, attention_weights):
    """Arrange pointer attention weights in a frames x tokens_per_frame grid.

    Args:
        pointer_timestamps: (num_pointers,) tensor of frame indices
        attention_weights: (num_pointers,) attention weights for one generated token

    Returns:
        grid: 2D numpy array (num_frames, max_tokens_per_frame)
        frame_labels: list of frame index labels
    """
    unique_frames = pointer_timestamps.unique(sorted=True)
    tokens_per_frame = [(pointer_timestamps == f).sum().item() for f in unique_frames]
    max_tokens = max(tokens_per_frame)

    grid = np.full((len(unique_frames), max_tokens), np.nan)
    for i, frame in enumerate(unique_frames):
        mask = pointer_timestamps == frame
        frame_weights = attention_weights[mask]
        grid[i, :len(frame_weights)] = frame_weights.numpy()

    return grid, [f.item() for f in unique_frames]


def show_mask_on_image(img, mask):
    """Overlay attention heatmap on an image.

    Args:
        img: numpy array (H, W, 3) in uint8
        mask: numpy array (H, W) in [0, 1]

    Returns:
        cam: overlayed image (H, W, 3) uint8
        heatmap: raw heatmap (H, W, 3) uint8
    """
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    hm = np.float32(heatmap) / 255
    cam = hm + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), heatmap


def heterogenous_stack(vecs):
    """Pad vectors with zeros then stack (from VLM-Visualizer)."""
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])
