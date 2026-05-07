"""
Visualize attention maps between pointer (visual) tokens and generated text tokens
in Qwen3VLForConditionalGenerationWithPoint3R.

Usage:
    python scripts/demo/visualize_attention.py \
        --model_path Qwen/Qwen3-VL-4B-Instruct \
        --pointer_data_path ./data/demo_data/scene0000_01_32f_video_compact.pt \
        --query "Describe this scene." \
        --output_dir ./outputs/attention_vis/
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/demo")

from demo_point3r import load_models, preprocess_images, set_seed
from attention_utils import (
    identify_pointer_indices,
    extract_pointer_attention,
    build_pointer_grid,
)
from time import time


def run_models_with_attention(
    model,
    processor,
    pointer_data_path="./data/demo_data/pointer_data.pt",
    query="Describe this scene.",
    pointer_data=None,
    max_new_tokens=128,
):
    """Run model inference and capture attention weights.

    Returns:
        dict with keys: generated_text, attention_matrix, pointer_timestamps,
                        input_ids, generated_token_ids, generated_tokens_text
    """
    # Load pointer data
    if pointer_data is None:
        print(f"\nLoading pointer data from {pointer_data_path}...")
        pointer_data = torch.load(pointer_data_path, weights_only=True)
    else:
        print("\nUsing pre-loaded pointer data")

    print("\n" + "=" * 70)
    print("Generating with attention capture (eager mode)")
    print("=" * 70)
    stage_start = time()

    # Build messages
    messages_with_pointer = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"<|vision_start|>{processor.pointer_token}<|vision_end|>"},
                {"type": "text", "text": query},
            ],
        }
    ]

    text_pointer = processor.apply_chat_template(
        messages_with_pointer, tokenize=False, add_generation_prompt=True
    )
    inputs_pointer = processor(
        text=[text_pointer],
        pointer_timestamps=pointer_data["pointer_timestamps"],
        frames_indices=pointer_data.get("frames_indices"),
        padding=True,
        return_tensors="pt",
    )

    inputs_pointer = inputs_pointer.to(model.device)
    pointer_memory_embeds = pointer_data["pointer_memory_embeds"].to(model.device)
    pointer_positions = pointer_data["pointer_positions"].to(model.device)

    deepstack_pointer_embeds = None
    if "deepstack_image_embeds" in pointer_data:
        deepstack_pointer_embeds = [
            layer.to(model.device) for layer in pointer_data["deepstack_image_embeds"]
        ]

    pointer_timestamps = None
    if "pointer_timestamps" in pointer_data:
        pointer_timestamps = pointer_data["pointer_timestamps"].to(model.device)

    input_ids = inputs_pointer.input_ids[0]
    input_len = input_ids.shape[0]
    pointer_indices = identify_pointer_indices(input_ids, model.pointer_token_id)

    print(f"Input length: {input_len}")
    print(f"Pointer tokens: {len(pointer_indices)}")
    print(f"Generating up to {max_new_tokens} tokens...")

    # Generate with attention output
    with torch.inference_mode():
        outputs = model.generate(
            **inputs_pointer,
            pointer_memory_embeds=pointer_memory_embeds,
            pointer_positions=pointer_positions,
            deepstack_pointer_embeds=deepstack_pointer_embeds,
            pointer_timestamps=pointer_timestamps,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
        )

    # Decode generated text
    generated_ids = outputs.sequences[0][input_len:]
    generated_text = processor.batch_decode(
        [generated_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    generated_tokens_text = [
        processor.tokenizer.decode([tid], skip_special_tokens=False) for tid in generated_ids
    ]

    print(f"Generated: {generated_text}")

    # Extract pointer attention matrix
    print("Extracting pointer attention...")
    attention_matrix = extract_pointer_attention(outputs, pointer_indices, input_len)

    stage_end = time()
    print(f"Generation + attention extraction: {stage_end - stage_start:.2f}s")

    # Get pointer timestamps on CPU for grid building
    ptr_ts = pointer_data["pointer_timestamps"].cpu() if "pointer_timestamps" in pointer_data else None

    return {
        "generated_text": generated_text,
        "attention_matrix": attention_matrix,
        "pointer_timestamps": ptr_ts,
        "input_ids": input_ids.cpu(),
        "generated_token_ids": generated_ids.cpu(),
        "generated_tokens_text": generated_tokens_text,
    }


def visualize_global_heatmap(attention_matrix, generated_tokens_text, pointer_timestamps, output_path):
    """Panel 1: Global heatmap of text tokens (y) vs pointer tokens (x)."""
    fig, ax = plt.subplots(figsize=(16, max(6, len(generated_tokens_text) * 0.25)), dpi=150)

    data = attention_matrix.numpy()

    # Apply gamma correction for better visibility
    gamma = 0.5
    data_vis = np.power(np.clip(data, 0, None), gamma)

    im = ax.imshow(data_vis, aspect="auto", cmap="viridis", interpolation="nearest")

    # Y-axis: generated tokens
    ax.set_yticks(range(len(generated_tokens_text)))
    ax.set_yticklabels(
        [t.strip() if t.strip() else repr(t) for t in generated_tokens_text],
        fontsize=6,
    )

    # X-axis: pointer token indices, with frame boundaries
    if pointer_timestamps is not None:
        unique_frames = pointer_timestamps.unique(sorted=True)
        boundaries = []
        frame_centers = []
        offset = 0
        for f in unique_frames:
            count = (pointer_timestamps == f).sum().item()
            frame_centers.append(offset + count / 2)
            offset += count
            boundaries.append(offset - 0.5)
        # Draw frame boundaries
        for b in boundaries[:-1]:
            ax.axvline(x=b, color="white", linewidth=0.5, alpha=0.7)
        ax.set_xticks([c for c in frame_centers])
        ax.set_xticklabels([f"F{f.item()}" for f in unique_frames], fontsize=6)
    else:
        ax.set_xlabel("Pointer Token Index")

    ax.set_ylabel("Generated Tokens")
    ax.set_title("Attention: Generated Tokens -> Pointer Tokens (gamma=0.5)")
    plt.colorbar(im, ax=ax, shrink=0.5, label="Attention Weight")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved global heatmap to {output_path}")


def visualize_per_token_grids(attention_matrix, generated_tokens_text, pointer_timestamps, output_path, top_k=16):
    """Panel 2: Per-token grid heatmaps for top-k tokens with highest pointer attention."""
    if pointer_timestamps is None:
        print("Skipping per-token grids (no pointer_timestamps)")
        return

    # Find tokens with highest total attention to pointers
    total_attn = attention_matrix.sum(dim=1)
    top_indices = total_attn.argsort(descending=True)[:top_k]
    top_indices = top_indices.sort().values  # Keep generation order

    cols = 4
    rows = (len(top_indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), dpi=150)
    if rows == 1:
        axes = axes[np.newaxis, :]

    for i, ax in enumerate(axes.flatten()):
        if i >= len(top_indices):
            ax.axis("off")
            continue

        idx = top_indices[i].item()
        attn_weights = attention_matrix[idx]
        grid, frame_labels = build_pointer_grid(pointer_timestamps, attn_weights)

        # Mask NaN values for display
        masked_grid = np.ma.masked_invalid(grid)
        im = ax.imshow(masked_grid, aspect="auto", cmap="hot", interpolation="nearest")

        token_text = generated_tokens_text[idx].strip() or repr(generated_tokens_text[idx])
        ax.set_title(f"[{idx}] \"{token_text}\"", fontsize=8, pad=3)
        ax.set_ylabel("Frame", fontsize=7)
        ax.set_xlabel("Token in frame", fontsize=7)

        # Frame labels on y-axis
        if len(frame_labels) <= 32:
            ax.set_yticks(range(len(frame_labels)))
            ax.set_yticklabels([f"F{fl}" for fl in frame_labels], fontsize=5)
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=16))

        ax.tick_params(axis="x", labelsize=5)

    plt.suptitle("Per-Token Pointer Attention Grids (Frames x Tokens/Frame)", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved per-token grids to {output_path}")


def visualize_temporal_profile(attention_matrix, generated_tokens_text, pointer_timestamps, output_path):
    """Panel 3: Total attention to each frame over generation steps."""
    if pointer_timestamps is None:
        print("Skipping temporal profile (no pointer_timestamps)")
        return

    unique_frames = pointer_timestamps.unique(sorted=True)
    num_generated = attention_matrix.shape[0]

    # Compute per-frame attention for each generation step
    frame_attention = np.zeros((num_generated, len(unique_frames)))
    for fi, frame in enumerate(unique_frames):
        mask = pointer_timestamps == frame
        frame_attention[:, fi] = attention_matrix[:, mask].sum(dim=1).numpy()

    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    im = ax.imshow(frame_attention.T, aspect="auto", cmap="viridis", interpolation="nearest")

    ax.set_xlabel("Generated Token Position")
    ax.set_ylabel("Frame")

    # Y-axis: frame labels
    if len(unique_frames) <= 32:
        ax.set_yticks(range(len(unique_frames)))
        ax.set_yticklabels([f"F{f.item()}" for f in unique_frames], fontsize=6)
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=16))

    # X-axis: token labels (show every few)
    step = max(1, num_generated // 30)
    x_ticks = list(range(0, num_generated, step))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [generated_tokens_text[i].strip() or "." for i in x_ticks],
        rotation=60,
        fontsize=6,
        ha="right",
    )

    ax.set_title("Temporal Attention Profile: Frame Attention over Generation Steps")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Sum of Attention to Frame")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved temporal profile to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize attention maps for Point3R-LLM")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--pointer_data_path", type=str, required=True,
                        help="Path to pre-computed pointer data .pt file")
    parser.add_argument("--query", type=str, default="Describe this scene.")
    parser.add_argument("--output_dir", type=str, default="./outputs/attention_vis/")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--pointer_format", type=str, default="video")
    parser.add_argument("--use_merge", action="store_true", default=True)
    parser.add_argument("--top_k", type=int, default=16,
                        help="Number of top tokens to show in per-token grid view")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the query and metadata to the output folder
    with open(os.path.join(args.output_dir, "query.txt"), "w") as f:
        f.write(f"Query: {args.query}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Pointer data: {args.pointer_data_path}\n")
        f.write(f"Max new tokens: {args.max_new_tokens}\n")

    # Load model with eager attention for attention weight extraction
    print("Loading model with attn_implementation='eager'...")
    model, processor, min_pixels, max_pixels, _ = load_models(
        load_point3r=False,
        model_path=args.model_path,
        pointer_format=args.pointer_format,
        use_merge=args.use_merge,
        attn_implementation="eager",
    )

    # Run inference with attention capture
    result = run_models_with_attention(
        model=model,
        processor=processor,
        pointer_data_path=args.pointer_data_path,
        query=args.query,
        max_new_tokens=args.max_new_tokens,
    )

    # Generate visualizations
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)

    visualize_global_heatmap(
        result["attention_matrix"],
        result["generated_tokens_text"],
        result["pointer_timestamps"],
        os.path.join(args.output_dir, "attention_heatmap.png"),
    )

    visualize_per_token_grids(
        result["attention_matrix"],
        result["generated_tokens_text"],
        result["pointer_timestamps"],
        os.path.join(args.output_dir, "per_token_grids.png"),
        top_k=args.top_k,
    )

    visualize_temporal_profile(
        result["attention_matrix"],
        result["generated_tokens_text"],
        result["pointer_timestamps"],
        os.path.join(args.output_dir, "temporal_profile.png"),
    )

    # Append generated answer to query.txt
    with open(os.path.join(args.output_dir, "query.txt"), "a") as f:
        f.write(f"Answer: {result['generated_text']}\n")

    print(f"\nAll visualizations saved to {args.output_dir}")
    print(f"Generated text: {result['generated_text']}")


if __name__ == "__main__":
    main()
