#!/usr/bin/env python3
"""
Convert VSTIBENCH training data to vanilla (image-based) format for Qwen3-VL training.
Samples frames from posed_images directories instead of using video files.

All VSTIBench data is scannet-only.

Usage:
    python scripts/preprocess/convert_vstibench_vanilla.py
    python scripts/preprocess/convert_vstibench_vanilla.py --nframes 8
"""

import json
import argparse
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm


def get_scene_images(scannet_base, scene_name, nframes=32):
    """Get uniformly sampled image paths for a scannet scene.

    Returns:
        rel_paths: List of image paths relative to scannet_base, or None if unavailable.
    """
    base = Path(scannet_base)
    p = base / "scannet" / "posed_images" / scene_name
    if not p.exists():
        return None

    image_paths = natsorted(list(p.glob("*.jpg")))

    if len(image_paths) == 0:
        return None

    # Uniform sampling (same approach as demo)
    if len(image_paths) > nframes:
        step = len(image_paths) / nframes
        indices = [int(i * step) for i in range(nframes)]
        image_paths = [image_paths[idx] for idx in indices]

    # Store relative paths (relative to scannet_base = data/media)
    return [str(img.relative_to(base)) for img in image_paths]


def convert_sample(sample, scannet_base, nframes=32):
    """Convert a single VSTIBENCH sample to vanilla image-based format."""
    scene_name = sample.get("scene_name", "")
    if not scene_name:
        return None

    image_paths = get_scene_images(scannet_base, scene_name, nframes)
    if image_paths is None:
        return None

    # Build image tokens
    image_tokens = "<image>" * len(image_paths)

    # Process conversations
    new_conversations = []
    for conv in sample.get("conversations", []):
        new_conv = conv.copy()
        value = conv.get("value", "")

        if conv.get("from") == "human":
            # Remove existing visual tokens
            value = value.replace("<image>", "").replace("<video>", "").strip()
            # Add image tokens at the beginning
            new_conv["value"] = f"{image_tokens}\n{value}"

        new_conversations.append(new_conv)

    converted = {
        "conversations": new_conversations,
        "images": image_paths,
        "metadata": {
            "dataset": "vstibench",
            "question_type": sample.get("question_type", ""),
            "scene_id": scene_name,
            "original_id": sample.get("id", ""),
        },
    }
    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert VSTIBENCH to vanilla image-based format"
    )
    parser.add_argument(
        "--nframes",
        type=int,
        default=32,
        help="Number of frames to sample per scene (default: 32)",
    )
    parser.add_argument(
        "--scannet_base",
        type=str,
        default="data/media",
        help="Base dir for scannet (default: data/media, images at {base}/scannet/posed_images/{scene}/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/train/vstibench_train.json)",
    )
    args = parser.parse_args()

    source_dir = Path("data") / "VLM-3R-DATA" / "vstibench_train"

    # Collect all qa_*.json source files
    source_files = sorted(source_dir.glob("qa_*.json"))
    if not source_files:
        print(f"No qa_*.json files found in {source_dir}")
        return

    print(f"Scannet base: {args.scannet_base}")
    print(f"Frames per scene: {args.nframes}")
    print(f"Source files: {len(source_files)}")

    # Collect and convert all samples
    all_converted = []
    total_loaded = 0
    skipped_no_scene = 0
    skipped_no_images = 0

    for source_file in source_files:
        print(f"Loading: {source_file}")
        with open(source_file, "r") as f:
            data = json.load(f)

        total_loaded += len(data)
        print(f"  Entries: {len(data)}")

        for sample in tqdm(data, desc=f"Converting {source_file.name}"):
            result = convert_sample(sample, args.scannet_base, args.nframes)
            if result is not None:
                all_converted.append(result)
            elif not sample.get("scene_name"):
                skipped_no_scene += 1
            else:
                skipped_no_images += 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("data") / "train" / "vstibench_train.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(all_converted, f, indent=2)

    print(f"\nConversion Statistics:")
    print(f"  Total loaded:              {total_loaded}")
    print(f"  Skipped (no scene_name):   {skipped_no_scene}")
    print(f"  Skipped (no images found): {skipped_no_images}")
    print(f"  Total converted:           {len(all_converted)}")

    # Print a sample
    if len(all_converted) > 0:
        print("\n" + "=" * 60)
        print("SAMPLE CONVERSION")
        print("=" * 60)
        item = all_converted[0]
        conv_preview = item["conversations"][0]["value"][:200]
        print(f"  Question type: {item['metadata']['question_type']}")
        print(f"  Conversation: {conv_preview}...")
        print(f"  Images: [{item['images'][0]}, ..., {item['images'][-1]}]")
        print(f"  Num images: {len(item['images'])}")


if __name__ == "__main__":
    main()
