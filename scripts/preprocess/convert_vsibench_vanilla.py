#!/usr/bin/env python3
"""
Convert VSIBENCH training data to vanilla (image-based) format for Qwen3-VL training.
Samples frames from posed_images directories instead of using video files.

Usage:
    python scripts/preprocess/convert_vsibench_vanilla.py
    python scripts/preprocess/convert_vsibench_vanilla.py --nframes 8 --exclude_scannetpp --exclude_arkitscenes
"""

import json
import argparse
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm


# Image file extensions per data source
IMAGE_EXTENSIONS = {
    "scannet": ("*.jpg",),
    "scannetpp": ("*.JPG",),
    "arkitscenes": ("*.jpg", "*.jpeg", "*.png", "*.JPG"),
}

# Default image base directories per data source
DEFAULT_IMAGE_BASES = {
    "scannet": "data/media",      # {base}/scannet/posed_images/{scene_name}/*.jpg
    "scannetpp": "data/media",  # {base}/scannetpp/data/data/{scene_name}/dslr/resized_undistorted_images/*.JPG
    "arkitscenes": "data/media",   # {base}/arkitscenes/3dod/{Training|Validation}/{scene_id}/{scene_id}_frames/lowres_wide/
}


def _find_scene_image_dir(base_dir, data_source, scene_name):
    """Resolve the image directory for a scene given the data source."""
    base = Path(base_dir)

    if data_source == "scannet":
        # data/media/scannet/posed_images/{scene_name}/
        return base / "scannet" / "posed_images" / scene_name

    elif data_source == "scannetpp":
        # {base}/scannetpp/data/data/{scene_name}/dslr/resized_undistorted_images/
        return base / "scannetpp" / "data" / "data" / scene_name / "dslr" / "resized_undistorted_images"

    elif data_source == "arkitscenes":
        # arkitscenes/3dod/{Training|Validation}/{scene_id}/{scene_id}_frames/lowres_wide/
        for split in ("Training", "Validation"):
            p = base / "arkitscenes" / "3dod" / split / scene_name / f"{scene_name}_frames" / "lowres_wide"
            if p.exists():
                return p
        return None

    return None


def get_scene_images(image_bases, data_source, scene_name, nframes=32):
    """Get uniformly sampled image paths for a scene.

    Args:
        image_bases: Dict mapping data_source -> base directory path.
        data_source: One of "scannet", "scannetpp", "arkitscenes".
        scene_name: Scene identifier.
        nframes: Number of frames to uniformly sample.

    Returns:
        rel_paths: List of image paths relative to "data/media", or None if unavailable.
    """
    base_dir = image_bases.get(data_source)
    if not base_dir:
        return None

    p = _find_scene_image_dir(base_dir, data_source, scene_name)
    if p is None or not p.exists():
        return None

    extensions = IMAGE_EXTENSIONS.get(data_source, ("*.jpg",))
    image_paths = natsorted([f for ext in extensions for f in p.glob(ext)])

    if len(image_paths) == 0:
        return None

    # Uniform sampling (same approach as demo)
    if len(image_paths) > nframes:
        step = len(image_paths) / nframes
        indices = [int(i * step) for i in range(nframes)]
        image_paths = [image_paths[idx] for idx in indices]

    # For scannet (data_path="data/media"), store relative paths.
    # For external sources (scannetpp, arkitscenes), store absolute paths
    # so that os.path.join(data_path, abs_path) returns the absolute path.
    if data_source == "scannet":
        base = Path(base_dir)
        return [str(img.relative_to(base)) for img in image_paths]
    else:
        return [str(img.resolve()) for img in image_paths]


def convert_sample(sample, image_bases, nframes=32, enabled_sources=None):
    """Convert a single VSIBENCH sample to vanilla image-based format."""
    data_source = sample.get("data_source", "")
    scene_name = sample.get("scene_name", "")

    if enabled_sources and data_source not in enabled_sources:
        return None

    if not scene_name:
        return None

    image_paths = get_scene_images(image_bases, data_source, scene_name, nframes)
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
            "dataset": "vsibench",
            "data_source": data_source,
            "scene_id": scene_name,
            "original_id": sample.get("id", ""),
        },
    }
    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert VSIBENCH to vanilla image-based format"
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
        default=DEFAULT_IMAGE_BASES["scannet"],
        help="Base dir for scannet (default: data/media, images at {base}/scannet/posed_images/{scene}/)",
    )
    parser.add_argument(
        "--exclude_scannetpp",
        action="store_true",
        help="Exclude ScanNet++ data (included by default)",
    )
    parser.add_argument(
        "--scannetpp_base",
        type=str,
        default=DEFAULT_IMAGE_BASES["scannetpp"],
        help="Base dir for scannetpp (default: /data/llm_data, images at {base}/scannetpp/data/data/{scene}/dslr/resized_undistorted_images/)",
    )
    parser.add_argument(
        "--exclude_arkitscenes",
        action="store_true",
        help="Exclude ARKitScenes data (included by default)",
    )
    parser.add_argument(
        "--arkitscenes_base",
        type=str,
        default=DEFAULT_IMAGE_BASES["arkitscenes"],
        help="Base dir for arkitscenes (default: data/media, images at {base}/arkitscenes/3dod/{Training|Validation}/{scene}/...)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/train/vsibench_train.json)",
    )
    args = parser.parse_args()

    base = Path("data")
    source_dir = base / "VLM-3R-DATA" / "vsibench_train"

    # Determine which data sources to include (all enabled by default)
    enabled_sources = {"scannet", "scannetpp", "arkitscenes"}
    if args.exclude_scannetpp:
        enabled_sources.discard("scannetpp")
    if args.exclude_arkitscenes:
        enabled_sources.discard("arkitscenes")

    # Build image base directories
    image_bases = {
        "scannet": args.scannet_base,
        "scannetpp": args.scannetpp_base,
        "arkitscenes": args.arkitscenes_base,
    }

    print(f"Enabled data sources: {enabled_sources}")
    print(f"Frames per scene: {args.nframes}")
    for src in enabled_sources:
        print(f"  {src} base: {image_bases[src]}")

    # Source files to process (no duplicates)
    source_files = [
        "merged_qa_scannet_train.json",
        "merged_qa_route_plan_train.json",  # Contains scannet, scannetpp, arkitscenes
    ]

    if "scannetpp" in enabled_sources:
        source_files.append("merged_qa_scannetpp_train.json")

    # Collect and convert all samples
    all_converted = []
    stats = {
        "scannet": 0,
        "scannetpp": 0,
        "arkitscenes": 0,
        "skipped_source": 0,
        "skipped_no_scene": 0,
        "skipped_no_images": 0,
    }

    for source_file in source_files:
        input_path = source_dir / source_file
        if not input_path.exists():
            print(f"Warning: {input_path} not found, skipping")
            continue

        print(f"Loading: {input_path}")
        with open(input_path, "r") as f:
            data = json.load(f)

        print(f"  Entries: {len(data)}")

        for sample in tqdm(data, desc=f"Converting {source_file}"):
            result = convert_sample(
                sample, image_bases, args.nframes,
                enabled_sources=enabled_sources,
            )
            if result is not None:
                all_converted.append(result)
                data_source = sample.get("data_source", "unknown")
                if data_source in stats:
                    stats[data_source] += 1
            else:
                if not sample.get("scene_name"):
                    stats["skipped_no_scene"] += 1
                elif sample.get("data_source") not in enabled_sources:
                    stats["skipped_source"] += 1
                else:
                    stats["skipped_no_images"] += 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = base / "train" / "vsibench_train.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(all_converted, f, indent=2)

    print(f"\nConversion Statistics:")
    print(f"  ScanNet:      {stats['scannet']}")
    print(f"  ScanNet++:    {stats['scannetpp']}")
    print(f"  ARKitScenes:  {stats['arkitscenes']}")
    print(f"  Skipped (disabled source): {stats['skipped_source']}")
    print(f"  Skipped (no scene_name):   {stats['skipped_no_scene']}")
    print(f"  Skipped (no images found): {stats['skipped_no_images']}")
    print(f"  Total converted: {len(all_converted)}")

    # Print sample from each data source
    if len(all_converted) > 0:
        print("\n" + "=" * 60)
        print("SAMPLE CONVERSIONS")
        print("=" * 60)

        shown_sources = set()
        for item in all_converted:
            source = item["metadata"]["data_source"]
            if source not in shown_sources:
                shown_sources.add(source)
                print(f"\n--- {source.upper()} ---")
                # Show conversation (truncated) and first/last image paths
                conv_preview = item["conversations"][0]["value"][:200]
                print(f"  Conversation: {conv_preview}...")
                print(f"  Images: [{item['images'][0]}, ..., {item['images'][-1]}]")
                print(f"  Num images: {len(item['images'])}")
                if len(shown_sources) >= 3:
                    break


if __name__ == "__main__":
    main()
