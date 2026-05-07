#!/usr/bin/env python3
"""
Convert VSIBENCH training data to Point3R format.
Supports ScanNet, ScanNet++, and ARKitScenes data sources.

Pointer memory paths:
- ScanNet: data/media/scannet/pointer_memory/{scene_name}.pt
- ScanNet++: data/media/scannetpp/pointer_memory/{scene_name}.pt
- ARKitScenes: data/media/arkitscenes/pointer_memory/{scene_name}.pt
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


# Mapping from data_source to pointer_memory path prefix
POINTER_MEMORY_PATHS = {
    "scannet": "scannet/pointer_memory",
    "scannetpp": "scannetpp/pointer_memory",
    "arkitscenes": "arkitscenes/pointer_memory",
}


def convert_sample(sample, num_pointer_tokens=1, enabled_sources=None):
    """Convert a single VSIBENCH sample to Point3R format.

    Args:
        sample: Original sample dict
        num_pointer_tokens: Number of pointer tokens to use
        enabled_sources: Set of enabled data sources (e.g., {"scannet", "scannetpp"})

    Returns:
        Converted sample dict or None if skipped
    """
    data_source = sample.get("data_source", "")
    scene_name = sample.get("scene_name", "")

    # Skip if data source is not enabled
    if enabled_sources and data_source not in enabled_sources:
        return None

    if not scene_name:
        return None

    # Get pointer memory path based on data source
    if data_source not in POINTER_MEMORY_PATHS:
        return None

    pointer_memory_prefix = POINTER_MEMORY_PATHS[data_source]

    # Build pointer sequence
    pointer_sequence = (
        "<|vision_start|>" +
        "<|pointer_pad|>" * num_pointer_tokens +
        "<|vision_end|>"
    )

    # Process conversations
    new_conversations = []
    for conv in sample.get("conversations", []):
        new_conv = conv.copy()
        value = conv.get("value", "")

        if conv.get("from") == "human":
            # Remove existing <image> token if present
            value = value.replace("<image>", "").strip()
            # Add pointer sequence at the beginning
            new_conv["value"] = f"{pointer_sequence}\n{value}"

        new_conversations.append(new_conv)

    converted = {
        "conversations": new_conversations,
        "pointer_data": f"{pointer_memory_prefix}/{scene_name}.pt",
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
        description="Convert VSIBENCH to Point3R format"
    )
    parser.add_argument(
        "--num_pointer_tokens",
        type=int,
        default=1,
        help="Number of pointer tokens per sample",
    )
    parser.add_argument(
        "--exclude_scannetpp",
        action="store_true",
        help="Exclude ScanNet++ data (included by default)",
    )
    parser.add_argument(
        "--exclude_arkitscenes",
        action="store_true",
        help="Exclude ARKitScenes data (included by default)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/train/vsibench_train_point3r.json)",
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

    print(f"Enabled data sources: {enabled_sources}")

    # Source files to process
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
        "skipped": 0,
        "no_scene_name": 0,
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
                sample,
                args.num_pointer_tokens,
                enabled_sources=enabled_sources
            )
            if result is not None:
                all_converted.append(result)
                data_source = sample.get("data_source", "unknown")
                if data_source in stats:
                    stats[data_source] += 1
            else:
                # Track why it was skipped
                if not sample.get("scene_name"):
                    stats["no_scene_name"] += 1
                else:
                    stats["skipped"] += 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = base / "train" / "vsibench_train_point3r.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(all_converted, f, indent=2)

    print(f"\nConversion Statistics:")
    print(f"  ScanNet:      {stats['scannet']}")
    print(f"  ScanNet++:    {stats['scannetpp']}")
    print(f"  ARKitScenes:  {stats['arkitscenes']}")
    print(f"  Skipped (disabled source): {stats['skipped']}")
    print(f"  Skipped (no scene_name):   {stats['no_scene_name']}")
    print(f"  Total converted: {len(all_converted)}")

    # Print samples from each data source
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
                print(json.dumps(item, indent=2)[:800])
                if len(shown_sources) >= 3:
                    break


if __name__ == "__main__":
    main()
