#!/usr/bin/env python3
"""
Convert VSTIBENCH training data to Point3R format.
All VSTIBENCH data is from ScanNet and has pointer memory available.
"""

import json
import argparse
import os
from pathlib import Path
from tqdm import tqdm


def convert_sample(sample, num_pointer_tokens=1):
    """Convert a single VSTIBENCH sample to Point3R format."""
    scene_name = sample.get("scene_name", "")
    question_type = sample.get("question_type", "")

    if not scene_name:
        return None

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
        "pointer_data": f"scannet/pointer_memory/{scene_name}.pt",
        "metadata": {
            "dataset": "vstibench",
            "question_type": question_type,
            "scene_id": scene_name,
            "original_id": sample.get("id", ""),
        },
    }
    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert VSTIBENCH to Point3R format"
    )
    parser.add_argument(
        "--num_pointer_tokens",
        type=int,
        default=1,
        help="Number of pointer tokens per sample",
    )
    args = parser.parse_args()

    base = Path("data")
    source_dir = base / "VLM-3R-DATA" / "vstibench_train"

    # Collect and convert all samples from all JSON files
    all_converted = []
    stats_by_type = {}
    skipped = 0

    json_files = sorted([f for f in os.listdir(source_dir) if f.endswith(".json")])

    for json_file in json_files:
        input_path = source_dir / json_file
        print(f"Loading: {input_path}")

        with open(input_path, "r") as f:
            data = json.load(f)

        print(f"  Entries: {len(data)}")

        for sample in tqdm(data, desc=f"Converting {json_file}"):
            result = convert_sample(sample, args.num_pointer_tokens)
            if result is not None:
                all_converted.append(result)
                qtype = sample.get("question_type", "unknown")
                stats_by_type[qtype] = stats_by_type.get(qtype, 0) + 1
            else:
                skipped += 1

    # Save output
    output_path = base / "train" / "vstibench_train_point3r.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(all_converted, f, indent=2)

    print(f"\nConversion Statistics by Question Type:")
    for qtype, count in sorted(stats_by_type.items()):
        print(f"  {qtype}: {count}")
    print(f"  Skipped: {skipped}")
    print(f"  Total: {len(all_converted)}")

    # Print sample
    if len(all_converted) > 0:
        print("\n" + "=" * 60)
        print("SAMPLE CONVERSION (First entry)")
        print("=" * 60)
        print(json.dumps(all_converted[0], indent=2))


if __name__ == "__main__":
    main()
