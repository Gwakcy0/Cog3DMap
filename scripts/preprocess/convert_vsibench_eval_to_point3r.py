#!/usr/bin/env python3
"""
Convert VSIBENCH evaluation data to Point3R format.

This script converts VSI-Bench evaluation data from HuggingFace format to Point3R format.
The HuggingFace dataset fields are:
- dataset: data source (e.g., "scannet", "scannetpp", "arkitscenes")
- scene_name: scene identifier
- question_type: type of question (MCA or NA)
- question: the question text
- options: list of options for MCA questions
- ground_truth: the answer

Pointer memory paths:
- ScanNet: data/media/scannet/pointer_memory/{scene_name}.pt
- ScanNet++: data/media/scannetpp/pointer_memory/{scene_name}.pt
- ARKitScenes: data/media/arkitscenes/pointer_memory/{scene_name}.pt
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Mapping from dataset field to pointer_memory path prefix
POINTER_MEMORY_PATHS = {
    "scannet": "scannet/pointer_memory",
    "scannetpp": "scannetpp/pointer_memory",
    "arkitscenes": "arkitscenes/pointer_memory",
}

# Question types
MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]

NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]


def convert_sample(sample, num_pointer_tokens=1, enabled_sources=None):
    """Convert a single VSIBENCH sample to Point3R format."""
    data_source = sample.get("dataset", "")
    scene_name = sample.get("scene_name", "")
    question_type = sample.get("question_type", "")

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

    # Build question text
    question = sample.get("question", "")
    options = sample.get("options", [])
    ground_truth = sample.get("ground_truth", "")

    pre_prompt = "These are frames of a video."

    if question_type in NA_QUESTION_TYPES or not options:
        post_prompt = "Please answer the question using a single word or phrase."
        full_question = f"{pre_prompt}\n{question}\n{post_prompt}"
    else:
        options_text = "Options:\n" + "\n".join(options)
        post_prompt = "Answer with the option's letter from the given choices directly."
        full_question = f"{pre_prompt}\n{question}\n{options_text}\n{post_prompt}"

    # Create conversations in Point3R format
    conversations = [
        {
            "from": "human",
            "value": f"{pointer_sequence}\n{full_question}"
        },
        {
            "from": "gpt",
            "value": str(ground_truth)
        }
    ]

    converted = {
        "id": sample.get("id", sample.get("idx", "")),
        "conversations": conversations,
        "pointer_data": f"{pointer_memory_prefix}/{scene_name}.pt",
        "ground_truth": ground_truth,
        "question_type": question_type,
        "question": question,
        "options": options,
        "metadata": {
            "dataset": "vsibench",
            "data_source": data_source,
            "scene_id": scene_name,
        },
    }
    return converted


def load_from_huggingface(dataset_name="nyu-visionx/VSI-Bench", split="test"):
    """Load dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        dataset = load_dataset(dataset_name, split=split)
        return list(dataset)
    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        return None


def load_from_json(json_path):
    """Load dataset from local JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Convert VSIBENCH evaluation data to Point3R format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input JSON file path (if not provided, downloads from HuggingFace)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation/vsibench_point3r/test.json",
        help="Output file path",
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
        "--hf_dataset",
        type=str,
        default="nyu-visionx/VSI-Bench",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    args = parser.parse_args()

    # Determine which data sources to include (all enabled by default)
    enabled_sources = {"scannet", "scannetpp", "arkitscenes"}
    if args.exclude_scannetpp:
        enabled_sources.discard("scannetpp")
    if args.exclude_arkitscenes:
        enabled_sources.discard("arkitscenes")

    print(f"Enabled data sources: {enabled_sources}")

    # Load data
    if args.input:
        print(f"Loading from local file: {args.input}")
        data = load_from_json(args.input)
    else:
        print(f"Loading from HuggingFace: {args.hf_dataset} (split: {args.split})")
        data = load_from_huggingface(args.hf_dataset, args.split)

    if data is None:
        print("Failed to load data")
        return

    print(f"Total entries: {len(data)}")

    # Convert all samples
    converted = []
    stats = {
        "scannet": 0,
        "scannetpp": 0,
        "arkitscenes": 0,
        "skipped": 0,
    }

    for sample in tqdm(data, desc="Converting"):
        result = convert_sample(
            sample,
            args.num_pointer_tokens,
            enabled_sources=enabled_sources
        )
        if result is not None:
            converted.append(result)
            data_source = sample.get("dataset", "unknown")
            if data_source in stats:
                stats[data_source] += 1
        else:
            stats["skipped"] += 1

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"\nConversion Statistics:")
    print(f"  ScanNet:      {stats['scannet']}")
    print(f"  ScanNet++:    {stats['scannetpp']}")
    print(f"  ARKitScenes:  {stats['arkitscenes']}")
    print(f"  Skipped:      {stats['skipped']}")
    print(f"  Total:        {len(converted)}")

    # Print sample
    if len(converted) > 0:
        print("\n" + "=" * 60)
        print("SAMPLE CONVERSION (First entry)")
        print("=" * 60)
        print(json.dumps(converted[0], indent=2)[:1000])


if __name__ == "__main__":
    main()
