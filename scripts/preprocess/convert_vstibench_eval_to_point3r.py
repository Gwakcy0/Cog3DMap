#!/usr/bin/env python3
"""
Convert VSTIBENCH evaluation data (test.json) to Point3R format.
"""

import json
import argparse
import re
from pathlib import Path
from tqdm import tqdm


def extract_scene_name(video_path):
    """Extract scene name from video path like 'ScanNet/videos/val/scene0568_00.mp4'"""
    match = re.search(r'(scene\d+_\d+)', video_path)
    if match:
        return match.group(1)
    return None


def convert_sample(sample, num_pointer_tokens=1):
    """Convert a single VSTIBENCH test sample to Point3R format."""
    video_path = sample.get("video_path", "")
    scene_name = extract_scene_name(video_path)
    question_type = sample.get("question_type", "")

    if not scene_name:
        return None

    # Build pointer sequence
    pointer_sequence = (
        "<|vision_start|>" +
        "<|pointer_pad|>" * num_pointer_tokens +
        "<|vision_end|>"
    )

    # Build the question text
    question = sample.get("question", "")
    options = sample.get("options", [])

    # Format similar to vsibench evaluation
    pre_prompt = "These are frames of a video."

    if options:
        # Multiple choice question
        options_text = "Options:\n" + "\n".join(options)
        post_prompt = "Answer with the option's letter from the given choices directly."
        full_question = f"{pre_prompt}\n{question}\n{options_text}\n{post_prompt}"
    else:
        # Numerical answer question
        post_prompt = "Please answer the question using a single word or phrase."
        full_question = f"{pre_prompt}\n{question}\n{post_prompt}"

    # Create conversations in Point3R format
    conversations = [
        {
            "from": "human",
            "value": f"{pointer_sequence}\n{full_question}"
        },
        {
            "from": "gpt",
            "value": str(sample.get("ground_truth", ""))
        }
    ]

    converted = {
        "id": sample.get("id"),
        "conversations": conversations,
        "pointer_data": f"scannet/pointer_memory/{scene_name}.pt",
        "ground_truth": sample.get("ground_truth"),
        "question_type": question_type,
        "question": question,
        "options": options,
        "mc_answer": sample.get("mc_answer"),
        "metadata": {
            "dataset": "vstibench",
            "scene_id": scene_name,
            "original_video_path": video_path,
        },
    }
    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert VSTIBENCH evaluation data to Point3R format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/vstibench/test.json",
        help="Input test.json file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation/vstibench_point3r/test.json",
        help="Output file path",
    )
    parser.add_argument(
        "--num_pointer_tokens",
        type=int,
        default=1,
        help="Number of pointer tokens per sample",
    )
    args = parser.parse_args()

    # Load input data
    print(f"Loading: {args.input}")
    with open(args.input, "r") as f:
        data = json.load(f)

    print(f"Total entries: {len(data)}")

    # Convert all samples
    converted = []
    skipped = 0
    stats_by_type = {}

    for sample in tqdm(data, desc="Converting"):
        result = convert_sample(sample, args.num_pointer_tokens)
        if result is not None:
            converted.append(result)
            qtype = sample.get("question_type", "unknown")
            stats_by_type[qtype] = stats_by_type.get(qtype, 0) + 1
        else:
            skipped += 1

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"\nConversion Statistics by Question Type:")
    for qtype, count in sorted(stats_by_type.items()):
        print(f"  {qtype}: {count}")
    print(f"  Skipped: {skipped}")
    print(f"  Total: {len(converted)}")

    # Print sample
    if len(converted) > 0:
        print("\n" + "=" * 60)
        print("SAMPLE CONVERSION (First entry)")
        print("=" * 60)
        print(json.dumps(converted[0], indent=2))


if __name__ == "__main__":
    main()
