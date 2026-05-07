#!/usr/bin/env python3
"""
Convert ScanQA dataset to Point3R format.
Produces both training and evaluation JSON files.
"""

import json
from pathlib import Path
from tqdm import tqdm


def convert_sample(sample):
    """Convert a single ScanQA sample to Point3R format."""
    scene_id = sample.get("scene_id")
    if not scene_id:
        return None
    
    # pre-prompt for 3D grounding dataset
    pre_prompt = "The video captures 3D spatial information of a scene. Please focus on the spatial relationships in the video and answer the following questions.\n"
    post_prompt = "Answer the question simply."
    pointer_token = "<|pointer_pad|>"

    question = sample["question"]
    # Use first answer for GPT response in training
    answers = sample.get("answers", [])
    if not answers:
        return None

    converted = {
        "conversations": [
            {
                "from": "human",
                "value": f"{pre_prompt}{pointer_token} {question} {post_prompt}",
            },
            {
                "from": "gpt",
                "value": answers[0],
            },
        ],
        "pointer_data": f"scannet/pointer_memory/{scene_id}.pt",
        "answers": answers,
        "metadata": {
            "dataset": "scanqa",
            "question_id": sample.get("question_id", ""),
            "scene_id": scene_id,
        },
    }
    return converted


def convert_file(input_path, output_path):
    """Convert a ScanQA JSON file to Point3R format."""
    print(f"Loading: {input_path}")
    with open(input_path, "r") as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")
    converted = []
    skipped = 0

    for sample in tqdm(data, desc="Converting"):
        result = convert_sample(sample)
        if result is not None:
            converted.append(result)
        else:
            skipped += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving: {output_path}")
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"  Converted: {len(converted)}, Skipped: {skipped}")
    return converted


def write_dataset_card(output_dir, splits):
    """Write a HuggingFace dataset card (README.md) with explicit split definitions."""
    data_files = "\n".join(
        f"  - split: {split}\n    path: {filename}" for split, filename in splits
    )
    readme = f"""---
license: apache-2.0
configs:
- config_name: default
  data_files:
{data_files}
---
"""
    with open(Path(output_dir) / "README.md", "w") as f:
        f.write(readme)


def main():
    base = Path("data")

    # Training set
    convert_file(
        base / "media" / "ScanQA" / "ScanQA_v1.0_train.json",
        base / "train" / "scanqa_train_point3r.json",
    )

    # Evaluation set (val, since test has no answers)
    eval_dir = base / "evaluation" / "scanqa_point3r"
    convert_file(
        base / "media" / "ScanQA" / "ScanQA_v1.0_val.json",
        eval_dir / "val.json",
    )
    write_dataset_card(eval_dir, [("val", "val.json")])


if __name__ == "__main__":
    main()
