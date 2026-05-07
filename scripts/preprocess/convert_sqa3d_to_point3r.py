#!/usr/bin/env python3
"""
Convert SQA3D dataset to Point3R format.
Merges separate question and annotation files, then produces
training and evaluation JSON files.
"""

import json
from pathlib import Path
from tqdm import tqdm

def get_sqa_question_type(question):
    question = question.lstrip()
    if question[:4].lower() == 'what':
        return 'what'
    elif question[:2].lower() == 'is':
        return 'is'
    elif question[:3].lower() == 'how':
        return 'how'
    elif question[:3].lower() == 'can':
        return 'can'
    elif question[:5].lower() == 'which':
        return 'which'
    else:
        return 'others'   # others

def load_questions_and_annotations(questions_path, annotations_path):
    """Load and merge SQA3D questions and annotations by question_id."""
    with open(questions_path, "r") as f:
        questions_data = json.load(f)
    with open(annotations_path, "r") as f:
        annotations_data = json.load(f)

    questions = questions_data["questions"]
    annotations = annotations_data["annotations"]

    # Build annotation lookup by question_id
    ann_lookup = {}
    for ann in annotations:
        ann_lookup[ann["question_id"]] = ann

    merged = []
    for q in questions:
        qid = q["question_id"]
        ann = ann_lookup.get(qid)
        if ann is None:
            continue
        merged.append({
            "scene_id": q["scene_id"],
            "situation": q["situation"],
            "question": q["question"],
            "question_id": qid,
            "answers": [a["answer"] for a in ann.get("answers", [])],
        })

    return merged


def convert_sample(sample):
    """Convert a single SQA3D sample to Point3R format."""
    scene_id = sample.get("scene_id")
    if not scene_id:
        return None

    answers = sample.get("answers", [])
    if not answers:
        return None

    # pre-prompt for 3D grounding dataset
    pre_prompt = "The video captures 3D spatial information of a scene. Please focus on the spatial relationships in the video and answer the following questions.\n"
    post_prompt = "Answer the question using a single word or phrase."
    pointer_token = "<|pointer_pad|>"

    situation = sample["situation"]
    question = sample["question"]
    prompt = f"{pre_prompt}\n{pointer_token} {situation} {question} {post_prompt}"

    converted = {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answers[0]},
        ],
        "pointer_data": f"scannet/pointer_memory/{scene_id}.pt",
        "answers": answers,
        "metadata": {
            "dataset": "sqa3d",
            "question_id": str(sample["question_id"]),
            "scene_id": scene_id,
        },
    }
    return converted


def convert_split(questions_path, annotations_path, output_path):
    """Convert one split of SQA3D to Point3R format."""
    print(f"Loading questions: {questions_path}")
    print(f"Loading annotations: {annotations_path}")
    merged = load_questions_and_annotations(questions_path, annotations_path)
    print(f"Merged samples: {len(merged)}")

    converted = []
    skipped = 0

    for sample in tqdm(merged, desc="Converting"):
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
    sqa3d_dir = base / "media" / "SQA3D" / "balanced"

    # Training set
    convert_split(
        sqa3d_dir / "v1_balanced_questions_train_scannetv2.json",
        sqa3d_dir / "v1_balanced_sqa_annotations_train_scannetv2.json",
        base / "train" / "sqa3d_train_point3r.json",
    )

    # Evaluation set (val)
    eval_dir = base / "evaluation" / "sqa3d_point3r"
    convert_split(
        sqa3d_dir / "v1_balanced_questions_val_scannetv2.json",
        sqa3d_dir / "v1_balanced_sqa_annotations_val_scannetv2.json",
        eval_dir / "val.json",
    )

    # Evaluation set (test)
    convert_split(
        sqa3d_dir / "v1_balanced_questions_test_scannetv2.json",
        sqa3d_dir / "v1_balanced_sqa_annotations_test_scannetv2.json",
        eval_dir / "test.json",
    )

    write_dataset_card(eval_dir, [("val", "val.json"), ("test", "test.json")])


if __name__ == "__main__":
    main()
