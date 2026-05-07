#!/usr/bin/env python3
"""
Convert SPAR-subset training and evaluation data to Point3R format.
Processes only "fill" variant from ScanNet and ScanNet++ data.
Converts free-form appearance order questions to VSI-bench MCA format
with A/B/C/D multiple-choice options.

Source data: data/SPAR-subset/train/*_fill.jsonl and data/SPAR-subset/val/*_fill.jsonl
Training output: data/train/spar_subset_point3r.json
Evaluation output: data/evaluation/spar_subset_point3r/test.json

Pointer memory paths:
- ScanNet: scannet/pointer_memory/{scene_name}.pt
- ScanNet++: scannetpp/pointer_memory/{scene_name}.pt
"""

import json
import argparse
import os
import random
from pathlib import Path
from tqdm import tqdm


POINTER_MEMORY_PATHS = {
    "scannet": "scannet/pointer_memory",
    "scannetpp": "scannetpp/pointer_memory",
}

OPTION_LETTERS = ["A", "B", "C", "D"]

QUESTION_TEMPLATE = (
    "What will be the first-time appearance order of the following "
    "categories in the video: {objects}?"
)


def detect_data_source(filename):
    """Detect data source (scannet/scannetpp) from filename."""
    if filename.startswith("scannetpp_"):
        return "scannetpp"
    elif filename.startswith("scannet_"):
        return "scannet"
    return None


def is_fill_variant(filename):
    """Check if file is a fill variant (skip sentence variant)."""
    return "_fill." in filename or "_fill_" in filename


def extract_scene_name(sample):
    """Extract scene name from image paths."""
    images = sample.get("image", [])
    if images:
        return images[0].split("/")[0]
    # Fallback: use id with rsplit
    sample_id = sample.get("id", "")
    if sample_id:
        return sample_id.rsplit("_", 1)[0]
    return None


def extract_objects_from_answer(answer):
    """Extract object list from comma-separated answer string."""
    return [obj.strip() for obj in answer.split(",") if obj.strip()]


def generate_distractors(correct_order, rng, num_distractors=3):
    """Generate unique distractor permutations different from the correct order."""
    distractors = []
    attempts = 0
    max_attempts = 100
    while len(distractors) < num_distractors and attempts < max_attempts:
        shuffled = list(correct_order)
        rng.shuffle(shuffled)
        if shuffled != correct_order and shuffled not in distractors:
            distractors.append(shuffled)
        attempts += 1
    return distractors


def build_mcq(correct_order, sample_id):
    """Build MCQ options and assign correct answer to a random position.

    Returns:
        (question_text, options_list, correct_letter, objects_str)
    """
    rng = random.Random(hash(sample_id))

    objects = list(correct_order)
    rng.shuffle(objects)
    objects_str = ", ".join(objects)
    question = QUESTION_TEMPLATE.format(objects=objects_str)

    distractors = generate_distractors(correct_order, rng)

    # Place correct answer at a random position
    correct_idx = rng.randint(0, 3)
    all_options = []
    distractor_iter = iter(distractors)
    for i in range(4):
        if i == correct_idx:
            all_options.append(correct_order)
        else:
            all_options.append(next(distractor_iter))

    correct_letter = OPTION_LETTERS[correct_idx]

    options_formatted = [
        f"{OPTION_LETTERS[i]}. {', '.join(opt)}"
        for i, opt in enumerate(all_options)
    ]

    return question, options_formatted, correct_letter, objects_str


def build_conversation_value(question, options_formatted):
    """Build the full human conversation value with pointer tokens."""
    options_text = "Options:\n" + "\n".join(options_formatted)
    full_text = (
        "<|vision_start|><|pointer_pad|><|vision_end|>\n"
        f"These are frames of a video.\n"
        f"{question}\n"
        f"{options_text}\n"
        "Answer with the option's letter from the given choices directly."
    )
    return full_text


def convert_train_sample(sample, data_source):
    """Convert a single SPAR-subset sample to Point3R MCA training format."""
    scene_name = extract_scene_name(sample)
    if not scene_name:
        return None

    if data_source not in POINTER_MEMORY_PATHS:
        return None

    # Extract answer
    gpt_answer = ""
    for conv in sample.get("conversations", []):
        if conv.get("from") == "gpt":
            gpt_answer = conv.get("value", "")
            break

    objects = extract_objects_from_answer(gpt_answer)
    if len(objects) < 2:
        return None

    sample_id = sample.get("id", "")
    question, options_formatted, correct_letter, _ = build_mcq(objects, sample_id)

    pointer_memory_prefix = POINTER_MEMORY_PATHS[data_source]

    conversations = [
        {
            "from": "human",
            "value": build_conversation_value(question, options_formatted),
        },
        {
            "from": "gpt",
            "value": correct_letter,
        },
    ]

    converted = {
        "conversations": conversations,
        "pointer_data": f"{pointer_memory_prefix}/{scene_name}.pt",
        "metadata": {
            "dataset": "spar_subset",
            "data_source": data_source,
            "scene_id": scene_name,
            "original_id": sample_id,
        },
    }
    return converted


def convert_eval_sample(sample, data_source):
    """Convert a single SPAR-subset sample to Point3R MCA evaluation format."""
    scene_name = extract_scene_name(sample)
    if not scene_name:
        return None

    if data_source not in POINTER_MEMORY_PATHS:
        return None

    # Extract answer
    gpt_answer = ""
    for conv in sample.get("conversations", []):
        if conv.get("from") == "gpt":
            gpt_answer = conv.get("value", "")
            break

    objects = extract_objects_from_answer(gpt_answer)
    if len(objects) < 2:
        return None

    sample_id = sample.get("id", "")
    question, options_formatted, correct_letter, _ = build_mcq(objects, sample_id)

    pointer_memory_prefix = POINTER_MEMORY_PATHS[data_source]

    conversations = [
        {
            "from": "human",
            "value": build_conversation_value(question, options_formatted),
        },
        {
            "from": "gpt",
            "value": correct_letter,
        },
    ]

    converted = {
        "id": sample_id,
        "conversations": conversations,
        "pointer_data": f"{pointer_memory_prefix}/{scene_name}.pt",
        "ground_truth": correct_letter,
        "question_type": "obj_appearance_order",
        "question": question,
        "options": options_formatted,
        "metadata": {
            "dataset": "spar_subset",
            "data_source": data_source,
            "scene_id": scene_name,
        },
    }
    return converted


def load_jsonl(filepath):
    """Load a JSONL file and return a list of dicts."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def process_split(source_dir, split_name, convert_fn, enabled_sources):
    """Process fill-variant JSONL files in a split directory."""
    all_converted = []
    stats = {
        "scannet": 0,
        "scannetpp": 0,
        "skipped": 0,
    }

    jsonl_files = sorted([
        f for f in os.listdir(source_dir) if f.endswith(".jsonl")
    ])

    for jsonl_file in jsonl_files:
        data_source = detect_data_source(jsonl_file)

        if data_source is None:
            print(f"Warning: Could not detect source from {jsonl_file}, skipping")
            continue

        # Only process fill variant
        if not is_fill_variant(jsonl_file):
            print(f"Skipping {jsonl_file} (sentence variant)")
            continue

        if enabled_sources and data_source not in enabled_sources:
            print(f"Skipping {jsonl_file} (source {data_source} not enabled)")
            continue

        input_path = source_dir / jsonl_file
        print(f"Loading: {input_path}")
        data = load_jsonl(input_path)
        print(f"  Entries: {len(data)}")

        for sample in tqdm(data, desc=f"Converting {jsonl_file}"):
            result = convert_fn(sample, data_source)
            if result is not None:
                all_converted.append(result)
                if data_source in stats:
                    stats[data_source] += 1
            else:
                stats["skipped"] += 1

    return all_converted, stats


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
    parser = argparse.ArgumentParser(
        description="Convert SPAR-subset to Point3R MCA format (train + eval)"
    )
    parser.add_argument(
        "--exclude_scannetpp",
        action="store_true",
        help="Exclude ScanNet++ data (included by default)",
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default=None,
        help="Training output path (default: data/train/spar_subset_point3r.json)",
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default=None,
        help="Eval output path (default: data/evaluation/spar_subset_point3r/test.json)",
    )
    parser.add_argument(
        "--eval_max_per_source",
        type=int,
        default=1500,
        help="Max eval samples per data source. Set 0 for no limit. (default: 1500)",
    )
    args = parser.parse_args()

    base = Path("data")
    train_dir = base / "SPAR-subset" / "train"
    val_dir = base / "SPAR-subset" / "val"

    # Determine which data sources to include
    enabled_sources = {"scannet", "scannetpp"}
    if args.exclude_scannetpp:
        enabled_sources.discard("scannetpp")

    print(f"Enabled data sources: {enabled_sources}")

    # --- Process training data ---
    print("\n" + "=" * 60)
    print("PROCESSING TRAINING DATA")
    print("=" * 60)

    train_converted, train_stats = process_split(
        train_dir, "train", convert_train_sample, enabled_sources,
    )

    train_output = Path(args.train_output) if args.train_output else (
        base / "train" / "spar_subset_point3r.json"
    )
    train_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving training data to: {train_output}")
    with open(train_output, "w") as f:
        json.dump(train_converted, f, indent=2)

    print(f"\nTraining Conversion Statistics:")
    for key, count in sorted(train_stats.items()):
        print(f"  {key}: {count}")
    print(f"  Total converted: {len(train_converted)}")

    # --- Process evaluation data ---
    print("\n" + "=" * 60)
    print("PROCESSING EVALUATION DATA")
    print("=" * 60)

    eval_converted, eval_stats = process_split(
        val_dir, "val", convert_eval_sample, enabled_sources,
    )

    # Subsample evaluation set per data source
    if args.eval_max_per_source > 0:
        rng = random.Random(42)
        by_source = {}
        for entry in eval_converted:
            ds = entry["metadata"]["data_source"]
            by_source.setdefault(ds, []).append(entry)

        subsampled = []
        eval_stats_sub = {"skipped": 0}
        for ds, entries in sorted(by_source.items()):
            if len(entries) > args.eval_max_per_source:
                entries = rng.sample(entries, args.eval_max_per_source)
            subsampled.extend(entries)
            eval_stats_sub[ds] = len(entries)

        print(f"\nSubsampled eval set ({args.eval_max_per_source} per source):")
        print(f"  Before: {len(eval_converted)}, After: {len(subsampled)}")
        eval_converted = subsampled
        eval_stats = eval_stats_sub

    eval_output = Path(args.eval_output) if args.eval_output else (
        base / "evaluation" / "spar_subset_point3r" / "test.json"
    )
    eval_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving evaluation data to: {eval_output}")
    with open(eval_output, "w") as f:
        json.dump(eval_converted, f, indent=2)

    write_dataset_card(eval_output.parent, [("test", eval_output.name)])

    print(f"\nEvaluation Conversion Statistics:")
    for key, count in sorted(eval_stats.items()):
        print(f"  {key}: {count}")
    print(f"  Total converted: {len(eval_converted)}")

    # --- Print samples ---
    for label, data in [("TRAINING", train_converted), ("EVALUATION", eval_converted)]:
        if len(data) > 0:
            print(f"\n{'=' * 60}")
            print(f"SAMPLE {label} CONVERSIONS")
            print("=" * 60)
            shown = set()
            for item in data:
                ds = item.get("metadata", {}).get("data_source", "")
                if ds not in shown:
                    shown.add(ds)
                    print(f"\n--- {ds.upper()} ---")
                    print(json.dumps(item, indent=2)[:1000])
                    if len(shown) >= 2:
                        break


if __name__ == "__main__":
    main()
