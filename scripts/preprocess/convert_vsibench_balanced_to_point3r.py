#!/usr/bin/env python3
"""
Convert VSI-bench + SPAR-subset training data into a single integrated
Point3R training dataset, globally shuffled.

Data sources:
  VSI-bench (data/VLM-3R-DATA/vsibench_train/):
    - merged_qa_scannet_train.json       (scannet)
    - merged_qa_scannetpp_train.json     (scannetpp)
    - merged_qa_route_plan_train.json    (scannet/scannetpp/arkitscenes)

  SPAR-subset (data/SPAR-subset/train/):
    - scannet_train_appr_order_fill.jsonl     (scannet, fill variant only)
    - scannetpp_train_appr_order_fill.jsonl   (scannetpp, fill variant only)

Output: data/train/vsibench_train_balanced_point3r.json

Pointer memory paths:
  - scannet:      scannet/pointer_memory/{scene_name}.pt
  - scannetpp:    scannetpp/pointer_memory/{scene_name}.pt
  - arkitscenes:  arkitscenes/pointer_memory/{scene_name}.pt
"""

import json
import argparse
import os
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


# ── Pointer memory path mapping ──────────────────────────────────────────────

POINTER_MEMORY_PATHS = {
    "scannet": "scannet/pointer_memory",
    "scannetpp": "scannetpp/pointer_memory",
    "arkitscenes": "arkitscenes/pointer_memory",
}

# ── SPAR-subset MCQ constants ───────────────────────────────────────────────

OPTION_LETTERS = ["A", "B", "C", "D"]

QUESTION_TEMPLATE = (
    "What will be the first-time appearance order of the following "
    "categories in the video: {objects}?"
)


# ── Question type inference (for stats only) ────────────────────────────────

def infer_question_type(text):
    """Infer VSI-bench question type from conversation text."""
    t = text.lower()
    if "how many" in t:
        return "counting"
    if "what is the direct distance" in t:
        return "abs_dist"
    if "longest dimension" in t:
        return "obj_size"
    if "what is the size of this room" in t:
        return "room_size"
    if "which of these objects" in t or "is the closest to" in t:
        return "rel_dist"
    if ("to the left or the right of" in t or "to my left" in t
            or "front-left" in t):
        return "rel_dir"
    return "unknown"


# ── VSI-bench conversion ────────────────────────────────────────────────────

def convert_vsibench_sample(sample, num_pointer_tokens=1):
    """Convert a single VSI-bench sample to Point3R format."""
    data_source = sample.get("data_source", "")
    scene_name = sample.get("scene_name", "")

    if not scene_name or data_source not in POINTER_MEMORY_PATHS:
        return None

    pointer_memory_prefix = POINTER_MEMORY_PATHS[data_source]

    pointer_sequence = (
        "<|vision_start|>"
        + "<|pointer_pad|>" * num_pointer_tokens
        + "<|vision_end|>"
    )

    new_conversations = []
    for conv in sample.get("conversations", []):
        new_conv = conv.copy()
        value = conv.get("value", "")
        if conv.get("from") == "human":
            value = value.replace("<image>", "").strip()
            new_conv["value"] = f"{pointer_sequence}\n{value}"
        new_conversations.append(new_conv)

    # Determine question type (always infer from text for consistency)
    human_text = next(
        (c["value"] for c in sample.get("conversations", [])
         if c.get("from") == "human"), ""
    )
    question_type = sample.get("question_type", "") or infer_question_type(human_text)
    # Normalize: explicit "object_abs_distance" → "abs_dist"
    if question_type == "object_abs_distance":
        question_type = "abs_dist"

    return {
        "conversations": new_conversations,
        "pointer_data": f"{pointer_memory_prefix}/{scene_name}.pt",
        "metadata": {
            "dataset": "vsibench",
            "data_source": data_source,
            "scene_id": scene_name,
            "original_id": sample.get("id", ""),
            "question_type": question_type,
        },
    }


# ── SPAR-subset conversion ──────────────────────────────────────────────────

def extract_objects_from_answer(answer):
    """Extract object list from comma-separated answer string."""
    return [obj.strip() for obj in answer.split(",") if obj.strip()]


def generate_distractors(correct_order, rng, num_distractors=3):
    """Generate unique distractor permutations different from correct order."""
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
    """Build MCQ options and assign correct answer to a random position."""
    rng = random.Random(hash(sample_id))

    objects = list(correct_order)
    rng.shuffle(objects)
    objects_str = ", ".join(objects)
    question = QUESTION_TEMPLATE.format(objects=objects_str)

    distractors = generate_distractors(correct_order, rng)

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


def extract_scene_name_spar(sample):
    """Extract scene name from SPAR-subset image paths."""
    images = sample.get("image", [])
    if images:
        return images[0].split("/")[0]
    sample_id = sample.get("id", "")
    if sample_id:
        return sample_id.rsplit("_", 1)[0]
    return None


def detect_spar_data_source(filename):
    """Detect data source from SPAR-subset filename."""
    if filename.startswith("scannetpp_"):
        return "scannetpp"
    elif filename.startswith("scannet_"):
        return "scannet"
    return None


def convert_spar_sample(sample, data_source):
    """Convert a single SPAR-subset sample to Point3R MCA training format."""
    scene_name = extract_scene_name_spar(sample)
    if not scene_name or data_source not in POINTER_MEMORY_PATHS:
        return None

    gpt_answer = ""
    for conv in sample.get("conversations", []):
        if conv.get("from") == "gpt":
            gpt_answer = conv.get("value", "")
            break

    objects = extract_objects_from_answer(gpt_answer)
    if len(objects) < 2:
        return None

    sample_id = sample.get("id", "")
    question, options_formatted, correct_letter, _ = build_mcq(
        objects, sample_id
    )

    pointer_memory_prefix = POINTER_MEMORY_PATHS[data_source]

    options_text = "Options:\n" + "\n".join(options_formatted)
    full_text = (
        "<|vision_start|><|pointer_pad|><|vision_end|>\n"
        "These are frames of a video.\n"
        f"{question}\n"
        f"{options_text}\n"
        "Answer with the option's letter from the given choices directly."
    )

    return {
        "conversations": [
            {"from": "human", "value": full_text},
            {"from": "gpt", "value": correct_letter},
        ],
        "pointer_data": f"{pointer_memory_prefix}/{scene_name}.pt",
        "metadata": {
            "dataset": "spar_subset",
            "data_source": data_source,
            "scene_id": scene_name,
            "original_id": sample_id,
            "question_type": "appearance_order",
        },
    }


# ── File I/O ────────────────────────────────────────────────────────────────

def load_jsonl(filepath):
    """Load a JSONL file and return a list of dicts."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert VSI-bench + SPAR-subset into a single integrated "
            "Point3R training dataset"
        )
    )
    parser.add_argument(
        "--exclude_scannetpp",
        action="store_true",
        help="Exclude ScanNet++ data",
    )
    parser.add_argument(
        "--exclude_arkitscenes",
        action="store_true",
        help="Exclude ARKitScenes route planning data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: data/train/vsibench_train_balanced_point3r.json)",
    )
    args = parser.parse_args()

    base = Path("data")

    # Determine enabled data sources
    enabled_sources = {"scannet", "scannetpp", "arkitscenes"}
    if args.exclude_scannetpp:
        enabled_sources.discard("scannetpp")
    if args.exclude_arkitscenes:
        enabled_sources.discard("arkitscenes")

    print(f"Enabled data sources: {enabled_sources}")

    all_converted = []
    # Stats: (data_source, question_type) -> count
    stats = defaultdict(int)
    skipped = 0

    # ── 1. Process VSI-bench data ────────────────────────────────────────

    print("\n" + "=" * 60)
    print("PROCESSING VSI-BENCH DATA")
    print("=" * 60)

    vsibench_dir = base / "VLM-3R-DATA" / "vsibench_train"

    vsibench_files = [
        "merged_qa_scannet_train.json",
        "merged_qa_route_plan_train.json",
    ]
    if "scannetpp" in enabled_sources:
        vsibench_files.append("merged_qa_scannetpp_train.json")

    for source_file in vsibench_files:
        input_path = vsibench_dir / source_file
        if not input_path.exists():
            print(f"Warning: {input_path} not found, skipping")
            continue

        print(f"Loading: {input_path}")
        with open(input_path, "r") as f:
            data = json.load(f)
        print(f"  Entries: {len(data)}")

        for sample in tqdm(data, desc=f"Converting {source_file}"):
            data_source = sample.get("data_source", "")
            if enabled_sources and data_source not in enabled_sources:
                skipped += 1
                continue

            result = convert_vsibench_sample(sample)
            if result is not None:
                all_converted.append(result)
                qtype = result["metadata"]["question_type"]
                stats[(data_source, qtype)] += 1
            else:
                skipped += 1

    # ── 2. Process SPAR-subset data ──────────────────────────────────────

    print("\n" + "=" * 60)
    print("PROCESSING SPAR-SUBSET DATA")
    print("=" * 60)

    spar_dir = base / "SPAR-subset" / "train"

    if spar_dir.exists():
        jsonl_files = sorted([
            f for f in os.listdir(spar_dir)
            if f.endswith(".jsonl") and "_fill" in f
        ])

        for jsonl_file in jsonl_files:
            data_source = detect_spar_data_source(jsonl_file)
            if data_source is None:
                print(f"Warning: Could not detect source from {jsonl_file}")
                continue
            if data_source not in enabled_sources:
                print(f"Skipping {jsonl_file} (source not enabled)")
                continue

            input_path = spar_dir / jsonl_file
            print(f"Loading: {input_path}")
            data = load_jsonl(input_path)
            print(f"  Entries: {len(data)}")

            for sample in tqdm(data, desc=f"Converting {jsonl_file}"):
                result = convert_spar_sample(sample, data_source)
                if result is not None:
                    all_converted.append(result)
                    stats[(data_source, "appearance_order")] += 1
                else:
                    skipped += 1
    else:
        print(f"Warning: {spar_dir} not found, skipping SPAR-subset")

    # ── 3. Save ───────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("FINALIZING")
    print("=" * 60)

    output_path = Path(args.output) if args.output else (
        base / "train" / "vsibench_train_balanced_point3r.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {len(all_converted)} samples to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(all_converted, f, indent=2)

    # ── 4. Print statistics ──────────────────────────────────────────────

    print(f"\n{'=' * 60}")
    print("STATISTICS")
    print("=" * 60)

    # Per data source
    source_counts = defaultdict(int)
    for (src, qtype), count in stats.items():
        source_counts[src] += count

    print("\nPer data source:")
    for src in sorted(source_counts):
        print(f"  {src:15s}: {source_counts[src]:>7,}")

    # Per question type (grouped by source)
    print("\nPer (data_source, question_type):")
    for (src, qtype) in sorted(stats):
        print(f"  {src:15s} / {qtype:20s}: {stats[(src, qtype)]:>7,}")

    # Per question type (aggregated)
    qtype_counts = defaultdict(int)
    for (src, qtype), count in stats.items():
        qtype_counts[qtype] += count

    print("\nPer question type (all sources):")
    for qtype in sorted(qtype_counts):
        print(f"  {qtype:20s}: {qtype_counts[qtype]:>7,}")

    print(f"\n  Skipped:           {skipped:>7,}")
    print(f"  Total converted:   {len(all_converted):>7,}")

    # ── 5. Print sample conversions ──────────────────────────────────────

    if all_converted:
        print(f"\n{'=' * 60}")
        print("SAMPLE CONVERSIONS")
        print("=" * 60)

        shown = set()
        for item in all_converted:
            ds = item["metadata"]["data_source"]
            dataset = item["metadata"]["dataset"]
            key = (ds, dataset)
            if key not in shown:
                shown.add(key)
                print(f"\n--- {ds.upper()} ({dataset}) ---")
                print(json.dumps(item, indent=2)[:1000])
                if len(shown) >= 5:
                    break


if __name__ == "__main__":
    main()
