"""
VSTIBENCH Evaluation Script (Video-based)

Evaluates spatio-temporal understanding using video inputs.
Question types:
- MCA: camera_movement_direction, camera_obj_rel_dist_v1/v2/v3, obj_obj_relative_pos_lr/nf/ud
- NA: camera_displacement, camera_obj_abs_dist
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from functools import partial
from loguru import logger as eval_logger

with open(Path(__file__).parent / "vstibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

config = yaml.safe_load("".join(safe_data))
dataset_path = config["dataset_path"]
cache_dir = dataset_path if os.path.isdir(dataset_path) else config.get("dataset_kwargs", {}).get("cache_dir", "vstibench")

# Multiple Choice Answer question types
MCA_QUESTION_TYPES = [
    "camera_movement_direction",
    "camera_obj_rel_dist_v1",
    "camera_obj_rel_dist_v2",
    "camera_obj_rel_dist_v3",
    "obj_obj_relative_pos_lr",
    "obj_obj_relative_pos_nf",
    "obj_obj_relative_pos_ud",
]

# Numerical Answer question types
NA_QUESTION_TYPES = [
    "camera_displacement",
    "camera_obj_abs_dist",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}


def vstibench_doc_to_visual(doc):
    """Load video from video_path."""
    video_path = doc.get("video_path", "")

    # Try to find video in cache_dir or dataset_path
    full_path = os.path.join(cache_dir, video_path)
    if not os.path.exists(full_path):
        # Try alternative paths
        alt_path = os.path.join("data/media", video_path.lower())
        if os.path.exists(alt_path):
            full_path = alt_path
        else:
            eval_logger.warning(f"Video not found: {video_path}")
            return [full_path]  # Return path anyway, let model handle it

    return [full_path]


def vstibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Format question with options if present."""
    question = doc["question"]
    question_type = doc.get("question_type", "")

    pre_prompt = "These are frames of a video."
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", pre_prompt) or pre_prompt

    options = doc.get("options", [])

    if question_type in NA_QUESTION_TYPES or not options:
        post_prompt = "Please answer the question using a single word or phrase."
        if lmms_eval_specific_kwargs:
            post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", post_prompt) or post_prompt
        return f"{pre_prompt}\n{question}\n{post_prompt}"
    else:
        # MCA question
        options_text = "Options:\n" + "\n".join(options)
        post_prompt = "Answer with the option's letter from the given choices directly."
        if lmms_eval_specific_kwargs:
            post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", post_prompt) or post_prompt
        return f"{pre_prompt}\n{question}\n{options_text}\n{post_prompt}"


def fuzzy_matching(pred):
    """Extract first word/letter from prediction."""
    return pred.split(' ')[0].rstrip('.').strip()


def exact_match(pred, target):
    """Check if prediction matches target (case-insensitive)."""
    return 1. if pred.lower() == target.lower() else 0.


def abs_dist_norm(pred, target):
    """Compute normalized absolute distance."""
    return abs(pred - target) / target


def mean_relative_accuracy(pred, target, start, end, interval):
    """Compute mean relative accuracy across confidence intervals."""
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


def to_float(pred):
    """Safely convert prediction to float."""
    try:
        pred = float(pred)
    except BaseException:
        pred = None
    return pred


def vstibench_process_results(doc, results):
    """Process results for a single document."""
    doc['prediction'] = results[0]
    question_type = doc['question_type']

    if question_type in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['mc_answer'])
    elif question_type in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(fuzzy_matching(doc['prediction'])), to_float(doc['ground_truth']))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        eval_logger.warning(f"Unknown question type: {question_type}")
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc['prediction']), str(doc['ground_truth']))

    return {"vstibench_score": doc}


def vstibench_aggregate_results(results):
    """Aggregate results across all documents."""
    results = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results.groupby('question_type').groups.items():
        per_question_type = results.iloc[question_type_indexes]

        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        else:
            for metric in METRICS_FOR_MCA.keys():
                if metric in per_question_type.columns:
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

    # Combine camera_obj_rel_dist variants
    rel_dist_keys = [
        'camera_obj_rel_dist_v1_accuracy',
        'camera_obj_rel_dist_v2_accuracy',
        'camera_obj_rel_dist_v3_accuracy',
    ]
    rel_dist_values = []
    for key in rel_dist_keys:
        if key in output:
            rel_dist_values.append(output.pop(key))
    if rel_dist_values:
        output['camera_obj_rel_dist_accuracy'] = sum(rel_dist_values) / len(rel_dist_values)

    # Combine obj_obj_relative_pos variants
    rel_pos_keys = [
        'obj_obj_relative_pos_lr_accuracy',
        'obj_obj_relative_pos_nf_accuracy',
        'obj_obj_relative_pos_ud_accuracy',
    ]
    rel_pos_values = []
    for key in rel_pos_keys:
        if key in output:
            rel_pos_values.append(output.pop(key))
    if rel_pos_values:
        output['obj_obj_relative_pos_accuracy'] = sum(rel_pos_values) / len(rel_pos_values)

    # Compute overall score
    if output:
        output['overall'] = sum([_ for _ in output.values()]) / len(output)
    else:
        output['overall'] = 0.0

    eval_logger.info(f"VSTIBENCH Evaluation results: {output}")
    return output['overall'] * 100.
