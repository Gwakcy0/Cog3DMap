"""
VSIBENCH Point3R Evaluation Script

Based on VSI-Bench evaluation but adapted for Point3R format with pointer_data.
Question types:
- MCA: object_rel_direction_easy/medium/hard, object_rel_distance, route_planning, obj_appearance_order
- NA: object_abs_distance, object_counting, object_size_estimation, room_size_estimation
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from functools import partial
from loguru import logger as eval_logger

with open(Path(__file__).parent / "vsibench_point3r.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
media_dir = yaml.safe_load("".join(safe_data))["metadata"]["media_dir"]

# Multiple Choice Answer question types
MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]

# Numerical Answer question types
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
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


def vsibench_doc_to_visual(doc):
    """Return empty list since Point3R uses pointer_data instead of images/videos."""
    return [[]]


def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Extract question text from conversations."""
    question = doc["conversations"][0]["value"]
    return question


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


def vsibench_process_results(doc, results):
    """Process results for a single document."""
    doc['prediction'] = results[0]
    question_type = doc['question_type']

    if question_type in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['ground_truth'])
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

    return {"vsibench_score": doc}


def vsibench_aggregate_results(results):
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

    # Combine object_rel_direction variants if present
    direction_keys = [
        'object_rel_direction_easy_accuracy',
        'object_rel_direction_medium_accuracy',
        'object_rel_direction_hard_accuracy'
    ]
    direction_values = []
    for key in direction_keys:
        if key in output:
            direction_values.append(output.pop(key))
    if direction_values:
        output['object_rel_direction_accuracy'] = sum(direction_values) / len(direction_values)

    # Compute overall score
    if output:
        output['overall'] = sum([_ for _ in output.values()]) / len(output)
    else:
        output['overall'] = 0.0

    eval_logger.info(f"VSIBENCH Point3R Evaluation results: {output}")
    return output['overall'] * 100.
