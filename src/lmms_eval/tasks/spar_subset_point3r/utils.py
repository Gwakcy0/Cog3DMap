"""
SPAR-Subset Point3R Evaluation Script

Evaluates appearance ordering tasks from SPAR-subset in MCA format.
Question type: obj_appearance_order (multiple choice A/B/C/D)
Data sources: scannet, scannetpp
"""

from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger

with open(Path(__file__).parent / "spar_subset_point3r.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
media_dir = yaml.safe_load("".join(safe_data))["metadata"]["media_dir"]


def spar_subset_doc_to_visual(doc):
    """Return empty list since Point3R uses pointer_data instead of images/videos."""
    return [[]]


def spar_subset_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Extract question text from conversations."""
    question = doc["conversations"][0]["value"]
    return question


def fuzzy_matching(pred):
    """Extract first word/letter from prediction."""
    return pred.split(' ')[0].rstrip('.').strip()


def exact_match(pred, target):
    """Check if prediction matches target (case-insensitive)."""
    return 1.0 if pred.strip().lower() == target.strip().lower() else 0.0


def spar_subset_process_results(doc, results):
    """Process results for a single document."""
    doc["prediction"] = results[0]
    doc["accuracy"] = exact_match(fuzzy_matching(doc["prediction"]), doc["ground_truth"])
    doc["data_source"] = doc["metadata"]["data_source"]
    return {"spar_subset_point3r_score": doc}


def spar_subset_aggregate_results(results):
    """Aggregate results across all documents."""
    results = pd.DataFrame(results)

    output = {}

    # Group by data_source (scannet/scannetpp)
    for data_source, ds_indexes in results.groupby("data_source").groups.items():
        per_ds = results.iloc[ds_indexes]
        output[f"{data_source}_accuracy"] = per_ds["accuracy"].mean()

    # Compute overall score
    if output:
        output["overall"] = sum(output.values()) / len(output)
    else:
        output["overall"] = 0.0

    eval_logger.info(f"SPAR-Subset Point3R Evaluation results: {output}")
    return output["overall"] * 100.0
