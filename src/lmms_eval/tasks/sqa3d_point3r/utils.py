"""
SQA3D Point3R Evaluation Script

Metrics: EM (Exact Match) as primary, plus BLEU, CIDEr, METEOR, ROUGE
"""

import os
from pathlib import Path

import yaml
from loguru import logger as eval_logger
from PIL import Image
from terminaltables import AsciiTable

with open(Path(__file__).parent / "sqa3d_point3r.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
media_dir = yaml.safe_load("".join(safe_data))["metadata"]["media_dir"]


def sqa3d_doc_to_visual(doc):
    image_files = doc.get("images", [])
    images = [
        Image.open(os.path.join(media_dir, image_file)).convert("RGB")
        for image_file in image_files
    ]
    return [images]


def sqa3d_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["conversations"][0]["value"].replace("<image>", "")
    return question


def sqa3d_process_results(doc, results):
    doc["pred_response"] = results[0]
    doc["gt_responses"] = doc["answers"]
    return {"sqa3d_score": doc}


def sqa3d_aggregate_results(results):
    from lmms_eval.tasks.sqa3d_point3r.caption_eval.bleu.bleu import Bleu
    from lmms_eval.tasks.sqa3d_point3r.caption_eval.cider.cider import Cider
    from lmms_eval.tasks.sqa3d_point3r.caption_eval.em.em import ExactMatch
    from lmms_eval.tasks.sqa3d_point3r.caption_eval.meteor.meteor import Meteor
    from lmms_eval.tasks.sqa3d_point3r.caption_eval.rouge.rouge import Rouge

    cider = Cider()
    bleu = Bleu()
    meteor = Meteor()
    rouge = Rouge()
    em = ExactMatch()

    res, gts = {}, {}
    for i, item in enumerate(results):
        res[i] = [item["pred_response"].rstrip(".")]
        gts[i] = item["gt_responses"]

    em_score = em.compute_score(gts, res)
    cider_score = cider.compute_score(gts, res)
    bleu_score = bleu.compute_score(gts, res)
    meteor_score = meteor.compute_score(gts, res)
    rouge_score = rouge.compute_score(gts, res)

    table_data = [
        ["Metric", "Score"],
        ["EM", f"{em_score[0]*100:.2f}"],
        ["CIDER", f"{cider_score[0]*100:.2f}"],
        ["BLEU-4", f"{bleu_score[0][-1]*100:.2f}"],
        ["METEOR", f"{meteor_score[0]*100:.2f}"],
        ["ROUGE", f"{rouge_score[0]*100:.2f}"],
        ["Data Num", f"{len(res)}"],
    ]

    table = AsciiTable(table_data)
    table.title = "SQA3D Evaluation Metrics"
    eval_logger.info("\n" + table.table)
    return em_score[0] * 100
