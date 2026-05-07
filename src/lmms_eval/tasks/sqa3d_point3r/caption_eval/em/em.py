#!/usr/bin/env python
#
# File Name : em.py
#
# Description : Computes Exact Match (EM) metric with text normalization

import string


class ExactMatch:
    """Exact Match metric with text normalization."""

    @staticmethod
    def normalize_text(text):
        """Normalize text: lowercase + remove punctuation + strip whitespace."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = " ".join(text.split())  # Normalize whitespace
        return text

    def compute_score(self, gts, res):
        """
        Compute Exact Match score.
        :param gts: dict {id: [list of acceptable answers]}
        :param res: dict {id: [predicted answer]}
        :return: (avg_score, list of per-item scores)
        """
        assert gts.keys() == res.keys()
        scores = []
        for id in gts.keys():
            pred = self.normalize_text(res[id][0])
            gt_normalized = [self.normalize_text(gt) for gt in gts[id]]
            match = 1.0 if pred in gt_normalized else 0.0
            scores.append(match)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score, scores

    def method(self):
        return "EM"
