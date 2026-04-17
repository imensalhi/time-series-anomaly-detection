from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def optimize_threshold(scores: np.ndarray, y_true_binary: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Optimize threshold by maximizing binary F1."""
    y = y_true_binary.astype(np.int64)

    candidates = np.linspace(float(np.percentile(scores, 1)), float(np.percentile(scores, 99)), 400)
    best = {"f1": -1.0, "precision": 0.0, "recall": 0.0}
    best_t = float(candidates[len(candidates) // 2])

    for t in candidates:
        yp = (scores > t).astype(np.int64)
        tp = int(np.sum((y == 1) & (yp == 1)))
        fp = int(np.sum((y == 0) & (yp == 1)))
        fn = int(np.sum((y == 1) & (yp == 0)))

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        if f1 > best["f1"]:
            best_t = float(t)
            best = {
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }

    return best_t, best
