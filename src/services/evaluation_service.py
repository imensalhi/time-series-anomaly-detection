from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from .anomaly_service import AnomalyInjectionService


class EvaluationService:
    @staticmethod
    def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        acc = (tp + tn) / max(len(y_true), 1)

        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    @staticmethod
    def save_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: Path,
        title: str,
    ) -> None:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1], ["normal", "anomaly"])
        ax.set_yticks([0, 1], ["normal", "anomaly"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def save_metrics_report(
        output_path: Path,
        binary_metrics: Dict[str, float],
        y_true_multi: np.ndarray,
        y_pred_binary: np.ndarray,
    ) -> None:
        per_type = AnomalyInjectionService.per_type_binary_metrics(y_true_multi, y_pred_binary)
        payload = {
            "binary": binary_metrics,
            "per_anomaly_type": per_type,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
