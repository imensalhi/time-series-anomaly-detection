from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

ANOMALY_TYPES = ["spike", "plateau", "drift", "variance", "dropout", "shape_shift"]
ANOMALY_TO_LABEL = {name: i + 1 for i, name in enumerate(ANOMALY_TYPES)}
ANOMALY_TO_LABEL["normal"] = 0


class AnomalyInjectionService:
    def __init__(self, random_state: int = 42) -> None:
        self.rng = np.random.default_rng(random_state)

    def inject(self, window: np.ndarray, anomaly_type: str, mean: float, std: float) -> np.ndarray:
        w = window.copy()
        std = max(std, 1e-6)

        if anomaly_type == "spike":
            n_spikes = int(self.rng.integers(1, 4))
            for _ in range(n_spikes):
                pos = int(self.rng.integers(0, len(w)))
                sign = float(self.rng.choice([-1.0, 1.0]))
                w[pos] += sign * float(self.rng.uniform(5, 10)) * std
            return w

        if anomaly_type == "plateau":
            length = int(self.rng.integers(max(3, int(0.3 * len(w))), max(4, int(0.7 * len(w)))))
            start = int(self.rng.integers(0, len(w) - length))
            plateau = mean + float(self.rng.uniform(-0.5, 0.5)) * mean
            w[start : start + length] = plateau
            return w

        if anomaly_type == "drift":
            magnitude = float(self.rng.uniform(2, 5)) * std * float(self.rng.choice([-1.0, 1.0]))
            w += np.linspace(0, magnitude, len(w), dtype=np.float32)
            return w

        if anomaly_type == "variance":
            length = int(self.rng.integers(max(3, int(0.3 * len(w))), max(4, int(0.7 * len(w)))))
            start = int(self.rng.integers(0, len(w) - length))
            factor = float(self.rng.uniform(3, 6))
            noise = self.rng.normal(0, factor * std, length).astype(np.float32)
            w[start : start + length] += noise
            return w

        if anomaly_type == "dropout":
            length = int(self.rng.integers(max(2, int(0.2 * len(w))), max(3, int(0.5 * len(w)))))
            start = int(self.rng.integers(0, len(w) - length))
            w[start : start + length] = 0.0
            return w

        if anomaly_type == "shape_shift":
            length = int(self.rng.integers(max(3, int(0.4 * len(w))), max(4, int(0.8 * len(w)))))
            start = int(self.rng.integers(0, len(w) - length))
            freq = float(self.rng.uniform(2, 8))
            amp = float(self.rng.uniform(2, 4)) * std
            t = np.linspace(0, 2 * np.pi * freq, length)
            w[start : start + length] = np.mean(w[start : start + length]) + amp * np.sin(t)
            return w

        return w

    def inject_batch(
        self,
        windows: np.ndarray,
        anomaly_ratio: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = windows.copy()
        labels_multi = np.zeros(len(out), dtype=np.int64)
        labels_binary = np.zeros(len(out), dtype=np.int64)

        n_anomaly = max(1, int(len(out) * anomaly_ratio))
        selected = self.rng.choice(len(out), size=n_anomaly, replace=False)
        mean, std = float(np.mean(out)), float(np.std(out))

        for idx in selected:
            anomaly_type = str(self.rng.choice(ANOMALY_TYPES))
            out[idx] = self.inject(out[idx], anomaly_type, mean, std)
            labels_multi[idx] = ANOMALY_TO_LABEL[anomaly_type]
            labels_binary[idx] = 1

        return out.astype(np.float32), labels_binary, labels_multi

    @staticmethod
    def per_type_binary_metrics(y_true_multi: np.ndarray, y_pred_binary: np.ndarray) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}
        for anomaly in ANOMALY_TYPES:
            label = ANOMALY_TO_LABEL[anomaly]
            true_mask = (y_true_multi == label).astype(np.int64)
            pred_mask = y_pred_binary.copy().astype(np.int64)

            tp = int(np.sum((true_mask == 1) & (pred_mask == 1)))
            fp = int(np.sum((true_mask == 0) & (pred_mask == 1)))
            fn = int(np.sum((true_mask == 1) & (pred_mask == 0)))

            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            metrics[anomaly] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(np.sum(true_mask)),
            }
        return metrics
