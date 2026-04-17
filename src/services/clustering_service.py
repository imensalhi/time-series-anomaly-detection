from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusterPrediction:
    assigned_cluster: int
    is_new_group: bool
    distance_to_centroid: float
    threshold: float


class KMeansMachineGrouper:
    """Cluster machines and detect if a new machine should open a new group."""

    def __init__(self, min_k: int = 2, max_k: int = 6, random_state: int = 42) -> None:
        self.min_k = min_k
        self.max_k = max_k
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: KMeans | None = None

    @staticmethod
    def feature_vector(run_values: np.ndarray) -> np.ndarray:
        if len(run_values) == 0:
            return np.zeros(8, dtype=np.float32)

        mean = float(np.mean(run_values))
        std = float(np.std(run_values))
        rms = float(np.sqrt(np.mean(run_values ** 2)))
        p95 = float(np.percentile(run_values, 95))
        p05 = float(np.percentile(run_values, 5))
        peak = float(np.max(np.abs(run_values)))
        crest = peak / (rms + 1e-12)
        iqr = float(np.percentile(run_values, 75) - np.percentile(run_values, 25))
        return np.array([mean, std, rms, p95, p05, peak, crest, iqr], dtype=np.float32)

    def fit(self, machine_run_values: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, int], int]:
        machine_ids = sorted(machine_run_values.keys())
        X = np.stack([self.feature_vector(machine_run_values[mid]) for mid in machine_ids])
        Xs = self.scaler.fit_transform(X)

        best_k = self.min_k
        best_score = -1.0
        best_model = None

        upper_k = min(self.max_k, len(machine_ids))
        for k in range(self.min_k, max(self.min_k, upper_k) + 1):
            if k >= len(machine_ids):
                continue
            candidate = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = candidate.fit_predict(Xs)
            score = silhouette_score(Xs, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_model = candidate

        if best_model is None:
            best_model = KMeans(n_clusters=1, random_state=self.random_state, n_init=10).fit(Xs)
            best_k = 1

        self.model = best_model
        labels = self.model.predict(Xs)
        mapping = {mid: int(label) for mid, label in zip(machine_ids, labels)}
        return Xs, mapping, best_k

    def predict_new_machine(self, run_values: np.ndarray, distance_factor: float = 1.2) -> ClusterPrediction:
        if self.model is None:
            raise RuntimeError("KMeans model is not trained")

        x = self.feature_vector(run_values).reshape(1, -1)
        xs = self.scaler.transform(x)
        cluster = int(self.model.predict(xs)[0])

        centroid = self.model.cluster_centers_[cluster]
        dist = float(np.linalg.norm(xs[0] - centroid))

        all_dists = np.linalg.norm(self.model.transform(self.model.cluster_centers_), axis=1)
        threshold = float(np.percentile(all_dists, 90) * distance_factor)
        is_new_group = dist > threshold

        return ClusterPrediction(
            assigned_cluster=cluster,
            is_new_group=is_new_group,
            distance_to_centroid=dist,
            threshold=threshold,
        )
