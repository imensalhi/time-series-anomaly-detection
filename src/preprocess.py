"""
Shared data loading and preprocessing utilities.

Used by all three training scripts (classification, autoencoder, contrastive)
so that data handling is consistent and DVC-tracked in one place.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants (also overridable via params.yaml)
# ─────────────────────────────────────────────────────────────────────────────
ANOMALY_TYPES: List[str] = [
    "spike", "plateau", "drift", "variance", "dropout", "shape_shift"
]
LABEL_MAP: dict = {name: i + 1 for i, name in enumerate(ANOMALY_TYPES)}
LABEL_MAP["normal"] = 0
LABEL_NAMES: List[str] = ["normal"] + ANOMALY_TYPES

ANOMALY_COLORS: dict = {
    "normal": "#2ECC71",
    "spike": "#E74C3C",
    "plateau": "#3498DB",
    "drift": "#F39C12",
    "variance": "#9B59B6",
    "dropout": "#1ABC9C",
    "shape_shift": "#E67E22",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(file_path: Path) -> np.ndarray:
    """Load an InfluxDB-formatted CSV and return the normalised _value column."""
    df = pd.read_csv(file_path, skiprows=3, header=0)
    if "_value" not in df.columns:
        raise ValueError(f"Column '_value' not found in {file_path}")
    values = df["_value"].dropna().values.astype(np.float32)
    scaler = StandardScaler()
    values = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    return values


def load_profile(
    data_dir: Path,
    sensor_names: List[str],
) -> np.ndarray:
    """Concatenate signals from all sensors belonging to the same profile."""
    segments = []
    for name in sensor_names:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Sensor file not found: {path}")
        segments.append(load_dataset(path))
    return np.concatenate(segments)


# ─────────────────────────────────────────────────────────────────────────────
# Sliding window
# ─────────────────────────────────────────────────────────────────────────────

def sliding_windows(
    signal: np.ndarray,
    window_size: int,
    step_size: int,
) -> np.ndarray:
    """Split a 1-D signal into overlapping windows of fixed length."""
    starts = range(0, len(signal) - window_size + 1, step_size)
    return np.array([signal[s: s + window_size] for s in starts], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly injection
# ─────────────────────────────────────────────────────────────────────────────

def inject_anomaly(window: np.ndarray, anomaly_type: str, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of *window* with a synthetic anomaly of *anomaly_type*."""
    w = window.copy()
    n = len(w)
    std = float(np.std(w)) if np.std(w) > 0 else 1.0

    if anomaly_type == "spike":
        n_spikes = rng.integers(1, 4)
        positions = rng.integers(0, n, size=n_spikes)
        amplitudes = rng.uniform(5, 10, size=n_spikes) * std * rng.choice([-1, 1], size=n_spikes)
        for pos, amp in zip(positions, amplitudes):
            w[pos] += amp

    elif anomaly_type == "plateau":
        start = rng.integers(0, int(n * 0.3))
        length = rng.integers(int(n * 0.3), int(n * 0.7))
        end = min(start + length, n)
        plateau_val = rng.uniform(-2, 2) * std
        w[start:end] = plateau_val

    elif anomaly_type == "drift":
        slope = rng.uniform(2, 5) * std * rng.choice([-1, 1])
        w += np.linspace(0, slope, n)

    elif anomaly_type == "variance":
        start = rng.integers(0, n // 2)
        end = rng.integers(n // 2, n)
        noise = rng.normal(0, rng.uniform(3, 6) * std, end - start)
        w[start:end] += noise

    elif anomaly_type == "dropout":
        start = rng.integers(0, int(n * 0.5))
        length = rng.integers(int(n * 0.2), int(n * 0.5))
        end = min(start + length, n)
        w[start:end] = 0.0

    elif anomaly_type == "shape_shift":
        freq = rng.uniform(2, 6)
        amp = rng.uniform(2, 4) * std
        t = np.linspace(0, 2 * np.pi * freq, n)
        w = amp * np.sin(t).astype(np.float32)

    return w


def build_anomaly_dataset(
    normal_windows: np.ndarray,
    anomaly_ratio: float,
    rng: np.random.Generator,
    anomaly_types: List[str] = ANOMALY_TYPES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine normal windows with synthetically injected anomalies.

    Returns
    -------
    windows : ndarray, shape (N, window_size)
    labels  : ndarray of int (0 = normal, 1-6 = anomaly type)
    """
    n_anomalies = int(len(normal_windows) * anomaly_ratio / (1 - anomaly_ratio))
    n_per_type = max(1, n_anomalies // len(anomaly_types))

    anomaly_windows: List[np.ndarray] = []
    anomaly_labels: List[int] = []

    for atype in anomaly_types:
        source_idx = rng.integers(0, len(normal_windows), size=n_per_type)
        for idx in source_idx:
            anomaly_windows.append(inject_anomaly(normal_windows[idx], atype, rng))
            anomaly_labels.append(LABEL_MAP[atype])

    all_windows = np.concatenate(
        [normal_windows, np.array(anomaly_windows, dtype=np.float32)], axis=0
    )
    all_labels = np.concatenate(
        [np.zeros(len(normal_windows), dtype=np.int64),
         np.array(anomaly_labels, dtype=np.int64)],
        axis=0,
    )
    shuffle = rng.permutation(len(all_windows))
    return all_windows[shuffle], all_labels[shuffle]


# ─────────────────────────────────────────────────────────────────────────────
# CWT helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_cwt(
    windows: np.ndarray,
    scales: np.ndarray,
    wavelet: str = "morl",
) -> np.ndarray:
    """
    Compute the CWT scalogram for each window.

    Returns
    -------
    scalograms : ndarray, shape (N, 1, len(scales), window_size)
    """
    scalograms = []
    for w in windows:
        coef, _ = pywt.cwt(w, scales, wavelet)
        scalograms.append(np.abs(coef).astype(np.float32))
    arr = np.array(scalograms)               # (N, scales, window_size)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr[:, np.newaxis, :, :]          # (N, 1, scales, window_size)
