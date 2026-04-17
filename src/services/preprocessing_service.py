from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class ProcessedSignal:
    sensor: str
    timestamp: np.ndarray
    value: np.ndarray
    run_mask: np.ndarray
    idle_mask: np.ndarray
    sampling_seconds: float


class PreprocessingService:
    """Handle nulls, gaps, and run/idle segmentation on electrical current signals."""

    def __init__(self, fill_method: str = "time", idle_quantile: float = 0.25) -> None:
        self.fill_method = fill_method
        self.idle_quantile = idle_quantile

    @staticmethod
    def infer_sampling_seconds(timestamps: pd.Series) -> float:
        deltas = timestamps.sort_values().diff().dropna().dt.total_seconds()
        if deltas.empty:
            return 1.0
        return max(float(deltas.median()), 1e-3)

    def clean_and_fill(self, df: pd.DataFrame) -> ProcessedSignal:
        sensor = str(df["sensor"].iloc[0])
        working = df.copy()
        working = working.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

        sampling_seconds = self.infer_sampling_seconds(working["timestamp"])
        freq_ms = max(int(round(sampling_seconds * 1000)), 1)
        full_index = pd.date_range(
            start=working["timestamp"].min(),
            end=working["timestamp"].max(),
            freq=f"{freq_ms}ms",
            tz="UTC",
        )

        indexed = working.set_index("timestamp").reindex(full_index)
        indexed["sensor"] = sensor
        indexed["value"] = pd.to_numeric(indexed["value"], errors="coerce")

        if self.fill_method == "ffill":
            indexed["value"] = indexed["value"].ffill().bfill()
        else:
            indexed["value"] = indexed["value"].interpolate(method="time").ffill().bfill()

        run_mask, idle_mask = self.detect_run_idle(indexed["value"].to_numpy(dtype=np.float32))

        return ProcessedSignal(
            sensor=sensor,
            timestamp=indexed.index.view("int64").to_numpy(),
            value=indexed["value"].to_numpy(dtype=np.float32),
            run_mask=run_mask,
            idle_mask=idle_mask,
            sampling_seconds=sampling_seconds,
        )

    def detect_run_idle(self, values: np.ndarray, window: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        if len(values) < window:
            run_mask = np.ones(len(values), dtype=bool)
            return run_mask, ~run_mask

        series = pd.Series(values)
        rolling_std = series.rolling(window=window, min_periods=1).std().fillna(0.0)
        rolling_abs = series.abs().rolling(window=window, min_periods=1).mean().fillna(0.0)

        std_threshold = float(np.quantile(rolling_std, self.idle_quantile))
        abs_threshold = float(np.quantile(rolling_abs, self.idle_quantile))

        idle_mask = (rolling_std <= std_threshold) & (rolling_abs <= abs_threshold)
        run_mask = ~idle_mask.to_numpy()
        return run_mask.astype(bool), idle_mask.to_numpy(dtype=bool)

    @staticmethod
    def create_windows(
        values: np.ndarray,
        labels: np.ndarray,
        window_size: int,
        step_size: int,
        require_run_only: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        windows = []
        y = []
        for start in range(0, len(values) - window_size + 1, step_size):
            end = start + window_size
            w = values[start:end]
            l = labels[start:end]
            if require_run_only and not np.all(l == 1):
                continue
            windows.append(w)
            y.append(1 if np.mean(l) > 0.5 else 0)

        if not windows:
            return np.empty((0, window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)

        return np.stack(windows).astype(np.float32), np.array(y, dtype=np.int64)

    @staticmethod
    def summarize_quality(raw_df: pd.DataFrame, processed: ProcessedSignal) -> Dict[str, float]:
        raw_nulls = float(raw_df["value"].isna().sum())
        raw_points = float(len(raw_df))
        processed_points = float(len(processed.value))
        gaps_filled = max(processed_points - raw_points, 0.0)
        return {
            "raw_points": raw_points,
            "processed_points": processed_points,
            "null_count": raw_nulls,
            "gaps_filled": gaps_filled,
            "run_ratio": float(np.mean(processed.run_mask)),
            "idle_ratio": float(np.mean(processed.idle_mask)),
            "sampling_seconds": processed.sampling_seconds,
        }
