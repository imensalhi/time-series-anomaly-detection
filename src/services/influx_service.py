from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


class InfluxCSVService:
    """Load InfluxDB CSV exports and normalize them into a common schema."""

    REQUIRED_COLUMNS = {"_time", "_value"}

    @staticmethod
    def load_file(path: Path) -> pd.DataFrame:
        raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        skiprows = 3 if raw and raw[0].startswith("#group") else 0

        df = pd.read_csv(path, skiprows=skiprows)
        if not InfluxCSVService.REQUIRED_COLUMNS.issubset(df.columns):
            raise ValueError(f"File {path} does not look like an Influx CSV export")

        df["_time"] = pd.to_datetime(df["_time"], errors="coerce", utc=True)
        df["_value"] = pd.to_numeric(df["_value"], errors="coerce")

        measurement = None
        if "_measurement" in df.columns and not df["_measurement"].dropna().empty:
            measurement = str(df["_measurement"].dropna().iloc[0])

        if not measurement:
            measurement = path.stem

        out = pd.DataFrame(
            {
                "timestamp": df["_time"],
                "value": df["_value"],
                "sensor": measurement,
                "source_file": path.name,
            }
        )
        return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def load_folder(data_dir: Path) -> Dict[str, pd.DataFrame]:
        datasets: Dict[str, pd.DataFrame] = {}
        for file_path in sorted(data_dir.glob("*.csv")):
            if file_path.stat().st_size == 0:
                continue
            df = InfluxCSVService.load_file(file_path)
            if df.empty:
                continue
            datasets[file_path.stem] = df
        return datasets

    @staticmethod
    def from_query_result(rows: Iterable[dict], sensor_name: str) -> pd.DataFrame:
        """Convert rows returned by an Influx query API to the same schema."""
        df = pd.DataFrame(rows)
        if "_time" not in df.columns or "_value" not in df.columns:
            raise ValueError("Influx query rows must contain _time and _value")
        df["timestamp"] = pd.to_datetime(df["_time"], errors="coerce", utc=True)
        df["value"] = pd.to_numeric(df["_value"], errors="coerce")
        df["sensor"] = sensor_name
        df["source_file"] = "influx_query"
        return df[["timestamp", "value", "sensor", "source_file"]].dropna(subset=["timestamp"])
