from __future__ import annotations

import json
from pathlib import Path

from common import save_json


def aggregate_model_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not payload:
        return {}

    f1 = [p["binary"]["f1"] for p in payload.values()]
    precision = [p["binary"]["precision"] for p in payload.values()]
    recall = [p["binary"]["recall"] for p in payload.values()]

    return {
        "n_profiles": len(payload),
        "mean_f1": float(sum(f1) / len(f1)),
        "mean_precision": float(sum(precision) / len(precision)),
        "mean_recall": float(sum(recall) / len(recall)),
        "profiles": payload,
    }


def main() -> None:
    ae = aggregate_model_metrics(Path("metrics/autoencoder_metrics.json"))
    cl = aggregate_model_metrics(Path("metrics/contrastive_metrics.json"))

    summary = {
        "autoencoder": ae,
        "contrastive": cl,
    }
    save_json(Path("metrics/model_comparison.json"), summary)
    print("Model comparison written to metrics/model_comparison.json")


if __name__ == "__main__":
    main()
