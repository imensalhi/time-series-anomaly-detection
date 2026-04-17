from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import load_params, save_json, set_seed
from services import InfluxCSVService, KMeansMachineGrouper, PreprocessingService


def load_run_values(sensor_npz_path: Path) -> np.ndarray:
    arr = np.load(sensor_npz_path)
    values = arr["value"].astype(np.float32)
    run_mask = arr["run_mask"].astype(bool)
    return values[run_mask]


def main() -> None:
    params = load_params()
    set_seed(int(params["seed"]))

    manifest = json.loads(Path("artifacts/preprocessed/manifest.json").read_text(encoding="utf-8"))

    run_values = {}
    for sensor_stem in manifest["sensors"].keys():
        npz_path = Path("artifacts/preprocessed") / f"{sensor_stem}.npz"
        if npz_path.exists():
            run_values[sensor_stem] = load_run_values(npz_path)

    grouper = KMeansMachineGrouper(
        min_k=int(params["clustering"]["min_k"]),
        max_k=int(params["clustering"]["max_k"]),
        random_state=int(params["seed"]),
    )
    _, mapping, k = grouper.fit(run_values)

    output = {
        "best_k": k,
        "machine_cluster_map": mapping,
    }

    new_machine_file = params["clustering"].get("new_machine_file", "")
    if new_machine_file:
        path = Path(new_machine_file)
        if path.exists() and path.stat().st_size > 0:
            influx_df = InfluxCSVService.load_file(path)
            preproc = PreprocessingService(
                fill_method=params["preprocessing"]["fill_method"],
                idle_quantile=float(params["preprocessing"]["idle_quantile"]),
            )
            processed = preproc.clean_and_fill(influx_df)
            prediction = grouper.predict_new_machine(processed.value[processed.run_mask])
            output["new_machine_prediction"] = {
                "assigned_cluster": prediction.assigned_cluster,
                "is_new_group": prediction.is_new_group,
                "distance_to_centroid": prediction.distance_to_centroid,
                "threshold": prediction.threshold,
            }

    save_json(Path("metrics/clustering_metrics.json"), output)
    save_json(Path("artifacts/clustering/clusters.json"), output)
    print("Clustering completed")


if __name__ == "__main__":
    main()
