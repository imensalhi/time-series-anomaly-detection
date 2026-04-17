from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import load_params, save_json, set_seed
from services import CWTService, CWTCNNAutoencoder, FineTuningService



def create_windows(signal: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    windows = []
    for i in range(0, len(signal) - window_size + 1, step_size):
        windows.append(signal[i : i + window_size])
    return np.array(windows, dtype=np.float32)


def main() -> None:
    params = load_params()
    set_seed(int(params["seed"]))

    target_sensor = params["fine_tuning"]["target_sensor"]
    profile_name = params["fine_tuning"]["source_profile"]

    npz_path = Path("artifacts/preprocessed") / f"{target_sensor}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing preprocessed file: {npz_path}")

    arr = np.load(npz_path)
    values = arr["value"].astype(np.float32)
    run_mask = arr["run_mask"].astype(bool)
    run_values = values[run_mask]
    run_values = (run_values - run_values.mean()) / (run_values.std() + 1e-6)

    windows = create_windows(
        run_values,
        window_size=int(params["windowing"]["window_size"]),
        step_size=int(params["windowing"]["step_size"]),
    )

    cwt = CWTService(
        scales_min=int(params["cwt"]["scales_min"]),
        scales_max=int(params["cwt"]["scales_max"]),
        wavelet=str(params["cwt"]["wavelet"]),
    )
    X = np.expand_dims(cwt.transform(windows), axis=1)

    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=int(params["fine_tuning"]["batch_size"]),
        shuffle=True,
    )

    model = CWTCNNAutoencoder()
    source_ckpt = Path(params["autoencoder"]["output_dir"]) / f"{profile_name}_autoencoder.pth"
    model.load_state_dict(torch.load(source_ckpt, map_location="cpu"))

    history = FineTuningService.fine_tune_autoencoder(
        model=model,
        loader=loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        epochs=int(params["fine_tuning"]["epochs"]),
        learning_rate=float(params["fine_tuning"]["learning_rate"]),
    )

    out_path = Path(params["autoencoder"]["output_dir"]) / f"{profile_name}_finetuned_{target_sensor}.pth"
    torch.save(model.state_dict(), out_path)
    save_json(Path("metrics/fine_tune_metrics.json"), {"history": history, "output_model": str(out_path)})
    print("Fine-tuning completed")


if __name__ == "__main__":
    main()
