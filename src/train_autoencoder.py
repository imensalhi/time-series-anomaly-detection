from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from common import load_params, save_json, set_seed
from services import (
	ANOMALY_TYPES,
	AnomalyInjectionService,
	CWTService,
	CWTCNNAutoencoder,
	EvaluationService,
	optimize_threshold,
	reconstruction_scores,
	train_autoencoder,
)


def load_run_values(sensor_npz_path: Path) -> np.ndarray:
	arr = np.load(sensor_npz_path)
	values = arr["value"].astype(np.float32)
	run_mask = arr["run_mask"].astype(bool)
	return values[run_mask]


def create_windows(signal: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
	windows = []
	for i in range(0, len(signal) - window_size + 1, step_size):
		windows.append(signal[i : i + window_size])
	return np.array(windows, dtype=np.float32)


def main() -> None:
	params = load_params()
	set_seed(int(params["seed"]))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	output_dir = Path(params["autoencoder"]["output_dir"])
	output_dir.mkdir(parents=True, exist_ok=True)

	cwt = CWTService(
		scales_min=int(params["cwt"]["scales_min"]),
		scales_max=int(params["cwt"]["scales_max"]),
		wavelet=str(params["cwt"]["wavelet"]),
	)
	injector = AnomalyInjectionService(random_state=int(params["seed"]))
	evaluator = EvaluationService()

	mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
	mlflow.set_experiment(params["mlflow"]["experiment_autoencoder"])

	all_metrics = {}

	for profile_name, sensors in params["data"]["profiles"].items():
		run_sequences = []
		for sensor in sensors:
			sensor_file = Path("artifacts/preprocessed") / f"{sensor}.npz"
			if not sensor_file.exists():
				continue
			run_sequences.append(load_run_values(sensor_file))

		if not run_sequences:
			continue

		signal = np.concatenate(run_sequences)
		signal = (signal - signal.mean()) / (signal.std() + 1e-6)

		windows = create_windows(
			signal,
			window_size=int(params["windowing"]["window_size"]),
			step_size=int(params["windowing"]["step_size"]),
		)
		if len(windows) < 20:
			continue

		train_w, test_w = train_test_split(windows, test_size=0.2, random_state=int(params["seed"]))
		train_w, val_w = train_test_split(train_w, test_size=0.2, random_state=int(params["seed"]))

		val_aug, y_val_bin, _ = injector.inject_batch(
			val_w,
			anomaly_ratio=float(params["autoencoder"]["anomaly_ratio_test"]),
		)
		test_aug, y_test_bin, y_test_multi = injector.inject_batch(
			test_w,
			anomaly_ratio=float(params["autoencoder"]["anomaly_ratio_test"]),
		)

		X_train = cwt.transform(train_w)
		X_val = cwt.transform(val_aug)
		X_test = cwt.transform(test_aug)

		X_train = np.expand_dims(X_train, 1)
		X_val = np.expand_dims(X_val, 1)
		X_test = np.expand_dims(X_test, 1)

		train_loader = DataLoader(
			TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
			batch_size=int(params["autoencoder"]["batch_size"]),
			shuffle=True,
		)
		val_loader = DataLoader(
			TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
			batch_size=int(params["autoencoder"]["batch_size"]),
			shuffle=False,
		)

		model = CWTCNNAutoencoder()
		history = train_autoencoder(
			model=model,
			train_loader=train_loader,
			val_loader=val_loader,
			device=device,
			epochs=int(params["autoencoder"]["epochs"]),
			learning_rate=float(params["autoencoder"]["learning_rate"]),
		)

		val_scores = reconstruction_scores(model.to(device), X_val, device)
		threshold, threshold_stats = optimize_threshold(val_scores, y_val_bin)

		test_scores = reconstruction_scores(model.to(device), X_test, device)
		y_pred = (test_scores > threshold).astype(np.int64)

		binary = evaluator.binary_metrics(y_test_bin, y_pred)
		metrics_path = Path("metrics") / f"{profile_name}_autoencoder_metrics.json"
		evaluator.save_metrics_report(metrics_path, binary, y_test_multi, y_pred)

		evaluator.save_confusion_matrix(
			y_true=y_test_bin,
			y_pred=y_pred,
			output_path=Path("reports") / f"{profile_name}_autoencoder_confusion.png",
			title=f"{profile_name} autoencoder confusion",
		)

		torch.save(model.state_dict(), output_dir / f"{profile_name}_autoencoder.pth")

		with mlflow.start_run(run_name=f"autoencoder-{profile_name}"):
			mlflow.log_params(
				{
					"profile": profile_name,
					"window_size": int(params["windowing"]["window_size"]),
					"step_size": int(params["windowing"]["step_size"]),
					"epochs": int(params["autoencoder"]["epochs"]),
				}
			)
			mlflow.log_metric("f1", binary["f1"])
			mlflow.log_metric("precision", binary["precision"])
			mlflow.log_metric("recall", binary["recall"])
			mlflow.log_metric("threshold", threshold)

		all_metrics[profile_name] = {
			"binary": binary,
			"threshold": threshold,
			"threshold_stats": threshold_stats,
			"history_last": {
				"train_loss": history["train_loss"][-1],
				"val_loss": history["val_loss"][-1],
			},
			"anomaly_types": ANOMALY_TYPES,
		}

	save_json(Path("metrics/autoencoder_metrics.json"), all_metrics)
	print("Autoencoder training complete")


if __name__ == "__main__":
	main()
