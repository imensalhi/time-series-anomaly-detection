from __future__ import annotations

from pathlib import Path

from common import load_params, save_json, set_seed
from services import InfluxCSVService, PreprocessingService


def main() -> None:
	params = load_params()
	set_seed(int(params["seed"]))

	data_dir = Path(params["data"]["data_dir"])
	out_dir = Path("artifacts/preprocessed")
	out_dir.mkdir(parents=True, exist_ok=True)

	influx = InfluxCSVService()
	preproc = PreprocessingService(
		fill_method=params["preprocessing"]["fill_method"],
		idle_quantile=float(params["preprocessing"]["idle_quantile"]),
	)

	datasets = influx.load_folder(data_dir)
	quality_report = {}
	manifest = {
		"profiles": params["data"]["profiles"],
		"sensors": {},
	}

	for stem, df in datasets.items():
		processed = preproc.clean_and_fill(df)
		out_path = out_dir / f"{stem}.npz"
		quality_report[stem] = preproc.summarize_quality(df, processed)

		manifest["sensors"][stem] = {
			"sensor": processed.sensor,
			"file": str(out_path),
			"sampling_seconds": processed.sampling_seconds,
			"run_ratio": quality_report[stem]["run_ratio"],
		}

		import numpy as np

		np.savez_compressed(
			out_path,
			timestamp=processed.timestamp,
			value=processed.value,
			run_mask=processed.run_mask.astype("int8"),
			idle_mask=processed.idle_mask.astype("int8"),
			sampling_seconds=processed.sampling_seconds,
		)

	save_json(Path("metrics/preprocessing_quality.json"), quality_report)
	save_json(out_dir / "manifest.json", manifest)
	print(f"Preprocessing completed for {len(manifest['sensors'])} sensors")


if __name__ == "__main__":
	main()
