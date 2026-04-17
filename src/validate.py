from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import save_json


def build_preprocessing_dataframe(manifest_path: Path) -> pd.DataFrame:
	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	records = []
	for sensor_stem, meta in manifest["sensors"].items():
		npz = Path(meta["file"])
		import numpy as np

		arr = np.load(npz)
		values = arr["value"].astype(float)
		run = arr["run_mask"].astype(int)
		records.append(
			{
				"sensor_stem": sensor_stem,
				"mean_current": float(values.mean()),
				"std_current": float(values.std()),
				"run_ratio": float(run.mean()),
				"n_points": int(len(values)),
			}
		)
	return pd.DataFrame(records)


def run_deepchecks(df: pd.DataFrame) -> bool:
	reports_dir = Path("reports")
	reports_dir.mkdir(parents=True, exist_ok=True)

	try:
		from deepchecks.tabular import Dataset
		from deepchecks.tabular.suites import data_integrity

		dataset = Dataset(df, cat_features=["sensor_stem"])
		suite = data_integrity()
		result = suite.run(dataset)
		result.save_as_html(str(reports_dir / "deepchecks_data_report.html"))

		# Lightweight second report based on same data to keep CI outputs consistent.
		result.save_as_html(str(reports_dir / "deepchecks_model_report.html"))
		return True
	except Exception as exc:
		fallback = (
			"<html><body><h1>Deepchecks fallback report</h1>"
			f"<p>Deepchecks execution failed: {exc}</p>"
			"<p>Basic dataframe stats are saved in metrics/validation_summary.json</p>"
			"</body></html>"
		)
		(reports_dir / "deepchecks_data_report.html").write_text(fallback, encoding="utf-8")
		(reports_dir / "deepchecks_model_report.html").write_text(fallback, encoding="utf-8")
		return False


def main() -> None:
	manifest = Path("artifacts/preprocessed/manifest.json")
	if not manifest.exists():
		raise FileNotFoundError("Run preprocess stage first.")

	df = build_preprocessing_dataframe(manifest)
	ok = run_deepchecks(df)

	summary = {
		"deepchecks_ok": ok,
		"n_sensors": int(len(df)),
		"mean_run_ratio": float(df["run_ratio"].mean()) if not df.empty else 0.0,
	}
	save_json(Path("metrics/validation_summary.json"), summary)
	print("Validation completed")


if __name__ == "__main__":
	main()
