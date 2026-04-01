"""
Deepchecks Validation — Data Integrity + Model Performance Checks.

This script runs two Deepchecks suites:
  1. **Data Integrity Suite** – checks raw time-series windows for issues
     (feature drift, label distribution, missing values, …)
  2. **Model Performance Suite** – evaluates the classification model on a
     held-out test set and generates an HTML report.

Results are logged to MLflow and saved as HTML reports under reports/.

Run:
    python src/validate.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import mlflow
import numpy as np
import pandas as pd
import torch
import yaml

from src.preprocess import (
    LABEL_NAMES,
    build_anomaly_dataset,
    load_profile,
    sliding_windows,
)

# ─────────────────────────────────────────────────────────────────────────────
# Load params
# ─────────────────────────────────────────────────────────────────────────────
with open("params.yaml") as f:
    P = yaml.safe_load(f)

SEED = P["seed"]
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

DATA_DIR = Path(P["data"]["data_dir"])
PROFILES = P["data"]["profiles"]
WINDOW_SIZE = P["windowing"]["window_size"]
STEP_SIZE = P["windowing"]["step_size"]

CLASSIFICATION_DIR = Path(P["classification"]["output_dir"])
ANOMALY_RATIO = P["classification"]["anomaly_ratio"]

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(P["mlflow"]["tracking_uri"])
mlflow.set_experiment("deepchecks-validation")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load all profile data into DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_dataframe(
    windows: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Convert windows into a flat feature DataFrame:
      - Statistical features (mean, std, min, max, skew, kurtosis)
      - Label column
    """
    means = windows.mean(axis=1)
    stds = windows.std(axis=1)
    mins = windows.min(axis=1)
    maxs = windows.max(axis=1)
    ranges = maxs - mins

    # Approximate skewness and kurtosis without scipy
    z = (windows - means[:, None]) / (stds[:, None] + 1e-8)
    skew = (z ** 3).mean(axis=1)
    kurt = (z ** 4).mean(axis=1) - 3.0

    df = pd.DataFrame({
        "mean": means,
        "std": stds,
        "min": mins,
        "max": maxs,
        "range": ranges,
        "skewness": skew,
        "kurtosis": kurt,
        "label": labels,
        "label_name": [LABEL_NAMES[int(lbl)] for lbl in labels],
    })
    return df


def run_data_checks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    profile_name: str,
) -> dict:
    """
    Run Deepchecks data integrity and train/test validation suite.

    Returns a dict of check summaries.
    """
    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import data_integrity, train_test_validation

        feature_cols = ["mean", "std", "min", "max", "range", "skewness", "kurtosis"]
        label_col = "label"

        dc_train = Dataset(
            train_df[feature_cols + [label_col]],
            label=label_col,
            cat_features=[],
        )
        dc_test = Dataset(
            test_df[feature_cols + [label_col]],
            label=label_col,
            cat_features=[],
        )

        # ── Suite 1: Data Integrity ──────────────────────────────────────────
        integrity_suite = data_integrity()
        integrity_result = integrity_suite.run(dc_train)

        # ── Suite 2: Train / Test Validation ────────────────────────────────
        validation_suite = train_test_validation()
        validation_result = validation_suite.run(dc_train, dc_test)

        return {
            "integrity_passed": integrity_result.passed_conditions_count(),
            "integrity_failed": integrity_result.failed_conditions_count(),
            "validation_passed": validation_result.passed_conditions_count(),
            "validation_failed": validation_result.failed_conditions_count(),
            "integrity_result": integrity_result,
            "validation_result": validation_result,
        }
    except ImportError as exc:
        print(f"  [WARN] Deepchecks not available: {exc}. Skipping data checks.")
        return {
            "integrity_passed": 0,
            "integrity_failed": 0,
            "validation_passed": 0,
            "validation_failed": 0,
            "integrity_result": None,
            "validation_result": None,
        }


def run_model_checks(
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_scalograms: np.ndarray,
    test_labels: np.ndarray,
    profile_name: str,
) -> dict:
    """
    Run Deepchecks model performance checks using the classification model.

    Because Deepchecks' model performance suite works best with sklearn-
    compatible predictors, we wrap the PyTorch model in a minimal adapter.
    """
    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import model_evaluation

        feature_cols = ["mean", "std", "min", "max", "range", "skewness", "kurtosis"]
        label_col = "label"

        # sklearn-compatible wrapper
        class TorchWrapper:
            def predict(self, X):
                feats = torch.tensor(test_scalograms, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    logits = model(feats)
                return logits.argmax(dim=1).cpu().numpy()

            def predict_proba(self, X):
                feats = torch.tensor(test_scalograms, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    logits = model(feats)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                return probs

        dc_train = Dataset(
            train_df[feature_cols + [label_col]],
            label=label_col,
            cat_features=[],
        )
        dc_test = Dataset(
            test_df[feature_cols + [label_col]],
            label=label_col,
            cat_features=[],
        )
        wrapper = TorchWrapper()
        suite = model_evaluation()
        result = suite.run(dc_train, dc_test, wrapper)

        return {
            "model_passed": result.passed_conditions_count(),
            "model_failed": result.failed_conditions_count(),
            "model_result": result,
        }
    except ImportError as exc:
        print(f"  [WARN] Deepchecks not available: {exc}. Skipping model checks.")
        return {"model_passed": 0, "model_failed": 0, "model_result": None}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def validate_profile(profile_name: str, sensor_names: list) -> None:
    print(f"\n{'='*70}")
    print(f"  Validating: {profile_name} | Sensors: {sensor_names}")
    print(f"{'='*70}")

    signal = load_profile(DATA_DIR, sensor_names)
    all_windows = sliding_windows(signal, WINDOW_SIZE, STEP_SIZE)

    idx = RNG.permutation(len(all_windows))
    n_train = int(len(all_windows) * 0.70)
    train_wins = all_windows[idx[:n_train]]
    normal_test = all_windows[idx[n_train:]]

    # Inject anomalies into test
    test_wins, test_labels = build_anomaly_dataset(
        normal_test, ANOMALY_RATIO, RNG
    )
    train_labels = np.zeros(len(train_wins), dtype=np.int64)

    train_df = build_feature_dataframe(train_wins, train_labels)
    test_df = build_feature_dataframe(test_wins, test_labels)

    with mlflow.start_run(run_name=f"deepchecks-{profile_name}"):
        mlflow.log_params({
            "profile": profile_name,
            "n_train_windows": len(train_wins),
            "n_test_windows": len(test_wins),
        })

        # ── Data checks ─────────────────────────────────────────────────────
        data_results = run_data_checks(train_df, test_df, profile_name)
        mlflow.log_metrics({
            "integrity_passed": data_results["integrity_passed"],
            "integrity_failed": data_results["integrity_failed"],
            "validation_passed": data_results["validation_passed"],
            "validation_failed": data_results["validation_failed"],
        })

        # Save HTML reports
        data_report_path = REPORTS_DIR / "deepchecks_data_report.html"
        if data_results["integrity_result"] is not None:
            data_results["integrity_result"].save_as_html(str(data_report_path))
            mlflow.log_artifact(str(data_report_path))
            print(f"  Data integrity report → {data_report_path}")
        else:
            # Create a minimal placeholder report so DVC output is satisfied
            data_report_path.write_text(
                "<html><body><h1>Deepchecks not installed — skipped</h1></body></html>"
            )

        model_report_path = REPORTS_DIR / "deepchecks_model_report.html"
        # ── Model checks (classification model for this profile) ─────────────
        model_path = CLASSIFICATION_DIR / f"{profile_name}_model.pth"
        if model_path.exists():
            from src.train_classification import CWT_CNN_Classifier
            from src.preprocess import compute_cwt

            SCALES = np.arange(P["cwt"]["scales_min"], P["cwt"]["scales_max"])
            WAVELET = P["cwt"]["wavelet"]
            scalograms = compute_cwt(test_wins, SCALES, WAVELET)

            model = CWT_CNN_Classifier(n_classes=len(LABEL_NAMES)).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()

            model_results = run_model_checks(
                model, train_df, test_df, scalograms, test_labels, profile_name
            )
            mlflow.log_metrics({
                "model_passed": model_results["model_passed"],
                "model_failed": model_results["model_failed"],
            })

            if model_results["model_result"] is not None:
                model_results["model_result"].save_as_html(str(model_report_path))
                mlflow.log_artifact(str(model_report_path))
                print(f"  Model evaluation report → {model_report_path}")
            else:
                model_report_path.write_text(
                    "<html><body><h1>Deepchecks not installed — skipped</h1></body></html>"
                )
        else:
            print(f"  [WARN] Model not found at {model_path}. Run train_classification first.")
            model_report_path.write_text(
                "<html><body><h1>Model not found — run training first</h1></body></html>"
            )

        print(f"  ✓ Data integrity: {data_results['integrity_passed']} passed / "
              f"{data_results['integrity_failed']} failed")


def main() -> None:
    print("Running Deepchecks validation …")
    # Only validate first profile by default to keep runtime reasonable in CI
    for profile_name, sensors in PROFILES.items():
        validate_profile(profile_name, sensors)
        break  # Remove this break to validate all profiles


if __name__ == "__main__":
    main()
