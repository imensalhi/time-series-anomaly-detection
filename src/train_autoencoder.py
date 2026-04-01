"""
CWT + CNN Autoencoder (Unsupervised) with MLflow tracking.

For each sensor profile the script:
  1. Loads and windows the raw signals
  2. Trains the autoencoder ONLY on normal windows
  3. Builds a test set mixing normal + anomaly windows
  4. Computes CWT scalograms and reconstruction errors
  5. Picks threshold at the 95th percentile of training errors
  6. Evaluates and logs metrics + artifacts to MLflow
  7. Saves the best model and writes metrics/autoencoder_metrics.json

Run:
    python src/train_autoencoder.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

from src.preprocess import (
    LABEL_NAMES,
    build_anomaly_dataset,
    compute_cwt,
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
torch.manual_seed(SEED)
RNG = np.random.default_rng(SEED)

DATA_DIR = Path(P["data"]["data_dir"])
PROFILES = P["data"]["profiles"]

WINDOW_SIZE = P["windowing"]["window_size"]
STEP_SIZE = P["windowing"]["step_size"]

SCALES = np.arange(P["cwt"]["scales_min"], P["cwt"]["scales_max"])
WAVELET = P["cwt"]["wavelet"]

BATCH_SIZE = P["autoencoder"]["batch_size"]
EPOCHS = P["autoencoder"]["epochs"]
LR = P["autoencoder"]["learning_rate"]
ANOMALY_RATIO_TEST = P["autoencoder"]["anomaly_ratio_test"]
THRESHOLD_PERCENTILE = P["autoencoder"]["threshold_percentile"]
OUTPUT_DIR = Path(P["autoencoder"]["output_dir"])
OUTPUT_DIR.mkdir(exist_ok=True)

METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlflow.set_tracking_uri(P["mlflow"]["tracking_uri"])
mlflow.set_experiment(P["mlflow"]["experiment_autoencoder"])


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

class CWT_CNN_Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_errors(
    model: nn.Module,
    scalograms: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Return per-sample MSE reconstruction errors."""
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(scalograms), batch_size):
            batch = torch.tensor(scalograms[i: i + batch_size], dtype=torch.float32).to(DEVICE)
            recon = model(batch)
            mse = ((batch - recon) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            errors.extend(mse)
    return np.array(errors, dtype=np.float32)


def save_roc_pr(
    y_true_binary: np.ndarray,
    scores: np.ndarray,
    path: Path,
    title: str,
) -> tuple[float, float]:
    fpr, tpr, _ = roc_curve(y_true_binary, scores)
    roc_auc = auc(fpr, tpr)
    prec_vals, rec_vals, _ = precision_recall_curve(y_true_binary, scores)
    ap = average_precision_score(y_true_binary, scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--")
    ax1.set_title(f"{title} — ROC")
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.legend()

    ax2.plot(rec_vals, prec_vals, label=f"AP={ap:.3f}")
    ax2.set_title(f"{title} — Precision-Recall")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return roc_auc, ap


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list,
    path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_error_distribution(
    normal_errors: np.ndarray,
    anomaly_errors: np.ndarray,
    threshold: float,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(normal_errors, bins=60, alpha=0.6, label="Normal", color="#2ECC71")
    ax.hist(anomaly_errors, bins=60, alpha=0.6, label="Anomaly", color="#E74C3C")
    ax.axvline(threshold, color="k", linestyle="--", label=f"Threshold ({THRESHOLD_PERCENTILE}th pct)")
    ax.set_xlabel("Reconstruction error (MSE)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_profile(profile_name: str, sensor_names: list) -> dict:
    print(f"\n{'='*70}")
    print(f"  Profile: {profile_name} | Sensors: {sensor_names}")
    print(f"{'='*70}")

    signal = load_profile(DATA_DIR, sensor_names)
    all_windows = sliding_windows(signal, WINDOW_SIZE, STEP_SIZE)

    # Train only on normal windows (80 % of data)
    n_total = len(all_windows)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.10)
    idx = RNG.permutation(n_total)
    train_wins = all_windows[idx[:n_train]]
    val_wins = all_windows[idx[n_train: n_train + n_val]]
    normal_test_wins = all_windows[idx[n_train + n_val:]]

    # Build test set with anomalies
    test_wins, test_labels = build_anomaly_dataset(
        normal_test_wins, ANOMALY_RATIO_TEST, RNG
    )

    # CWT
    scalo_train = compute_cwt(train_wins, SCALES, WAVELET)
    scalo_val = compute_cwt(val_wins, SCALES, WAVELET)
    scalo_test = compute_cwt(test_wins, SCALES, WAVELET)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(scalo_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(scalo_val)),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    model = CWT_CNN_Autoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []

    with mlflow.start_run(run_name=profile_name):
        mlflow.log_params({
            "profile": profile_name,
            "sensors": str(sensor_names),
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "anomaly_ratio_test": ANOMALY_RATIO_TEST,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "wavelet": WAVELET,
            "n_scales": len(SCALES),
            "seed": SEED,
        })

        best_val_loss = float("inf")
        best_model_path = OUTPUT_DIR / f"{profile_name}_autoencoder.pth"

        for epoch in range(1, EPOCHS + 1):
            # Training step
            model.train()
            tr_loss = 0.0
            for (X,) in train_loader:
                X = X.to(DEVICE)
                optimizer.zero_grad()
                recon = model(X)
                loss = criterion(recon, X)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * len(X)
            tr_loss /= len(train_loader.dataset)

            # Validation step
            model.eval()
            vl_loss = 0.0
            with torch.no_grad():
                for (X,) in val_loader:
                    X = X.to(DEVICE)
                    recon = model(X)
                    vl_loss += criterion(recon, X).item() * len(X)
            vl_loss /= len(val_loader.dataset)

            train_losses.append(tr_loss)
            val_losses.append(vl_loss)

            mlflow.log_metrics(
                {"train_loss": tr_loss, "val_loss": vl_loss}, step=epoch
            )

            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                torch.save(model.state_dict(), best_model_path)

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}/{EPOCHS}  train={tr_loss:.5f}  val={vl_loss:.5f}")

        # ── Threshold from training errors ──────────────────────────────────
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        train_errors = compute_errors(model, scalo_train)
        threshold = float(np.percentile(train_errors, THRESHOLD_PERCENTILE))

        # ── Test evaluation ──────────────────────────────────────────────────
        test_errors = compute_errors(model, scalo_test)
        y_pred_binary = (test_errors > threshold).astype(int)
        y_true_binary = (test_labels > 0).astype(int)

        acc = accuracy_score(y_true_binary, y_pred_binary)
        prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        roc_path = OUTPUT_DIR / f"{profile_name}_roc_pr.png"
        roc_auc, ap = save_roc_pr(y_true_binary, test_errors, roc_path, profile_name)

        mlflow.log_metrics({
            "threshold": threshold,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "average_precision": ap,
        })

        # ── Training loss plot ───────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_losses, label="Train")
        ax.plot(val_losses, label="Val")
        ax.set_title(f"{profile_name} — Autoencoder Training Loss")
        ax.legend()
        plt.tight_layout()
        loss_path = OUTPUT_DIR / f"{profile_name}_ae_training_loss.png"
        fig.savefig(loss_path, dpi=120)
        plt.close(fig)

        # ── Error distribution ───────────────────────────────────────────────
        normal_errors_test = test_errors[y_true_binary == 0]
        anomaly_errors_test = test_errors[y_true_binary == 1]
        err_dist_path = OUTPUT_DIR / f"{profile_name}_error_distribution.png"
        save_error_distribution(normal_errors_test, anomaly_errors_test, threshold,
                                err_dist_path, f"{profile_name} Error Distribution")

        # ── Confusion matrix (binary) ────────────────────────────────────────
        cm_bin_path = OUTPUT_DIR / f"{profile_name}_cm_binary.png"
        save_confusion_matrix(y_true_binary, y_pred_binary, ["normal", "anomaly"],
                              cm_bin_path, f"{profile_name} Binary Confusion Matrix")

        # ── Confusion matrix (multi-class) ───────────────────────────────────
        y_pred_multi = np.where(y_pred_binary == 0, 0, test_labels)
        cm_multi_path = OUTPUT_DIR / f"{profile_name}_cm_multiclass.png"
        save_confusion_matrix(test_labels, y_pred_multi, LABEL_NAMES,
                              cm_multi_path, f"{profile_name} Multi-class Confusion")

        for artifact_path in [best_model_path, loss_path, err_dist_path,
                               roc_path, cm_bin_path, cm_multi_path]:
            mlflow.log_artifact(str(artifact_path))

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"autoencoder-{profile_name}",
        )

        print(f"  ✓ {profile_name}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} "
              f"f1={f1:.4f} roc_auc={roc_auc:.4f}")

    return {
        "profile": profile_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": ap,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Window: {WINDOW_SIZE}  Scales: {len(SCALES)}  Wavelet: {WAVELET}")
    print(f"MLflow experiment: {P['mlflow']['experiment_autoencoder']}")

    all_results = {}
    for profile_name, sensors in PROFILES.items():
        result = train_profile(profile_name, sensors)
        all_results[profile_name] = result

    METRICS_DIR.mkdir(exist_ok=True)
    metrics_file = METRICS_DIR / "autoencoder_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved → {metrics_file}")

    # Global comparison plot
    profiles = list(all_results.keys())
    accs = [all_results[p]["accuracy"] for p in profiles]
    f1s = [all_results[p]["f1"] for p in profiles]
    rocs = [all_results[p]["roc_auc"] for p in profiles]

    x = np.arange(len(profiles))
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.25
    ax.bar(x - w, accs, w, label="Accuracy")
    ax.bar(x, f1s, w, label="F1")
    ax.bar(x + w, rocs, w, label="ROC AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_title("Autoencoder — Global Profile Comparison")
    ax.legend()
    plt.tight_layout()
    global_plot = OUTPUT_DIR / "global_comparison.png"
    fig.savefig(global_plot, dpi=120)
    plt.close(fig)
    print(f"Global comparison saved → {global_plot}")


if __name__ == "__main__":
    main()
