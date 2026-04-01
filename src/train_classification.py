"""
CWT + CNN Supervised Classification with MLflow tracking.

For each sensor profile the script:
  1. Loads and windows the raw signals
  2. Injects synthetic anomalies (15 % ratio)
  3. Computes CWT scalograms
  4. Trains a CNN classifier (7 classes: normal + 6 anomaly types)
  5. Evaluates and logs metrics + artifacts to MLflow
  6. Saves the best model and writes metrics/classification_metrics.json

Run:
    python src/train_classification.py
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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.preprocess import (
    ANOMALY_TYPES,
    LABEL_MAP,
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

BATCH_SIZE = P["classification"]["batch_size"]
EPOCHS = P["classification"]["epochs"]
LR = P["classification"]["learning_rate"]
ANOMALY_RATIO = P["classification"]["anomaly_ratio"]
OUTPUT_DIR = Path(P["classification"]["output_dir"])
OUTPUT_DIR.mkdir(exist_ok=True)

METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = len(LABEL_NAMES)  # 7

mlflow.set_tracking_uri(P["mlflow"]["tracking_uri"])
mlflow.set_experiment(P["mlflow"]["experiment_classification"])


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers: list = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CWT_CNN_Classifier(nn.Module):
    def __init__(self, n_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128, pool=False),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_loaders(
    windows: np.ndarray,
    labels: np.ndarray,
) -> tuple[DataLoader, DataLoader]:
    X_train, X_test, y_train, y_test = train_test_split(
        windows, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    scalograms_train = compute_cwt(X_train, SCALES, WAVELET)
    scalograms_test = compute_cwt(X_test, SCALES, WAVELET)

    t_train = torch.tensor(scalograms_train, dtype=torch.float32)
    t_test = torch.tensor(scalograms_test, dtype=torch.float32)
    l_train = torch.tensor(y_train, dtype=torch.long)
    l_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(t_train, l_train), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(t_test, l_test), batch_size=BATCH_SIZE, shuffle=False
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        total_correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        out = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * len(y)
        preds = out.argmax(1)
        total_correct += (preds == y).sum().item()
        total += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return (
        total_loss / total,
        total_correct / total,
        np.array(all_preds),
        np.array(all_labels),
    )


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    title: str = "Confusion Matrix",
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_training_history(
    train_losses: list,
    test_losses: list,
    train_accs: list,
    test_accs: list,
    path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train")
    ax1.plot(test_losses, label="Test")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(train_accs, label="Train")
    ax2.plot(test_accs, label="Test")
    ax2.set_title("Accuracy")
    ax2.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_metrics_per_type(
    report: dict,
    path: Path,
    profile_name: str,
) -> None:
    types = ANOMALY_TYPES
    precision = [report.get(LABEL_NAMES[LABEL_MAP[t]], {}).get("precision", 0) for t in types]
    recall = [report.get(LABEL_NAMES[LABEL_MAP[t]], {}).get("recall", 0) for t in types]
    f1 = [report.get(LABEL_NAMES[LABEL_MAP[t]], {}).get("f1-score", 0) for t in types]

    x = np.arange(len(types))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=30)
    ax.set_title(f"{profile_name} — Metrics per anomaly type")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop (one profile)
# ─────────────────────────────────────────────────────────────────────────────

def train_profile(profile_name: str, sensor_names: list) -> dict:
    print(f"\n{'='*70}")
    print(f"  Profile: {profile_name} | Sensors: {sensor_names}")
    print(f"{'='*70}")

    signal = load_profile(DATA_DIR, sensor_names)
    windows = sliding_windows(signal, WINDOW_SIZE, STEP_SIZE)
    windows, labels = build_anomaly_dataset(windows, ANOMALY_RATIO, RNG)

    train_loader, test_loader = make_loaders(windows, labels)

    model = CWT_CNN_Classifier(n_classes=N_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    with mlflow.start_run(run_name=profile_name):
        # Log parameters
        mlflow.log_params({
            "profile": profile_name,
            "sensors": str(sensor_names),
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "anomaly_ratio": ANOMALY_RATIO,
            "wavelet": WAVELET,
            "n_scales": len(SCALES),
            "seed": SEED,
        })

        best_f1 = 0.0
        best_model_path = OUTPUT_DIR / f"{profile_name}_model.pth"

        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            te_loss, te_acc, preds, labels_true = evaluate(model, test_loader, criterion)
            scheduler.step(te_loss)

            train_losses.append(tr_loss)
            test_losses.append(te_loss)
            train_accs.append(tr_acc)
            test_accs.append(te_acc)

            f1 = f1_score(labels_true, preds, average="weighted", zero_division=0)

            mlflow.log_metrics(
                {
                    "train_loss": tr_loss,
                    "test_loss": te_loss,
                    "train_acc": tr_acc,
                    "test_acc": te_acc,
                    "test_f1_weighted": f1,
                },
                step=epoch,
            )

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), best_model_path)

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={te_loss:.4f}  acc={te_acc:.4f}  f1={f1:.4f}")

        # Load best checkpoint for final evaluation
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        _, _, final_preds, final_labels = evaluate(model, test_loader, criterion)

        acc = accuracy_score(final_labels, final_preds)
        prec = precision_score(final_labels, final_preds, average="weighted", zero_division=0)
        rec = recall_score(final_labels, final_preds, average="weighted", zero_division=0)
        f1_final = f1_score(final_labels, final_preds, average="weighted", zero_division=0)
        report = classification_report(
            final_labels, final_preds,
            target_names=LABEL_NAMES, output_dict=True, zero_division=0,
        )

        mlflow.log_metrics({
            "best_accuracy": acc,
            "best_precision": prec,
            "best_recall": rec,
            "best_f1": f1_final,
        })

        # Plots
        history_path = OUTPUT_DIR / f"{profile_name}_training_history.png"
        cm_path = OUTPUT_DIR / f"{profile_name}_confusion_matrix.png"
        metrics_path = OUTPUT_DIR / f"{profile_name}_metrics_by_type.png"

        save_training_history(train_losses, test_losses, train_accs, test_accs, history_path)
        save_confusion_matrix(final_labels, final_preds, cm_path, f"{profile_name} Confusion Matrix")
        save_metrics_per_type(report, metrics_path, profile_name)

        mlflow.log_artifact(str(best_model_path))
        mlflow.log_artifact(str(history_path))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(metrics_path))

        # Register model to MLflow Model Registry
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"classification-{profile_name}",
        )

        print(f"  ✓ {profile_name}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1_final:.4f}")

    return {
        "profile": profile_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1_final,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Window: {WINDOW_SIZE}  Scales: {len(SCALES)}  Wavelet: {WAVELET}")
    print(f"MLflow experiment: {P['mlflow']['experiment_classification']}")

    all_results = {}
    for profile_name, sensors in PROFILES.items():
        result = train_profile(profile_name, sensors)
        all_results[profile_name] = result

    # Save consolidated metrics for DVC
    METRICS_DIR.mkdir(exist_ok=True)
    metrics_file = METRICS_DIR / "classification_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved → {metrics_file}")

    # Global comparison plot
    profiles = list(all_results.keys())
    accs = [all_results[p]["accuracy"] for p in profiles]
    f1s = [all_results[p]["f1"] for p in profiles]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(profiles))
    ax.bar(x - 0.2, accs, 0.35, label="Accuracy")
    ax.bar(x + 0.2, f1s, 0.35, label="F1 (weighted)")
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_title("Classification — Global Profile Comparison")
    ax.legend()
    plt.tight_layout()
    global_plot = OUTPUT_DIR / "global_comparison.png"
    fig.savefig(global_plot, dpi=120)
    plt.close(fig)
    print(f"Global comparison saved → {global_plot}")


if __name__ == "__main__":
    main()
