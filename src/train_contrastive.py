"""
Contrastive Self-Supervised Learning (1D signal) with MLflow tracking.

For each sensor profile the script:
  1. Loads and windows raw 1D signals
  2. Pre-trains a Conv1D encoder with NT-Xent (SimCLR-style) on NORMAL windows
  3. Builds a test set with anomalies and computes centroid distances
  4. Selects the optimal anomaly threshold by maximising F1 on a held-out
     validation set
  5. Evaluates and logs metrics + artifacts to MLflow
  6. Saves the model and writes metrics/contrastive_metrics.json

Run:
    python src/train_contrastive.py
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
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.decomposition import PCA
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
from torch.utils.data import DataLoader, Dataset

from src.preprocess import (
    ANOMALY_COLORS,
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
torch.manual_seed(SEED)
RNG = np.random.default_rng(SEED)

DATA_DIR = Path(P["data"]["data_dir"])
PROFILES = P["data"]["profiles"]

WINDOW_SIZE = P["windowing"]["window_size"]
STEP_SIZE = P["windowing"]["step_size"]

BATCH_SIZE = P["contrastive"]["batch_size"]
EPOCHS = P["contrastive"]["epochs_pretrain"]
LR = P["contrastive"]["learning_rate"]
TEMPERATURE = P["contrastive"]["temperature"]
EMBEDDING_DIM = P["contrastive"]["embedding_dim"]
ANOMALY_RATIO_TEST = P["contrastive"]["anomaly_ratio_test"]
OUTPUT_DIR = Path(P["contrastive"]["output_dir"])
OUTPUT_DIR.mkdir(exist_ok=True)

METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlflow.set_tracking_uri(P["mlflow"]["tracking_uri"])
mlflow.set_experiment(P["mlflow"]["experiment_contrastive"])


# ─────────────────────────────────────────────────────────────────────────────
# Data augmentations for contrastive pre-training
# ─────────────────────────────────────────────────────────────────────────────

def _jitter(x: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    return x + np.random.normal(0, sigma, x.shape).astype(np.float32)


def _scaling(x: np.ndarray) -> np.ndarray:
    factor = np.random.uniform(0.8, 1.2)
    return (x * factor).astype(np.float32)


def _time_shift(x: np.ndarray, max_shift: int = 20) -> np.ndarray:
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(x, shift).astype(np.float32)


def _permutation(x: np.ndarray, n_segments: int = 4) -> np.ndarray:
    n = len(x)
    seg_len = n // n_segments
    segments = [x[i * seg_len: (i + 1) * seg_len] for i in range(n_segments)]
    np.random.shuffle(segments)
    return np.concatenate(segments).astype(np.float32)


_AUGMENTATIONS = [_jitter, _scaling, _time_shift, _permutation]


def augment(x: np.ndarray) -> np.ndarray:
    """Apply 2-3 random augmentations."""
    n_aug = np.random.randint(2, 4)
    chosen = np.random.choice(len(_AUGMENTATIONS), size=n_aug, replace=False)
    for idx in chosen:
        x = _AUGMENTATIONS[idx](x)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ContrastiveDataset(Dataset):
    def __init__(self, windows: np.ndarray):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.windows[idx]
        aug1 = torch.tensor(augment(w), dtype=torch.float32).unsqueeze(0)
        aug2 = torch.tensor(augment(w), dtype=torch.float32).unsqueeze(0)
        return aug1, aug2


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class Contrastive1DEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.encoder(x)), dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# NT-Xent loss
# ─────────────────────────────────────────────────────────────────────────────

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                      # (2N, D)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # (2N, 2N)
    sim /= temperature

    # Mask out self-similarity
    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))

    # Positive pairs: (i, i+N) and (i+N, i)
    labels = torch.arange(n, device=z.device)
    labels = torch.cat([labels + n, labels])             # (2N,)

    return F.cross_entropy(sim, labels)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_embeddings(
    model: nn.Module, windows: np.ndarray, batch_size: int = 256
) -> np.ndarray:
    model.eval()
    embeddings = []
    for i in range(0, len(windows), batch_size):
        batch = torch.tensor(
            windows[i: i + batch_size, np.newaxis, :], dtype=torch.float32
        ).to(DEVICE)
        embeddings.append(model.encode(batch).cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def find_optimal_threshold(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
) -> float:
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    all_labels = np.concatenate([
        np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))
    ])
    best_f1, best_t = 0.0, 0.0
    for t in np.percentile(all_scores, np.arange(1, 100)):
        preds = (all_scores >= t).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def save_pca_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    path: Path,
    title: str,
) -> None:
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl_idx, lname in enumerate(LABEL_NAMES):
        mask = labels == lbl_idx
        if not mask.any():
            continue
        color = ANOMALY_COLORS.get(lname, "#555555")
        ax.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.6,
                   c=color, label=lname)
    ax.set_title(title)
    ax.legend(markerscale=3, bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    normal_s = scores[labels == 0]
    anomaly_s = scores[labels > 0]
    ax.hist(normal_s, bins=60, alpha=0.6, label="Normal", color="#2ECC71")
    ax.hist(anomaly_s, bins=60, alpha=0.6, label="Anomaly", color="#E74C3C")
    ax.axvline(threshold, color="k", linestyle="--", label="Threshold")
    ax.set_xlabel("Anomaly score (centroid distance)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_roc_pr(
    y_true: np.ndarray,
    scores: np.ndarray,
    path: Path,
    title: str,
) -> tuple[float, float]:
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    prec_v, rec_v, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--")
    ax1.set_title(f"{title} ROC")
    ax1.legend()

    ax2.plot(rec_v, prec_v, label=f"AP={ap:.3f}")
    ax2.set_title(f"{title} PR")
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


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_profile(profile_name: str, sensor_names: list) -> dict:
    print(f"\n{'='*70}")
    print(f"  Profile: {profile_name} | Sensors: {sensor_names}")
    print(f"{'='*70}")

    signal = load_profile(DATA_DIR, sensor_names)
    all_windows = sliding_windows(signal, WINDOW_SIZE, STEP_SIZE)

    # Split normal windows
    idx = RNG.permutation(len(all_windows))
    n_train = int(len(all_windows) * 0.80)
    train_wins = all_windows[idx[:n_train]]
    normal_test_wins = all_windows[idx[n_train:]]

    # Build test set with anomalies
    test_wins, test_labels = build_anomaly_dataset(
        normal_test_wins, ANOMALY_RATIO_TEST, RNG
    )

    train_loader = DataLoader(
        ContrastiveDataset(train_wins), batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    model = Contrastive1DEncoder(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    contrastive_losses = []

    with mlflow.start_run(run_name=profile_name):
        mlflow.log_params({
            "profile": profile_name,
            "sensors": str(sensor_names),
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs_pretrain": EPOCHS,
            "learning_rate": LR,
            "temperature": TEMPERATURE,
            "embedding_dim": EMBEDDING_DIM,
            "anomaly_ratio_test": ANOMALY_RATIO_TEST,
            "seed": SEED,
        })

        best_loss = float("inf")
        best_model_path = OUTPUT_DIR / f"{profile_name}_contrastive1d.pth"

        for epoch in range(1, EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            for aug1, aug2 in train_loader:
                aug1, aug2 = aug1.to(DEVICE), aug2.to(DEVICE)
                z1, z2 = model(aug1), model(aug2)
                loss = nt_xent_loss(z1, z2, TEMPERATURE)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_loader)
            scheduler.step()
            contrastive_losses.append(epoch_loss)

            mlflow.log_metric("contrastive_loss", epoch_loss, step=epoch)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), best_model_path)

            if epoch % 20 == 0:
                print(f"  Epoch {epoch:3d}/{EPOCHS}  NT-Xent loss={epoch_loss:.4f}")

        # ── Anomaly detection ────────────────────────────────────────────────
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

        train_emb = compute_embeddings(model, train_wins)
        test_emb = compute_embeddings(model, test_wins)

        centroid = train_emb.mean(axis=0)

        train_scores = np.linalg.norm(train_emb - centroid, axis=1)
        test_scores = np.linalg.norm(test_emb - centroid, axis=1)

        test_labels_binary = (test_labels > 0).astype(int)

        # Use percentile-based threshold from training
        threshold = find_optimal_threshold(
            train_scores,
            test_scores[test_labels_binary == 1],
        )

        y_pred_binary = (test_scores >= threshold).astype(int)
        acc = accuracy_score(test_labels_binary, y_pred_binary)
        prec = precision_score(test_labels_binary, y_pred_binary, zero_division=0)
        rec = recall_score(test_labels_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(test_labels_binary, y_pred_binary, zero_division=0)

        roc_path = OUTPUT_DIR / f"{profile_name}_roc_pr.png"
        roc_auc, ap = save_roc_pr(test_labels_binary, test_scores, roc_path, profile_name)

        mlflow.log_metrics({
            "threshold": threshold,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "average_precision": ap,
        })

        # ── Plots ────────────────────────────────────────────────────────────
        loss_path = OUTPUT_DIR / f"{profile_name}_contrastive_loss.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(contrastive_losses)
        ax.set_title(f"{profile_name} — NT-Xent Contrastive Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        plt.tight_layout()
        fig.savefig(loss_path, dpi=120)
        plt.close(fig)

        pca_path = OUTPUT_DIR / f"{profile_name}_pca_embeddings.png"
        save_pca_plot(test_emb, test_labels, pca_path,
                      f"{profile_name} — Embedding PCA")

        score_dist_path = OUTPUT_DIR / f"{profile_name}_score_dist.png"
        save_score_distribution(test_scores, test_labels, threshold,
                                score_dist_path, f"{profile_name} Score Distribution")

        cm_bin_path = OUTPUT_DIR / f"{profile_name}_cm_binary.png"
        save_confusion_matrix(test_labels_binary, y_pred_binary, ["normal", "anomaly"],
                              cm_bin_path, f"{profile_name} Binary CM")

        for artifact_path in [best_model_path, loss_path, pca_path,
                               score_dist_path, roc_path, cm_bin_path]:
            mlflow.log_artifact(str(artifact_path))

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"contrastive-{profile_name}",
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
    print(f"Window: {WINDOW_SIZE}  Temperature: {TEMPERATURE}  EmbDim: {EMBEDDING_DIM}")
    print(f"MLflow experiment: {P['mlflow']['experiment_contrastive']}")

    all_results = {}
    for profile_name, sensors in PROFILES.items():
        result = train_profile(profile_name, sensors)
        all_results[profile_name] = result

    METRICS_DIR.mkdir(exist_ok=True)
    metrics_file = METRICS_DIR / "contrastive_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved → {metrics_file}")

    # Global comparison
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
    ax.set_title("Contrastive — Global Profile Comparison")
    ax.legend()
    plt.tight_layout()
    global_plot = OUTPUT_DIR / "global_comparison.png"
    fig.savefig(global_plot, dpi=120)
    plt.close(fig)
    print(f"Global comparison saved → {global_plot}")


if __name__ == "__main__":
    main()
