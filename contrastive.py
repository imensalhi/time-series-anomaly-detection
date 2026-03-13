"""
=============================================================================
Contrastive Learning (Self-Supervised) — Signal 1D Direct (sans CWT/CNN 2D)
=============================================================================
Approche :
- Encoder Conv1D sur les fenêtres de signal brut (pas de CWT)
- SimCLR-style : 2 augmentations par fenêtre → paire positive
- NT-Xent loss pour apprendre des représentations discriminantes
- Entraîné UNIQUEMENT sur signal normal
- Détection : distance au centroïde dans l'espace latent
- Seuil OPTIMAL par maximisation du F1 sur validation

Architecture :
  Signal 1D (128 pts) → Conv1D Encoder → Embedding (128-dim) → Projection Head
  Pas de CWT, pas de CNN 2D — tout est en 1D.

Profils capteurs :
    Profil 1 (Stable)   : I23, I61
    Profil 2 (Bimodal)  : I42, I43
    Profil 3 (Multi)    : I52, I11, I113, I32
=============================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# Configuration
# =====================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

WINDOW_SIZE = 128
STEP_SIZE = 64
BATCH_SIZE = 256
EPOCHS_PRETRAIN = 80
LEARNING_RATE = 3e-4
TEMPERATURE = 0.05
EMBEDDING_DIM = 128
ANOMALY_RATIO_TEST = 0.20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = Path('data')
OUTPUT_DIR = Path('results_contrastive')
OUTPUT_DIR.mkdir(exist_ok=True)

ANOMALY_COLORS = {
    "normal":      "#2ECC71",
    "spike":       "#E74C3C",
    "plateau":     "#3498DB",
    "drift":       "#F39C12",
    "variance":    "#9B59B6",
    "dropout":     "#1ABC9C",
    "shape_shift": "#E67E22",
}

ANOMALY_TYPES = ["spike", "plateau", "drift", "variance", "dropout", "shape_shift"]
LABEL_MAP = {name: i + 1 for i, name in enumerate(ANOMALY_TYPES)}
LABEL_MAP["normal"] = 0
LABEL_NAMES = ["normal"] + ANOMALY_TYPES

PROFILES = {
    "profil_1_stable": ["I23", "I61"],
    "profil_2_bimodal": ["I42", "I43"],
    "profil_3_multi": ["52", "I11", "I113", "I32"],
}

print(f"Device: {DEVICE}")
print(f"Contrastive 1D config: temp={TEMPERATURE}, embed_dim={EMBEDDING_DIM}, "
      f"epochs={EPOCHS_PRETRAIN}, window={WINDOW_SIZE}")
print("Mode: Signal 1D direct (pas de CWT)")

# =====================================================================
# 1. Chargement
# =====================================================================
print("\n" + "=" * 70)
print("1. CHARGEMENT DES DONNÉES")
print("=" * 70)


def load_dataset(file_path):
    df = pd.read_csv(file_path, skiprows=3)
    df['_value'] = pd.to_numeric(df['_value'], errors='coerce')
    values = df['_value'].dropna().values.astype(np.float32)
    name = df['_measurement'].iloc[0] if '_measurement' in df.columns else file_path.stem
    return values, name


datasets = {}
for f in sorted(DATA_DIR.glob('*.csv')):
    values, name = load_dataset(f)
    datasets[f.stem] = {'values': values, 'name': name}
    print(f"  {f.stem:>5s} ({name:>4s}): {len(values):>6d} pts, "
          f"mean={values.mean():.3f}, std={values.std():.3f}")


# =====================================================================
# 2. Injection d'anomalies
# =====================================================================
def inject_spike(w, std):
    w = w.copy()
    for _ in range(np.random.randint(1, 4)):
        pos = np.random.randint(0, len(w))
        w[pos] += np.random.choice([-1, 1]) * np.random.uniform(5, 10) * std
    return w

def inject_plateau(w, mean):
    w = w.copy()
    length = np.random.randint(int(0.3 * len(w)), int(0.7 * len(w)))
    start = np.random.randint(0, len(w) - length)
    w[start:start + length] = mean + np.random.uniform(-0.5, 0.5) * mean
    return w

def inject_drift(w, std):
    w = w.copy()
    mag = np.random.uniform(2, 5) * std * np.random.choice([-1, 1])
    w += np.linspace(0, mag, len(w))
    return w

def inject_variance(w, std):
    w = w.copy()
    length = np.random.randint(int(0.3 * len(w)), int(0.7 * len(w)))
    start = np.random.randint(0, len(w) - length)
    w[start:start+length] += np.random.normal(
        0, np.random.uniform(3, 6) * std, length
    ).astype(np.float32)
    return w

def inject_dropout(w):
    w = w.copy()
    length = np.random.randint(int(0.2 * len(w)), int(0.5 * len(w)))
    start = np.random.randint(0, len(w) - length)
    w[start:start + length] = 0.0
    return w

def inject_shape_shift(w, std):
    w = w.copy()
    length = np.random.randint(int(0.4 * len(w)), int(0.8 * len(w)))
    start = np.random.randint(0, len(w) - length)
    freq = np.random.uniform(2, 8)
    amp = np.random.uniform(2, 4) * std
    t = np.linspace(0, 2 * np.pi * freq, length)
    w[start:start + length] = w[start:start + length].mean() + amp * np.sin(t)
    return w

def inject_anomaly(window, anomaly_type, mean, std):
    funcs = {
        "spike":       lambda w: inject_spike(w, std),
        "plateau":     lambda w: inject_plateau(w, mean),
        "drift":       lambda w: inject_drift(w, std),
        "variance":    lambda w: inject_variance(w, std),
        "dropout":     lambda w: inject_dropout(w),
        "shape_shift": lambda w: inject_shape_shift(w, std),
    }
    return funcs[anomaly_type](window)


# =====================================================================
# 3. Augmentations 1D pour paires contrastives
# =====================================================================
def aug_jitter(w, sigma=0.08):
    return w + np.random.normal(0, sigma, len(w)).astype(np.float32)

def aug_scaling(w):
    return w * np.random.normal(1.0, 0.1)

def aug_time_shift(w):
    shift = np.random.randint(-15, 16)
    return np.roll(w, shift)

def aug_crop_resize(w):
    n = len(w)
    crop_len = int(n * np.random.uniform(0.75, 1.0))
    start = np.random.randint(0, n - crop_len + 1)
    cropped = w[start:start + crop_len]
    return np.interp(
        np.linspace(0, len(cropped) - 1, n), np.arange(len(cropped)), cropped
    ).astype(np.float32)

def aug_permutation(w, n_seg=5):
    segs = np.array_split(w, n_seg)
    np.random.shuffle(segs)
    return np.concatenate(segs).astype(np.float32)

def aug_magnitude_warp(w):
    """Déformation non-linéaire de l'amplitude."""
    n = len(w)
    n_knots = 4
    knot_pos = np.sort(np.random.choice(n, n_knots, replace=False))
    knot_vals = np.random.normal(1.0, 0.15, n_knots)
    warp = np.interp(np.arange(n), knot_pos, knot_vals).astype(np.float32)
    return w * warp

def aug_time_warp(w):
    """Déformation temporelle non-linéaire."""
    n = len(w)
    n_knots = 4
    orig = np.linspace(0, n - 1, n_knots + 2)
    warped = orig.copy()
    warped[1:-1] += np.random.normal(0, n * 0.05, n_knots)
    warped = np.sort(np.clip(warped, 0, n - 1))
    new_time = np.interp(np.linspace(0, n - 1, n), warped, orig)
    return np.interp(new_time, np.arange(n), w).astype(np.float32)

ALL_AUGMENTATIONS = [
    aug_jitter, aug_scaling, aug_time_shift, aug_crop_resize,
    aug_permutation, aug_magnitude_warp, aug_time_warp,
]

def random_augmentation(w):
    """Applique 2-3 augmentations aléatoires."""
    w = w.copy()
    k = np.random.randint(2, 4)
    chosen = np.random.choice(len(ALL_AUGMENTATIONS), size=k, replace=False)
    for idx in chosen:
        w = ALL_AUGMENTATIONS[idx](w)
    return w


# =====================================================================
# 4. Dataset contrastif 1D
# =====================================================================
class Contrastive1DDataset(Dataset):
    """Génère des paires (view1, view2) de signaux 1D augmentés."""
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        v1 = torch.FloatTensor(random_augmentation(w)).unsqueeze(0)  # (1, T)
        v2 = torch.FloatTensor(random_augmentation(w)).unsqueeze(0)  # (1, T)
        return v1, v2


# =====================================================================
# 5. Modèle : Encoder Conv1D + Projection Head
# =====================================================================
class Contrastive1DEncoder(nn.Module):
    """
    Encoder 1D pour signaux temporels bruts.
    Input  : (batch, 1, 128) — signal 1D
    Output : embedding normalisé (batch, embed_dim)
    """
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            # (1, 128) → (64, 64)
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # (64, 64) → (128, 32)
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # (128, 32) → (256, 16)
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # (256, 16) → (512, 8)
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            # (512, 8) → (512, 1)
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),  # (512,)
        )

        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        """Projection normalisée (pour NT-Xent loss)."""
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=1)

    def get_embedding(self, x):
        """Embedding de l'encoder (pour la détection)."""
        h = self.encoder(x)
        return F.normalize(h, dim=1)


# =====================================================================
# 6. NT-Xent Loss
# =====================================================================
class NTXentLoss(nn.Module):
    def __init__(self, temperature=TEMPERATURE):
        super().__init__()
        self.temp = temperature

    def forward(self, z1, z2):
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temp
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)

        pos_log_probs = torch.cat([
            torch.diag(sim - torch.logsumexp(sim, dim=1, keepdim=True).squeeze(), N),
            torch.diag(sim - torch.logsumexp(sim, dim=1, keepdim=True).squeeze(), -N),
        ])
        return -pos_log_probs.mean()


# =====================================================================
# 7. Entraînement contrastif
# =====================================================================
def train_contrastive(model, train_loader, epochs=EPOCHS_PRETRAIN, lr=LEARNING_RATE):
    criterion = NTXentLoss(temperature=TEMPERATURE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'loss': []}
    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for v1, v2 in train_loader:
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
            z1, z2 = model(v1), model(v2)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        epoch_loss = total_loss / n_batches
        scheduler.step()
        history['loss'].append(epoch_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3d}/{epochs} | "
                  f"Loss: {epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    return history


# =====================================================================
# 8. Scoring
# =====================================================================
def compute_embeddings(model, windows, batch_size=512):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.FloatTensor(windows[i:i+batch_size]).unsqueeze(1).to(DEVICE)
            emb = model.get_embedding(batch)
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def anomaly_scores(embeddings, centroid):
    return np.sqrt(((embeddings - centroid) ** 2).sum(axis=1))


def find_optimal_threshold(scores, labels):
    """Maximise F1-score sur 2000 candidats."""
    y_true = (labels > 0).astype(int)
    candidates = np.linspace(np.percentile(scores, 1), np.percentile(scores, 99.9), 2000)
    best_f1, best_t = 0, candidates[len(candidates) // 2]
    for t in candidates:
        yp = (scores > t).astype(int)
        if yp.sum() == 0 or yp.sum() == len(yp):
            continue
        f = f1_score(y_true, yp, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


# =====================================================================
# 9. Visualisations
# =====================================================================
def create_windows(signal, window_size, step_size):
    wins = []
    for i in range(0, len(signal) - window_size + 1, step_size):
        wins.append(signal[i:i + window_size])
    return np.array(wins)


def plot_loss(history, pname):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history['loss'], color='#E74C3C', linewidth=2)
    ax.set_title(f'{pname} - Contrastive Loss (NT-Xent) — 1D', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{pname}_1d_contrastive_loss.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_embedding_pca(embeddings, labels, pname):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    e2d = pca.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(12, 8))
    for atype in LABEL_NAMES:
        lid = LABEL_MAP[atype]
        mask = labels == lid
        if mask.sum() == 0:
            continue
        ax.scatter(e2d[mask, 0], e2d[mask, 1], c=ANOMALY_COLORS[atype],
                   label=f'{atype} ({mask.sum()})', alpha=0.5, s=15, edgecolors='none')
    ax.set_title(f'{pname} - Espace Latent PCA 2D (1D Contrastive)', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{pname}_1d_embedding_pca.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_score_distribution(s_normal, s_anomaly, anom_labels, threshold, pname):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.hist(s_normal, bins=100, alpha=0.6, color=ANOMALY_COLORS["normal"],
            label=f'Normal (n={len(s_normal)})', density=True)
    for atype in ANOMALY_TYPES:
        mask = anom_labels == LABEL_MAP[atype]
        if mask.sum() > 0:
            ax.hist(s_anomaly[mask], bins=50, alpha=0.4, color=ANOMALY_COLORS[atype],
                    label=f'{atype} (n={mask.sum()})', density=True)
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
               label=f'Seuil optimal ({threshold:.4f})')
    ax.set_title(f'{pname} - Distribution des Scores (1D Contrastive)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Distance au centroïde')
    ax.set_ylabel('Densité')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{pname}_1d_score_dist.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_score_boxplot(scores, labels, pname):
    fig, ax = plt.subplots(figsize=(12, 6))
    data, ticks, cols = [], [], []
    for atype in LABEL_NAMES:
        mask = labels == LABEL_MAP[atype]
        if mask.sum() > 0:
            data.append(scores[mask])
            ticks.append(atype)
            cols.append(ANOMALY_COLORS[atype])
    bp = ax.boxplot(data, labels=ticks, patch_artist=True, showfliers=False)
    for patch, c in zip(bp['boxes'], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_title(f'{pname} - Score par Type (1D Contrastive)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Distance')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{pname}_1d_score_box.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_pr(y_true_bin, scores, pname):
    fpr, tpr, _ = roc_curve(y_true_bin, scores)
    rauc = auc(fpr, tpr)
    prec_c, rec_c, _ = precision_recall_curve(y_true_bin, scores)
    ap = average_precision_score(y_true_bin, scores)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(fpr, tpr, color='#E74C3C', linewidth=2, label=f'AUC={rauc:.4f}')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_title(f'{pname} - ROC (1D Contrastive)', fontweight='bold')
    ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(rec_c, prec_c, color='#3498DB', linewidth=2, label=f'AP={ap:.4f}')
    ax2.set_title(f'{pname} - Precision-Recall', fontweight='bold')
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{pname}_1d_roc_pr.png', dpi=150, bbox_inches='tight')
    plt.close()
    return rauc, ap


def plot_cm_binary(yt, yp, pname):
    cm = confusion_matrix(yt, yp)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomalie'],
                yticklabels=['Normal', 'Anomalie'], ax=ax)
    ax.set_title(f'{pname} - Confusion (1D Contrastive)', fontweight='bold')
    ax.set_xlabel('Prédit'); ax.set_ylabel('Réel')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{pname}_1d_cm_binary.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_cm_multiclass(yt, yp, pname):
    cm = confusion_matrix(yt, yp)
    present = sorted(set(yt) | set(yp))
    names = [LABEL_NAMES[i] for i in present]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names, ax=ax)
    ax.set_title(f'{pname} - Confusion Multi-Classes (1D Contrastive)', fontweight='bold')
    ax.set_xlabel('Prédit'); ax.set_ylabel('Réel')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{pname}_1d_cm_multi.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_per_type(scores, labels, threshold, pname):
    results = {}
    for atype in ANOMALY_TYPES:
        lid = LABEL_MAP[atype]
        mask = (labels == 0) | (labels == lid)
        if (labels[mask] == lid).sum() == 0:
            continue
        yt = (labels[mask] > 0).astype(int)
        yp = (scores[mask] > threshold).astype(int)
        results[atype] = {
            'precision': precision_score(yt, yp, zero_division=0),
            'recall':    recall_score(yt, yp, zero_division=0),
            'f1':        f1_score(yt, yp, zero_division=0),
        }
    if not results:
        return
    types_l = list(results.keys())
    x = np.arange(len(types_l))
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - w, [results[t]['precision'] for t in types_l], w,
                label='Precision', color='#3498DB', alpha=0.8)
    b2 = ax.bar(x,     [results[t]['recall']    for t in types_l], w,
                label='Recall',    color='#E74C3C', alpha=0.8)
    b3 = ax.bar(x + w, [results[t]['f1']        for t in types_l], w,
                label='F1-Score',  color='#2ECC71', alpha=0.8)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    ax.set_title(f'{pname} - Métriques par Type (1D Contrastive)', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(types_l, rotation=45, ha='right')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{pname}_1d_metrics_type.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Détection par type:")
    print(f"  {'Type':>12s} | {'Prec':>6s} | {'Rec':>6s} | {'F1':>6s}")
    print(f"  {'-'*36}")
    for t in types_l:
        print(f"  {t:>12s} | {results[t]['precision']:.4f} | "
              f"{results[t]['recall']:.4f} | {results[t]['f1']:.4f}")


def plot_threshold_search(scores, labels, best_t, pname):
    yt = (labels > 0).astype(int)
    cands = np.linspace(np.percentile(scores, 1), np.percentile(scores, 99.9), 500)
    f1s, ps, rs = [], [], []
    for t in cands:
        yp = (scores > t).astype(int)
        if yp.sum() == 0 or yp.sum() == len(yp):
            f1s.append(0); ps.append(0); rs.append(0)
            continue
        f1s.append(f1_score(yt, yp, zero_division=0))
        ps.append(precision_score(yt, yp, zero_division=0))
        rs.append(recall_score(yt, yp, zero_division=0))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cands, f1s, color='#2ECC71', linewidth=2, label='F1')
    ax.plot(cands, ps, color='#3498DB', linewidth=1.5, alpha=0.7, label='Precision')
    ax.plot(cands, rs, color='#E74C3C', linewidth=1.5, alpha=0.7, label='Recall')
    ax.axvline(x=best_t, color='black', linestyle='--', linewidth=1.5,
               label=f'Optimal = {best_t:.4f}')
    ax.set_title(f'{pname} - Recherche Seuil Optimal', fontweight='bold')
    ax.set_xlabel('Seuil'); ax.set_ylabel('Score')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{pname}_1d_threshold.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_signal_examples(windows, labels, scores, threshold, pname):
    """Montre un exemple de signal par type avec son score."""
    fig, axes = plt.subplots(len(LABEL_NAMES), 1, figsize=(16, 2.8 * len(LABEL_NAMES)))
    fig.suptitle(f'{pname} - Signaux 1D avec Score d\'Anomalie', fontsize=14, fontweight='bold')
    for i, atype in enumerate(LABEL_NAMES):
        lid = LABEL_MAP[atype]
        idxs = np.where(labels == lid)[0]
        if len(idxs) == 0:
            continue
        idx = idxs[0]
        ax = axes[i]
        color = ANOMALY_COLORS[atype]
        ax.plot(windows[idx], color=color, linewidth=1.2)
        s = scores[idx]
        status = "ANOMALIE" if s > threshold else "NORMAL"
        ax.set_title(f'{atype} | Score={s:.4f} | Seuil={threshold:.4f} | → {status}',
                     fontsize=10, color=color, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_ylabel('Amp')
    axes[-1].set_xlabel('Temps')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{pname}_1d_signal_examples.png', dpi=150, bbox_inches='tight')
    plt.close()


# =====================================================================
# 10. Pipeline principal
# =====================================================================
print("\n" + "=" * 70)
print("10. PIPELINE CONTRASTIF 1D PAR PROFIL")
print("=" * 70)

all_results = {}

for pname, sensor_keys in PROFILES.items():
    print(f"\n{'#' * 70}")
    print(f"# {pname} — Capteurs : {sensor_keys}")
    print(f"{'#' * 70}")

    # --- Fenêtres normales ---
    print("\n  [1/7] Fenêtres normales...")
    all_w = []
    for key in sensor_keys:
        if key not in datasets:
            continue
        vals = datasets[key]['values']
        sc = StandardScaler()
        vals_s = sc.fit_transform(vals.reshape(-1, 1)).flatten().astype(np.float32)
        wins = create_windows(vals_s, WINDOW_SIZE, STEP_SIZE)
        all_w.append(wins)
        print(f"    {key}: {len(wins)} fenêtres")

    all_w = np.concatenate(all_w, axis=0)
    np.random.shuffle(all_w)
    n = len(all_w)

    n_train = int(n * 0.60)
    n_val   = int(n * 0.15)
    n_test_n = n - n_train - n_val

    train_w = all_w[:n_train]
    val_w   = all_w[n_train:n_train + n_val]
    test_n  = all_w[n_train + n_val:]
    print(f"    Total: {n} | Train: {n_train} | Val: {n_val} | Test normal: {n_test_n}")

    # --- Entraînement contrastif 1D ---
    print(f"\n  [2/7] Entraînement contrastif 1D ({EPOCHS_PRETRAIN} epochs)...")
    ds = Contrastive1DDataset(train_w)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    model = Contrastive1DEncoder(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Paramètres: {n_params:,}")

    history = train_contrastive(model, loader)
    plot_loss(history, pname)

    # --- Centroïde ---
    print("\n  [3/7] Centroïde normal...")
    train_emb = compute_embeddings(model, train_w)
    centroid = train_emb.mean(axis=0)
    print(f"    Centroïde norm: {np.linalg.norm(centroid):.4f}")

    # --- Validation + seuil optimal ---
    print("\n  [4/7] Validation pour seuil optimal...")
    g_mean, g_std = all_w.mean(), all_w.std()

    n_va = int(len(val_w) * ANOMALY_RATIO_TEST / (1 - ANOMALY_RATIO_TEST))
    va_wins, va_labs = [], []
    for _ in range(n_va):
        bi = np.random.randint(0, len(val_w))
        at = np.random.choice(ANOMALY_TYPES)
        va_wins.append(inject_anomaly(val_w[bi].copy(), at, g_mean, g_std))
        va_labs.append(LABEL_MAP[at])

    val_all = np.concatenate([val_w, np.array(va_wins)], axis=0)
    val_labs = np.concatenate([np.zeros(len(val_w), dtype=np.int64), np.array(va_labs)])
    shuf = np.random.permutation(len(val_all))
    val_all, val_labs = val_all[shuf], val_labs[shuf]

    val_emb = compute_embeddings(model, val_all)
    val_sc = anomaly_scores(val_emb, centroid)
    opt_t, val_f1 = find_optimal_threshold(val_sc, val_labs)
    print(f"    Seuil optimal: {opt_t:.4f} (F1 val = {val_f1:.4f})")
    plot_threshold_search(val_sc, val_labs, opt_t, pname)

    # --- Test ---
    print("\n  [5/7] Test avec anomalies...")
    n_ta = int(n_test_n * ANOMALY_RATIO_TEST / (1 - ANOMALY_RATIO_TEST))
    ta_wins, ta_labs = [], []
    acounts = {t: 0 for t in ANOMALY_TYPES}
    for _ in range(n_ta):
        bi = np.random.randint(0, n_test_n)
        at = np.random.choice(ANOMALY_TYPES)
        ta_wins.append(inject_anomaly(test_n[bi].copy(), at, g_mean, g_std))
        ta_labs.append(LABEL_MAP[at])
        acounts[at] += 1

    test_all = np.concatenate([test_n, np.array(ta_wins)], axis=0)
    test_labs = np.concatenate([np.zeros(n_test_n, dtype=np.int64), np.array(ta_labs)])
    shuf = np.random.permutation(len(test_all))
    test_all, test_labs = test_all[shuf], test_labs[shuf]

    print(f"    Test: {n_test_n} normal + {n_ta} anomalies = {len(test_all)}")
    for t, c in acounts.items():
        print(f"      {t:>12s}: {c}")

    # --- Évaluation ---
    print("\n  [6/7] Évaluation...")
    test_emb = compute_embeddings(model, test_all)
    test_sc = anomaly_scores(test_emb, centroid)

    yt_bin = (test_labs > 0).astype(int)
    yp_bin = (test_sc > opt_t).astype(int)

    acc  = accuracy_score(yt_bin, yp_bin)
    prec = precision_score(yt_bin, yp_bin, zero_division=0)
    rec  = recall_score(yt_bin, yp_bin, zero_division=0)
    f1   = f1_score(yt_bin, yp_bin, zero_division=0)

    print(f"\n  {'=' * 55}")
    print(f"  RÉSULTATS — {pname} (Contrastive 1D)")
    print(f"  {'=' * 55}")
    print(f"  Seuil optimal:  {opt_t:.4f}")
    print(f"  Accuracy :      {acc:.4f}")
    print(f"  Precision:      {prec:.4f}")
    print(f"  Recall   :      {rec:.4f}")
    print(f"  F1-Score :      {f1:.4f}")
    print(f"\n{classification_report(yt_bin, yp_bin, target_names=['Normal','Anomalie'], zero_division=0)}")

    # --- Visualisations ---
    print("  [7/7] Visualisations...")
    plot_embedding_pca(test_emb, test_labs, pname)
    plot_score_distribution(test_sc[test_labs == 0], test_sc[test_labs > 0],
                            test_labs[test_labs > 0], opt_t, pname)
    plot_score_boxplot(test_sc, test_labs, pname)
    rauc, ap = plot_roc_pr(yt_bin, test_sc, pname)
    print(f"    ROC AUC: {rauc:.4f}, AP: {ap:.4f}")
    plot_cm_binary(yt_bin, yp_bin, pname)
    yp_multi = test_labs.copy()
    yp_multi[yp_bin == 0] = 0
    plot_cm_multiclass(test_labs, yp_multi, pname)
    plot_metrics_per_type(test_sc, test_labs, opt_t, pname)
    plot_signal_examples(test_all, test_labs, test_sc, opt_t, pname)

    torch.save(model.state_dict(), OUTPUT_DIR / f'{pname}_contrastive1d.pth')
    print(f"  Modèle: {OUTPUT_DIR / f'{pname}_contrastive1d.pth'}")

    all_results[pname] = {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'roc_auc': rauc, 'avg_precision': ap, 'threshold': opt_t,
        'history': history,
    }

# =====================================================================
# 11. Résumé global
# =====================================================================
print("\n" + "=" * 70)
print("11. RÉSUMÉ GLOBAL — CONTRASTIVE 1D")
print("=" * 70)

rows = []
for p, r in all_results.items():
    rows.append({
        'Profil': p, 'Acc': f"{r['accuracy']:.4f}", 'Prec': f"{r['precision']:.4f}",
        'Rec': f"{r['recall']:.4f}", 'F1': f"{r['f1']:.4f}",
        'AUC': f"{r['roc_auc']:.4f}", 'AP': f"{r['avg_precision']:.4f}",
        'Seuil': f"{r['threshold']:.4f}",
    })
print(pd.DataFrame(rows).to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6))
pl = list(all_results.keys())
mets = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
x = np.arange(len(pl))
wd = 0.15
for i, m in enumerate(mets):
    ax.bar(x + i * wd, [all_results[p][m] for p in pl], wd, label=m.upper())
ax.set_xticks(x + 2 * wd)
ax.set_xticklabels(pl, rotation=15)
ax.set_title('Contrastive 1D — Comparaison par Profil', fontweight='bold')
ax.set_ylabel('Score')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'global_contrastive1d_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  Contrastive 1D terminé! Résultats: {OUTPUT_DIR}")
