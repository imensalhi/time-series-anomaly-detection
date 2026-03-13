"""
=============================================================================
CWT + CNN Autoencoder — Détection d'Anomalies Non-Supervisée
=============================================================================
Principe :
- L'autoencoder est entraîné UNIQUEMENT sur des signaux NORMAUX (sans anomalies).
- En test, on injecte les 6 types d'anomalies synthétiques.
- L'erreur de reconstruction (MSE) sur une fenêtre anormale sera plus élevée
  que sur une fenêtre normale → un seuil sépare normal / anomalie.

Stratégie multi-profils (comme pour la classification) :
    Profil 1 (Stable/bruit)     : I23, I61
    Profil 2 (ON/OFF bimodal)   : I42, I43
    Profil 3 (Multi-niveaux)    : I52, I11, I113, I32

Pipeline :
1. Charger + normaliser (colonne _value)
2. Fenêtrage glissant
3. Train : fenêtres normales uniquement
4. Test  : fenêtres normales + fenêtres avec anomalies injectées
5. CWT → scalogrammes
6. Autoencoder CNN 2D : encode le scalogramme → latent → décode
7. Reconstruire et calculer l'erreur de reconstruction
8. Fixer le seuil (percentile 95 des erreurs normales du train)
9. Classifier : erreur > seuil → anomalie
10. Métriques, matrice de confusion, graphes
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
SCALES = np.arange(1, 65)   # 64 échelles → scalogramme 64 x 128
WAVELET = 'morl'
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
ANOMALY_RATIO_TEST = 0.20   # 20% d'anomalies dans le test
THRESHOLD_PERCENTILE = 95   # Seuil = percentile 95 des erreurs normales
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = Path('data')
OUTPUT_DIR = Path('results_autoencoder')
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
print(f"Window: {WINDOW_SIZE}, Scales: {len(SCALES)}, Wavelet: {WAVELET}")
print(f"Threshold percentile: {THRESHOLD_PERCENTILE}")

# =====================================================================
# 1. Chargement des données
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
    print(f"  {f.stem:>5s} ({name:>4s}): {len(values):>6d} points, "
          f"mean={values.mean():.3f}, std={values.std():.3f}")

# =====================================================================
# 2. Injection d'anomalies (mêmes fonctions que classification)
# =====================================================================
print("\n" + "=" * 70)
print("2. FONCTIONS D'INJECTION D'ANOMALIES")
print("=" * 70)


def inject_spike(window, std):
    w = window.copy()
    n_spikes = np.random.randint(1, 4)
    for _ in range(n_spikes):
        pos = np.random.randint(0, len(w))
        direction = np.random.choice([-1, 1])
        magnitude = np.random.uniform(5, 10) * std
        w[pos] += direction * magnitude
    return w


def inject_plateau(window, mean):
    w = window.copy()
    length = np.random.randint(int(0.3 * len(w)), int(0.7 * len(w)))
    start = np.random.randint(0, len(w) - length)
    plateau_val = mean + np.random.uniform(-0.5, 0.5) * mean
    w[start:start + length] = plateau_val
    return w


def inject_drift(window, std):
    w = window.copy()
    drift_magnitude = np.random.uniform(2, 5) * std
    direction = np.random.choice([-1, 1])
    drift = np.linspace(0, direction * drift_magnitude, len(w))
    w += drift
    return w


def inject_variance(window, std):
    w = window.copy()
    length = np.random.randint(int(0.3 * len(w)), int(0.7 * len(w)))
    start = np.random.randint(0, len(w) - length)
    factor = np.random.uniform(3, 6)
    noise = np.random.normal(0, factor * std, length).astype(np.float32)
    w[start:start + length] += noise
    return w


def inject_dropout(window):
    w = window.copy()
    length = np.random.randint(int(0.2 * len(w)), int(0.5 * len(w)))
    start = np.random.randint(0, len(w) - length)
    w[start:start + length] = 0.0
    return w


def inject_shape_shift(window, std):
    w = window.copy()
    length = np.random.randint(int(0.4 * len(w)), int(0.8 * len(w)))
    start = np.random.randint(0, len(w) - length)
    freq = np.random.uniform(2, 8)
    amplitude = np.random.uniform(2, 4) * std
    t = np.linspace(0, 2 * np.pi * freq, length)
    w[start:start + length] = w[start:start + length].mean() + amplitude * np.sin(t)
    return w


def inject_anomaly(window, anomaly_type, mean, std):
    if anomaly_type == "spike":
        return inject_spike(window, std)
    elif anomaly_type == "plateau":
        return inject_plateau(window, mean)
    elif anomaly_type == "drift":
        return inject_drift(window, std)
    elif anomaly_type == "variance":
        return inject_variance(window, std)
    elif anomaly_type == "dropout":
        return inject_dropout(window)
    elif anomaly_type == "shape_shift":
        return inject_shape_shift(window, std)
    return window


print("  6 types d'anomalies définis: spike, plateau, drift, variance, dropout, shape_shift")

# =====================================================================
# 3. Fenêtrage
# =====================================================================
def create_windows(signal, window_size, step_size):
    windows = []
    for i in range(0, len(signal) - window_size + 1, step_size):
        windows.append(signal[i:i + window_size])
    return np.array(windows)


# =====================================================================
# 4. CWT
# =====================================================================
def compute_cwt_batch(windows, scales=SCALES, wavelet=WAVELET):
    scalograms = []
    for i, w in enumerate(windows):
        coeffs, _ = pywt.cwt(w, scales, wavelet)
        scalograms.append(np.abs(coeffs))
        if (i + 1) % 500 == 0:
            print(f"    CWT: {i + 1}/{len(windows)} fenêtres...")
    return np.array(scalograms, dtype=np.float32)


# =====================================================================
# 5. Modèle CNN Autoencoder
# =====================================================================
class CWT_CNN_Autoencoder(nn.Module):
    """
    Autoencoder CNN 2D pour scalogrammes CWT.
    Encoder : compresse le scalogramme (64x128) en représentation latente
    Decoder : reconstruit le scalogramme original
    L'erreur de reconstruction sert de score d'anomalie.
    """
    def __init__(self, n_scales=64, window_size=128):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # (1, 64, 128) → (32, 32, 64)
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # (32, 32, 64) → (64, 16, 32)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (64, 16, 32) → (128, 8, 16)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # (128, 8, 16) → (256, 4, 8)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Decoder (symétrique)
        self.decoder = nn.Sequential(
            # (256, 4, 8) → (128, 8, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # (128, 8, 16) → (64, 16, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (64, 16, 32) → (32, 32, 64)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # (32, 32, 64) → (1, 64, 128)
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def get_reconstruction_error(self, x):
        """Calcule l'erreur MSE par échantillon."""
        with torch.no_grad():
            reconstructed = self.forward(x)
            # MSE par échantillon (moyenné sur les dimensions spatiales)
            error = ((x - reconstructed) ** 2).mean(dim=(1, 2, 3))
        return error


# =====================================================================
# 6. Fonctions d'entraînement
# =====================================================================
def train_autoencoder(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    """Entraîne l'autoencoder uniquement sur des données normales."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        running_loss, total = 0.0, 0
        for (X_batch,) in train_loader:
            X_batch = X_batch.to(DEVICE)
            optimizer.zero_grad()
            reconstructed = model(X_batch)
            loss = criterion(reconstructed, X_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            total += X_batch.size(0)

        train_loss = running_loss / total

        # --- Validation ---
        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for (X_batch,) in val_loader:
                X_batch = X_batch.to(DEVICE)
                reconstructed = model(X_batch)
                loss = criterion(reconstructed, X_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_total += X_batch.size(0)

        val_loss = val_loss / val_total
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    return history


# =====================================================================
# 7. Visualisations spécifiques à l'autoencoder
# =====================================================================
def plot_training_history_ae(history, profile_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history['train_loss'], label='Train Loss (MSE)', color='#E74C3C', linewidth=2)
    ax.plot(history['val_loss'], label='Val Loss (MSE)', color='#3498DB', linewidth=2)
    ax.set_title(f'{profile_name} - Autoencoder Training Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{profile_name}_ae_training_loss.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_error_distribution(errors_normal, errors_anomaly, anomaly_type_labels,
                            threshold, profile_name):
    """Distribution des erreurs de reconstruction : normal vs chaque type d'anomalie."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Erreurs normales
    ax.hist(errors_normal, bins=100, alpha=0.6, color=ANOMALY_COLORS["normal"],
            label=f'Normal (n={len(errors_normal)})', density=True)

    # Erreurs par type d'anomalie
    for atype in ANOMALY_TYPES:
        mask = anomaly_type_labels == LABEL_MAP[atype]
        if mask.sum() > 0:
            ax.hist(errors_anomaly[mask], bins=50, alpha=0.4,
                    color=ANOMALY_COLORS[atype],
                    label=f'{atype} (n={mask.sum()})', density=True)

    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
               label=f'Seuil (P{THRESHOLD_PERCENTILE}={threshold:.4f})')
    ax.set_title(f'{profile_name} - Distribution des Erreurs de Reconstruction',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Erreur de Reconstruction (MSE)')
    ax.set_ylabel('Densité')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{profile_name}_error_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_error_by_type_boxplot(errors_test, labels_test, profile_name):
    """Boxplot des erreurs par type."""
    fig, ax = plt.subplots(figsize=(12, 6))
    data_per_type = []
    tick_labels = []
    colors = []

    for atype in LABEL_NAMES:
        label_id = LABEL_MAP[atype]
        mask = labels_test == label_id
        if mask.sum() > 0:
            data_per_type.append(errors_test[mask])
            tick_labels.append(atype)
            colors.append(ANOMALY_COLORS[atype])

    bp = ax.boxplot(data_per_type, labels=tick_labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_title(f'{profile_name} - Erreur de Reconstruction par Type',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Type')
    ax.set_ylabel('Erreur MSE')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{profile_name}_error_boxplot.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true_binary, errors, profile_name):
    """Courbe ROC pour la détection binaire (normal vs anomalie)."""
    fpr, tpr, _ = roc_curve(y_true_binary, errors)
    roc_auc = auc(fpr, tpr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ROC
    ax1.plot(fpr, tpr, color='#E74C3C', linewidth=2, label=f'AUC = {roc_auc:.4f}')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_title(f'{profile_name} - Courbe ROC', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Taux de Faux Positifs')
    ax1.set_ylabel('Taux de Vrais Positifs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Precision-Recall
    precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, errors)
    ap = average_precision_score(y_true_binary, errors)
    ax2.plot(recall_curve, precision_curve, color='#3498DB', linewidth=2,
             label=f'AP = {ap:.4f}')
    ax2.set_title(f'{profile_name} - Courbe Precision-Recall', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{profile_name}_roc_pr_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

    return roc_auc, ap


def plot_confusion_matrix_ae(y_true, y_pred, profile_name):
    """Matrice de confusion binaire (normal vs anomalie)."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomalie'],
                yticklabels=['Normal', 'Anomalie'], ax=ax)
    ax.set_title(f'{profile_name} - Matrice de Confusion (Autoencoder)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Prédit')
    ax.set_ylabel('Réel')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{profile_name}_confusion_matrix_binary.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix_multiclass(y_true, y_pred, profile_name):
    """Matrice de confusion multi-classes (normal + 6 types d'anomalies).
    La prédiction est basée sur le seuil d'erreur : si > seuil, on assigne
    le label 'anomalie générique', sinon normal."""
    cm = confusion_matrix(y_true, y_pred)
    labels_present = sorted(set(y_true) | set(y_pred))
    label_names_present = [LABEL_NAMES[i] for i in labels_present]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names_present,
                yticklabels=label_names_present, ax=ax)
    ax.set_title(f'{profile_name} - Matrice de Confusion Multi-Classes',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Prédit (basé sur reconstruction)')
    ax.set_ylabel('Réel')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{profile_name}_confusion_matrix_multiclass.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_reconstruction_examples(model, test_cwt, test_labels, profile_name):
    """Affiche des exemples de reconstruction normal vs anomalie."""
    model.eval()
    fig, axes = plt.subplots(min(len(LABEL_NAMES), 7), 3, figsize=(18, 3.5 * min(len(LABEL_NAMES), 7)))
    fig.suptitle(f'{profile_name} - Reconstruction: Original vs Reconstruit vs Erreur',
                 fontsize=14, fontweight='bold')

    for i, atype in enumerate(LABEL_NAMES):
        if i >= 7:
            break
        label_id = LABEL_MAP[atype]
        indices = np.where(test_labels == label_id)[0]
        if len(indices) == 0:
            continue

        idx = indices[0]
        original = test_cwt[idx:idx+1]
        original_tensor = torch.FloatTensor(original).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            reconstructed = model(original_tensor).cpu().numpy().squeeze()

        original_img = original.squeeze()
        error_img = np.abs(original_img - reconstructed)

        color = ANOMALY_COLORS[atype]

        ax1 = axes[i, 0]
        ax1.imshow(original_img, aspect='auto', cmap='jet')
        ax1.set_title(f'{atype} - Original', color=color, fontweight='bold', fontsize=10)
        ax1.set_ylabel('Scale')

        ax2 = axes[i, 1]
        ax2.imshow(reconstructed, aspect='auto', cmap='jet')
        ax2.set_title(f'{atype} - Reconstruit', fontsize=10)

        ax3 = axes[i, 2]
        ax3.imshow(error_img, aspect='auto', cmap='hot')
        mse = ((original_img - reconstructed) ** 2).mean()
        ax3.set_title(f'{atype} - Erreur (MSE={mse:.4f})', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{profile_name}_reconstruction_examples.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_metrics_per_anomaly_type(errors_test, labels_test, threshold, profile_name):
    """Barplot Precision/Recall/F1 par type d'anomalie (détection binaire par type)."""
    results = {}
    for atype in ANOMALY_TYPES:
        label_id = LABEL_MAP[atype]
        # Sous-ensemble : normal + ce type d'anomalie
        mask = (labels_test == 0) | (labels_test == label_id)
        sub_errors = errors_test[mask]
        sub_labels = labels_test[mask]
        # Binaire : 0 = normal, 1 = anomalie
        y_true_bin = (sub_labels > 0).astype(int)
        y_pred_bin = (sub_errors > threshold).astype(int)

        if y_true_bin.sum() > 0:
            p = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            r = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            f = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            results[atype] = {'precision': p, 'recall': r, 'f1': f}

    if not results:
        return

    types_list = list(results.keys())
    precision_vals = [results[t]['precision'] for t in types_list]
    recall_vals = [results[t]['recall'] for t in types_list]
    f1_vals = [results[t]['f1'] for t in types_list]

    x = np.arange(len(types_list))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, precision_vals, width, label='Precision', color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x, recall_vals, width, label='Recall', color='#E74C3C', alpha=0.8)
    bars3 = ax.bar(x + width, f1_vals, width, label='F1-Score', color='#2ECC71', alpha=0.8)

    ax.set_xlabel('Type d\'Anomalie')
    ax.set_ylabel('Score')
    ax.set_title(f'{profile_name} - Détection par Type (Autoencoder)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(types_list, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{profile_name}_metrics_per_type_ae.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    # Afficher les résultats
    print(f"\n  Détection par type d'anomalie:")
    print(f"  {'Type':>12s} | {'Precision':>9s} | {'Recall':>9s} | {'F1':>9s}")
    print(f"  {'-'*48}")
    for t in types_list:
        print(f"  {t:>12s} | {results[t]['precision']:>9.4f} | "
              f"{results[t]['recall']:>9.4f} | {results[t]['f1']:>9.4f}")


def plot_example_signals_with_errors(windows_test, errors_test, labels_test,
                                     threshold, profile_name):
    """Signal temporel avec erreur de reconstruction en overlay."""
    fig, axes = plt.subplots(len(LABEL_NAMES), 1, figsize=(16, 3 * len(LABEL_NAMES)))
    fig.suptitle(f'{profile_name} - Signaux Test avec Score d\'Anomalie',
                 fontsize=14, fontweight='bold')

    for i, atype in enumerate(LABEL_NAMES):
        label_id = LABEL_MAP[atype]
        indices = np.where(labels_test == label_id)[0]
        if len(indices) == 0:
            continue

        idx = indices[0]
        ax = axes[i]
        color = ANOMALY_COLORS[atype]
        ax.plot(windows_test[idx], color=color, linewidth=1.2, label=f'{atype}')
        err = errors_test[idx]
        status = "ANOMALIE" if err > threshold else "NORMAL"
        ax.set_title(f'{atype} | Erreur MSE = {err:.4f} | Seuil = {threshold:.4f} | → {status}',
                     fontsize=10, color=color, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_ylabel('Amplitude')

    axes[-1].set_xlabel('Temps')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{profile_name}_signals_with_errors.png',
                dpi=150, bbox_inches='tight')
    plt.show()


# =====================================================================
# 8. Pipeline principal — Entraînement par profil
# =====================================================================
print("\n" + "=" * 70)
print("8. PIPELINE AUTOENCODER PAR PROFIL")
print("=" * 70)

all_results = {}

for profile_name, sensor_keys in PROFILES.items():
    print(f"\n{'#' * 70}")
    print(f"# PROFIL : {profile_name} — Capteurs : {sensor_keys}")
    print(f"{'#' * 70}")

    # --- Préparer les fenêtres normales ---
    print("\n  [1/7] Préparation des fenêtres normales...")
    all_windows = []
    for key in sensor_keys:
        if key not in datasets:
            print(f"  [WARN] Capteur {key} non trouvé, ignoré.")
            continue
        values = datasets[key]['values']
        scaler = StandardScaler()
        values_scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten().astype(np.float32)
        windows = create_windows(values_scaled, WINDOW_SIZE, STEP_SIZE)
        all_windows.append(windows)
        print(f"    {key}: {len(windows)} fenêtres normales")

    all_windows = np.concatenate(all_windows, axis=0)
    np.random.shuffle(all_windows)

    n_total = len(all_windows)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.1)
    n_test_normal = n_total - n_train - n_val

    train_windows = all_windows[:n_train]
    val_windows = all_windows[n_train:n_train + n_val]
    test_normal_windows = all_windows[n_train + n_val:]

    print(f"    Total: {n_total} fenêtres")
    print(f"    Train (normal): {n_train}")
    print(f"    Val (normal)  : {n_val}")
    print(f"    Test (normal) : {n_test_normal}")

    # --- Préparer les fenêtres de test avec anomalies ---
    print("\n  [2/7] Injection d'anomalies dans le test...")
    n_anomalies = int(n_test_normal * ANOMALY_RATIO_TEST / (1 - ANOMALY_RATIO_TEST))
    global_mean = all_windows.mean()
    global_std = all_windows.std()

    anomaly_windows = []
    anomaly_labels = []
    anomaly_counts = {t: 0 for t in ANOMALY_TYPES}

    for _ in range(n_anomalies):
        # Prendre une fenêtre normale aléatoire et la corrompre
        base_idx = np.random.randint(0, n_test_normal)
        base_window = test_normal_windows[base_idx].copy()
        atype = np.random.choice(ANOMALY_TYPES)
        corrupted = inject_anomaly(base_window, atype, global_mean, global_std)
        anomaly_windows.append(corrupted)
        anomaly_labels.append(LABEL_MAP[atype])
        anomaly_counts[atype] += 1

    anomaly_windows = np.array(anomaly_windows)
    anomaly_labels = np.array(anomaly_labels)

    # Combiner test normal + anomalies
    test_windows = np.concatenate([test_normal_windows, anomaly_windows], axis=0)
    test_labels = np.concatenate([
        np.zeros(n_test_normal, dtype=np.int64),
        anomaly_labels
    ])

    # Mélanger
    shuffle_idx = np.random.permutation(len(test_windows))
    test_windows = test_windows[shuffle_idx]
    test_labels = test_labels[shuffle_idx]

    print(f"    Test total    : {len(test_windows)} ({n_test_normal} normal + {n_anomalies} anomalies)")
    for t, c in anomaly_counts.items():
        print(f"      {t:>12s}: {c}")

    # --- CWT ---
    print("\n  [3/7] Transformation CWT...")
    print("    Train:")
    train_cwt = compute_cwt_batch(train_windows)
    print("    Val:")
    val_cwt = compute_cwt_batch(val_windows)
    print("    Test:")
    test_cwt = compute_cwt_batch(test_windows)

    # Normaliser
    cwt_max = train_cwt.max()
    if cwt_max > 0:
        train_cwt /= cwt_max
        val_cwt /= cwt_max
        test_cwt /= cwt_max

    # Tenseurs
    train_tensor = torch.FloatTensor(train_cwt).unsqueeze(1)
    val_tensor = torch.FloatTensor(val_cwt).unsqueeze(1)
    test_tensor = torch.FloatTensor(test_cwt).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=BATCH_SIZE, shuffle=False)

    print(f"    Shapes: Train={train_tensor.shape}, Val={val_tensor.shape}, Test={test_tensor.shape}")

    # --- Entraîner ---
    print(f"\n  [4/7] Entraînement de l'autoencoder ({EPOCHS} epochs)...")
    model = CWT_CNN_Autoencoder(n_scales=len(SCALES), window_size=WINDOW_SIZE).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Architecture: {total_params:,} paramètres")

    history = train_autoencoder(model, train_loader, val_loader)
    plot_training_history_ae(history, profile_name)

    # --- Calculer les erreurs de reconstruction ---
    print("\n  [5/7] Calcul des erreurs de reconstruction...")
    model.eval()

    # Erreurs sur le train (normal) → pour fixer le seuil
    train_errors = []
    with torch.no_grad():
        for start in range(0, len(train_tensor), BATCH_SIZE):
            batch = train_tensor[start:start + BATCH_SIZE].to(DEVICE)
            errors = model.get_reconstruction_error(batch)
            train_errors.extend(errors.cpu().numpy())
    train_errors = np.array(train_errors)

    # Erreurs sur le test
    test_errors = []
    with torch.no_grad():
        for start in range(0, len(test_tensor), BATCH_SIZE):
            batch = test_tensor[start:start + BATCH_SIZE].to(DEVICE)
            errors = model.get_reconstruction_error(batch)
            test_errors.extend(errors.cpu().numpy())
    test_errors = np.array(test_errors)

    # --- Fixer le seuil ---
    threshold = np.percentile(train_errors, THRESHOLD_PERCENTILE)
    print(f"    Seuil (P{THRESHOLD_PERCENTILE} du train): {threshold:.6f}")
    print(f"    Erreur train: mean={train_errors.mean():.6f}, "
          f"std={train_errors.std():.6f}, max={train_errors.max():.6f}")
    print(f"    Erreur test (normal): mean={test_errors[test_labels==0].mean():.6f}")
    for atype in ANOMALY_TYPES:
        mask = test_labels == LABEL_MAP[atype]
        if mask.sum() > 0:
            print(f"    Erreur test ({atype:>12s}): mean={test_errors[mask].mean():.6f}")

    # --- Prédictions ---
    print("\n  [6/7] Prédictions et évaluation...")

    # Binaire : 0 = normal, 1 = anomalie
    y_true_binary = (test_labels > 0).astype(int)
    y_pred_binary = (test_errors > threshold).astype(int)

    acc = accuracy_score(y_true_binary, y_pred_binary)
    prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    print(f"\n  {'=' * 50}")
    print(f"  RÉSULTATS BINAIRES — {profile_name}")
    print(f"  {'=' * 50}")
    print(f"  Accuracy :  {acc:.4f}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall   :  {rec:.4f}")
    print(f"  F1-Score :  {f1:.4f}")
    print(f"\n  Classification Report (binaire):")
    print(classification_report(y_true_binary, y_pred_binary,
                                target_names=['Normal', 'Anomalie'], zero_division=0))

    # --- Visualisations ---
    print("  [7/7] Visualisations...")

    # Distribution des erreurs
    plot_error_distribution(
        test_errors[test_labels == 0],
        test_errors[test_labels > 0],
        test_labels[test_labels > 0],
        threshold, profile_name
    )

    # Boxplot par type
    plot_error_by_type_boxplot(test_errors, test_labels, profile_name)

    # ROC + PR
    roc_auc, ap = plot_roc_curve(y_true_binary, test_errors, profile_name)
    print(f"  ROC AUC: {roc_auc:.4f}, Average Precision: {ap:.4f}")

    # Matrice de confusion binaire
    plot_confusion_matrix_ae(y_true_binary, y_pred_binary, profile_name)

    # Métriques par type d'anomalie
    plot_metrics_per_anomaly_type(test_errors, test_labels, threshold, profile_name)

    # Exemples de reconstruction
    plot_reconstruction_examples(model, test_cwt, test_labels, profile_name)

    # Signaux avec erreurs
    plot_example_signals_with_errors(test_windows, test_errors, test_labels,
                                     threshold, profile_name)

    # Matrice de confusion multi-classes (approximative)
    # Pour l'autoencoder, on ne peut distinguer le TYPE d'anomalie,
    # seulement normal vs anomalie. Mais on montre quels types sont détectés.
    y_pred_multiclass = test_labels.copy()
    # Si prédit normal mais était anomalie → on met 0
    y_pred_multiclass[y_pred_binary == 0] = 0
    plot_confusion_matrix_multiclass(test_labels, y_pred_multiclass, profile_name)

    # Sauvegarder le modèle
    model_path = OUTPUT_DIR / f'{profile_name}_autoencoder.pth'
    torch.save(model.state_dict(), model_path)
    print(f"  Modèle sauvegardé: {model_path}")

    all_results[profile_name] = {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'roc_auc': roc_auc, 'avg_precision': ap,
        'threshold': threshold, 'history': history,
    }

# =====================================================================
# 9. Résumé global
# =====================================================================
print("\n" + "=" * 70)
print("9. RÉSUMÉ GLOBAL — AUTOENCODER TOUS PROFILS")
print("=" * 70)

summary_data = []
for profile_name, res in all_results.items():
    summary_data.append({
        'Profil': profile_name,
        'Accuracy': f"{res['accuracy']:.4f}",
        'Precision': f"{res['precision']:.4f}",
        'Recall': f"{res['recall']:.4f}",
        'F1-Score': f"{res['f1']:.4f}",
        'ROC AUC': f"{res['roc_auc']:.4f}",
        'Avg Prec': f"{res['avg_precision']:.4f}",
        'Seuil': f"{res['threshold']:.6f}",
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Barplot comparatif
fig, ax = plt.subplots(figsize=(12, 6))
profiles_list = list(all_results.keys())
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
x = np.arange(len(profiles_list))
width = 0.15

for i, m in enumerate(metrics):
    vals = [all_results[p][m] for p in profiles_list]
    ax.bar(x + i * width, vals, width, label=m.upper())

ax.set_xlabel('Profil')
ax.set_ylabel('Score')
ax.set_title('Autoencoder — Comparaison des Métriques par Profil',
             fontsize=14, fontweight='bold')
ax.set_xticks(x + 2 * width)
ax.set_xticklabels(profiles_list, rotation=15)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'global_autoencoder_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Autoencoder CWT+CNN terminé! Résultats sauvegardés dans: {OUTPUT_DIR}")
