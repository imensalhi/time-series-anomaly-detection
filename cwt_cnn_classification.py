"""
=============================================================================
CWT + CNN Classification - Détection d'Anomalies par Classification Supervisée
=============================================================================
Stratégie Expert :
- Les 8 capteurs sont regroupés par profil de distribution (EDA) :
    Profil 1 (Stable/bruit)     : I23, I61
    Profil 2 (ON/OFF bimodal)   : I42, I43
    Profil 3 (Multi-niveaux)    : I52, I11, I113, I32
- Pour chaque profil, on concatène les signaux des capteurs similaires
  et on entraîne UN SEUL modèle CNN par profil.
- Cela permet au modèle d'apprendre des patterns généralisés au sein
  d'un profil et évite le surajustement sur un seul capteur.

Pipeline :
1. Charger et normaliser les signaux (colonne _value uniquement)
2. Découper en fenêtres (windows) de taille fixe
3. Injecter 6 types d'anomalies synthétiques dans train ET test
4. Appliquer la CWT (Continuous Wavelet Transform) sur chaque fenêtre
   → image temps-fréquence (scalogramme)
5. Entraîner un CNN 2D pour classifier : normal vs 6 types d'anomalies
6. Évaluer avec métriques, matrice de confusion, graphes par type
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
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

WINDOW_SIZE = 128          # Taille de la fenêtre temporelle
STEP_SIZE = 64             # Pas de glissement (overlap = 50%)
SCALES = np.arange(1, 65)  # 64 échelles CWT → image 64 x 128
WAVELET = 'morl'           # Ondelette de Morlet
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
ANOMALY_RATIO = 0.15       # 15% d'anomalies injectées
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = Path('data')
OUTPUT_DIR = Path('results_classification')
OUTPUT_DIR.mkdir(exist_ok=True)

# Couleurs par type d'anomalie (comme demandé)
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

# Profils de capteurs identifiés dans l'EDA
PROFILES = {
    "profil_1_stable": ["I23", "I61"],
    "profil_2_bimodal": ["I42", "I43"],
    "profil_3_multi": ["52", "I11", "I113", "I32"],
}

print(f"Device: {DEVICE}")
print(f"Window: {WINDOW_SIZE}, Scales: {len(SCALES)}, Wavelet: {WAVELET}")

# =====================================================================
# 1. Chargement des données
# =====================================================================
print("\n" + "=" * 70)
print("1. CHARGEMENT DES DONNÉES")
print("=" * 70)


def load_dataset(file_path):
    """Charge un CSV InfluxDB et retourne la série _value."""
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
# 2. Injection d'anomalies synthétiques
# =====================================================================
print("\n" + "=" * 70)
print("2. INJECTION D'ANOMALIES SYNTHÉTIQUES")
print("=" * 70)


def inject_spike(window, std):
    """Spike : pic brutal de 5-10x l'écart-type sur 1-3 points."""
    w = window.copy()
    n_spikes = np.random.randint(1, 4)
    for _ in range(n_spikes):
        pos = np.random.randint(0, len(w))
        direction = np.random.choice([-1, 1])
        magnitude = np.random.uniform(5, 10) * std
        w[pos] += direction * magnitude
    return w


def inject_plateau(window, mean):
    """Plateau : valeur constante sur 30-70% de la fenêtre."""
    w = window.copy()
    length = np.random.randint(int(0.3 * len(w)), int(0.7 * len(w)))
    start = np.random.randint(0, len(w) - length)
    plateau_val = mean + np.random.uniform(-0.5, 0.5) * mean
    w[start:start + length] = plateau_val
    return w


def inject_drift(window, std):
    """Drift : dérive linéaire progressive sur toute la fenêtre."""
    w = window.copy()
    drift_magnitude = np.random.uniform(2, 5) * std
    direction = np.random.choice([-1, 1])
    drift = np.linspace(0, direction * drift_magnitude, len(w))
    w += drift
    return w


def inject_variance(window, std):
    """Variance : changement brutal de variance sur une portion."""
    w = window.copy()
    length = np.random.randint(int(0.3 * len(w)), int(0.7 * len(w)))
    start = np.random.randint(0, len(w) - length)
    factor = np.random.uniform(3, 6)
    noise = np.random.normal(0, factor * std, length).astype(np.float32)
    w[start:start + length] += noise
    return w


def inject_dropout(window):
    """Dropout : valeurs tombent à zéro sur 20-50% de la fenêtre."""
    w = window.copy()
    length = np.random.randint(int(0.2 * len(w)), int(0.5 * len(w)))
    start = np.random.randint(0, len(w) - length)
    w[start:start + length] = 0.0
    return w


def inject_shape_shift(window, std):
    """Shape shift : remplacement par un pattern sinusoïdal anormal."""
    w = window.copy()
    length = np.random.randint(int(0.4 * len(w)), int(0.8 * len(w)))
    start = np.random.randint(0, len(w) - length)
    freq = np.random.uniform(2, 8)
    amplitude = np.random.uniform(2, 4) * std
    t = np.linspace(0, 2 * np.pi * freq, length)
    w[start:start + length] = w[start:start + length].mean() + amplitude * np.sin(t)
    return w


def inject_anomaly(window, anomaly_type, mean, std):
    """Injecte une anomalie du type demandé."""
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


# =====================================================================
# 3. Fenêtrage et préparation des données par profil
# =====================================================================
print("\n" + "=" * 70)
print("3. FENÊTRAGE ET PRÉPARATION PAR PROFIL")
print("=" * 70)


def create_windows(signal, window_size, step_size):
    """Découpe le signal en fenêtres glissantes."""
    windows = []
    for i in range(0, len(signal) - window_size + 1, step_size):
        windows.append(signal[i:i + window_size])
    return np.array(windows)


def prepare_profile_data(profile_name, sensor_keys, anomaly_ratio=ANOMALY_RATIO):
    """
    Prépare les données pour un profil entier :
    1. Concatène les fenêtres de tous les capteurs du profil
    2. Normalise par capteur (StandardScaler)
    3. Injecte des anomalies aléatoires
    4. Split train/test
    """
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
        print(f"  {key}: {len(windows)} fenêtres")

    all_windows = np.concatenate(all_windows, axis=0)
    n_total = len(all_windows)
    n_anomalies = int(n_total * anomaly_ratio)
    n_normal = n_total - n_anomalies

    # Labels : 0 = normal
    labels = np.zeros(n_total, dtype=np.int64)

    # Sélectionner les indices à corrompre
    anomaly_indices = np.random.choice(n_total, n_anomalies, replace=False)

    # Statistiques globales pour le calibrage des anomalies
    global_mean = all_windows.mean()
    global_std = all_windows.std()

    anomaly_counts = {t: 0 for t in ANOMALY_TYPES}
    for idx in anomaly_indices:
        atype = np.random.choice(ANOMALY_TYPES)
        all_windows[idx] = inject_anomaly(
            all_windows[idx], atype, global_mean, global_std
        )
        labels[idx] = LABEL_MAP[atype]
        anomaly_counts[atype] += 1

    print(f"  Total: {n_total} fenêtres ({n_normal} normal, {n_anomalies} anomalies)")
    for t, c in anomaly_counts.items():
        print(f"    {t:>12s}: {c}")

    return all_windows, labels


# =====================================================================
# 4. CWT (Continuous Wavelet Transform)
# =====================================================================
print("\n" + "=" * 70)
print("4. TRANSFORMATION CWT (Scalogrammes)")
print("=" * 70)


def compute_cwt_batch(windows, scales=SCALES, wavelet=WAVELET):
    """Calcule la CWT pour un batch de fenêtres → scalogrammes."""
    scalograms = []
    for i, w in enumerate(windows):
        coeffs, _ = pywt.cwt(w, scales, wavelet)
        scalograms.append(np.abs(coeffs))
        if (i + 1) % 500 == 0:
            print(f"    CWT: {i + 1}/{len(windows)} fenêtres traitées...")
    return np.array(scalograms, dtype=np.float32)


# =====================================================================
# 5. Modèle CNN
# =====================================================================
class CWT_CNN_Classifier(nn.Module):
    """
    CNN 2D pour classification de scalogrammes CWT.
    Input : (batch, 1, n_scales, window_size) = (batch, 1, 64, 128)
    Output : 7 classes (normal + 6 types d'anomalies)
    """
    def __init__(self, n_classes=7, n_scales=64, window_size=128):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =====================================================================
# 6. Fonctions d'entraînement et évaluation
# =====================================================================
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    """Entraîne le modèle et retourne l'historique."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            _, preds = outputs.max(1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    return history


def evaluate_model(model, test_loader):
    """Évalue le modèle et retourne prédictions + vrai labels."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    return np.array(all_preds), np.array(all_labels)


# =====================================================================
# 7. Visualisation
# =====================================================================
def plot_training_history(history, profile_name):
    """Courbes de loss et accuracy train/val."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['train_loss'], label='Train Loss', color='#E74C3C')
    ax1.plot(history['val_loss'], label='Val Loss', color='#3498DB')
    ax1.set_title(f'{profile_name} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['train_acc'], label='Train Acc', color='#E74C3C')
    ax2.plot(history['val_acc'], label='Val Acc', color='#3498DB')
    ax2.set_title(f'{profile_name} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{profile_name}_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, profile_name):
    """Matrice de confusion avec couleurs par type."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title(f'{profile_name} - Matrice de Confusion', fontsize=14, fontweight='bold')
    ax.set_xlabel('Prédit')
    ax.set_ylabel('Réel')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{profile_name}_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_metrics_by_anomaly(y_true, y_pred, profile_name):
    """Barplot des métriques (precision, recall, F1) par type d'anomalie."""
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True)

    types_list = LABEL_NAMES
    precision_vals = [report[t]['precision'] for t in types_list]
    recall_vals = [report[t]['recall'] for t in types_list]
    f1_vals = [report[t]['f1-score'] for t in types_list]

    x = np.arange(len(types_list))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, precision_vals, width, label='Precision', color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x, recall_vals, width, label='Recall', color='#E74C3C', alpha=0.8)
    bars3 = ax.bar(x + width, f1_vals, width, label='F1-Score', color='#2ECC71', alpha=0.8)

    ax.set_xlabel('Type')
    ax.set_ylabel('Score')
    ax.set_title(f'{profile_name} - Métriques par Type d\'Anomalie', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(types_list, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    # Ajouter les valeurs sur les barres
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{profile_name}_metrics_by_type.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_example_anomalies(windows, labels, profile_name, n_examples=2):
    """Affiche des exemples de chaque type d'anomalie avec couleurs."""
    fig, axes = plt.subplots(len(LABEL_NAMES), n_examples, figsize=(14, 3 * len(LABEL_NAMES)))
    fig.suptitle(f'{profile_name} - Exemples de Signaux par Type', fontsize=16, fontweight='bold')

    for i, atype in enumerate(LABEL_NAMES):
        label_id = LABEL_MAP[atype]
        indices = np.where(labels == label_id)[0]
        color = ANOMALY_COLORS[atype]

        for j in range(min(n_examples, len(indices))):
            ax = axes[i, j] if n_examples > 1 else axes[i]
            ax.plot(windows[indices[j]], color=color, linewidth=1.2)
            ax.set_title(f'{atype}', fontsize=10, color=color, fontweight='bold')
            ax.grid(True, alpha=0.2)
            if j == 0:
                ax.set_ylabel('Amplitude')
            ax.set_xlabel('Temps')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{profile_name}_example_anomalies.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_cwt_examples(windows, labels, profile_name):
    """Affiche les scalogrammes CWT pour un exemple de chaque type."""
    fig, axes = plt.subplots(1, len(LABEL_NAMES), figsize=(3.5 * len(LABEL_NAMES), 4))
    fig.suptitle(f'{profile_name} - Scalogrammes CWT par Type', fontsize=14, fontweight='bold')

    for i, atype in enumerate(LABEL_NAMES):
        label_id = LABEL_MAP[atype]
        indices = np.where(labels == label_id)[0]
        if len(indices) == 0:
            continue

        window = windows[indices[0]]
        coeffs, _ = pywt.cwt(window, SCALES, WAVELET)
        ax = axes[i]
        ax.imshow(np.abs(coeffs), aspect='auto', cmap='jet', interpolation='bilinear')
        ax.set_title(f'{atype}', fontsize=10, color=ANOMALY_COLORS[atype], fontweight='bold')
        ax.set_ylabel('Scale')
        ax.set_xlabel('Time')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{profile_name}_cwt_examples.png', dpi=150, bbox_inches='tight')
    plt.show()


# =====================================================================
# 8. Pipeline principal — Entraînement par profil
# =====================================================================
print("\n" + "=" * 70)
print("8. PIPELINE D'ENTRAÎNEMENT PAR PROFIL")
print("=" * 70)

all_results = {}

for profile_name, sensor_keys in PROFILES.items():
    print(f"\n{'#' * 70}")
    print(f"# PROFIL : {profile_name} — Capteurs : {sensor_keys}")
    print(f"{'#' * 70}")

    # --- Préparer les données ---
    windows, labels = prepare_profile_data(profile_name, sensor_keys)

    # --- Visualiser des exemples ---
    print("\n  [Visualisation] Exemples de signaux par type...")
    plot_example_anomalies(windows, labels, profile_name)
    plot_cwt_examples(windows, labels, profile_name)

    # --- Split train/test ---
    X_train_w, X_test_w, y_train, y_test = train_test_split(
        windows, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    print(f"\n  Split: Train={len(X_train_w)}, Test={len(X_test_w)}")

    # --- Calculer les CWT ---
    print("\n  [CWT] Transformation des fenêtres en scalogrammes...")
    print("    Train:")
    X_train_cwt = compute_cwt_batch(X_train_w)
    print("    Test:")
    X_test_cwt = compute_cwt_batch(X_test_w)

    # Normaliser les scalogrammes
    cwt_max = X_train_cwt.max()
    if cwt_max > 0:
        X_train_cwt /= cwt_max
        X_test_cwt /= cwt_max

    # Ajouter la dimension channel (1 canal) : (N, 1, scales, time)
    X_train_tensor = torch.FloatTensor(X_train_cwt).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_cwt).unsqueeze(1)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n  Tensor shapes: Train={X_train_tensor.shape}, Test={X_test_tensor.shape}")

    # --- Entraîner le modèle ---
    print(f"\n  [TRAINING] CNN pour {profile_name}...")
    model = CWT_CNN_Classifier(
        n_classes=7, n_scales=len(SCALES), window_size=WINDOW_SIZE
    ).to(DEVICE)

    # Afficher l'architecture
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: {total_params:,} paramètres")

    history = train_model(model, train_loader, test_loader)

    # --- Évaluer ---
    y_pred, y_true = evaluate_model(model, test_loader)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n  {'=' * 50}")
    print(f"  RÉSULTATS — {profile_name}")
    print(f"  {'=' * 50}")
    print(f"  Accuracy :  {acc:.4f}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall   :  {rec:.4f}")
    print(f"  F1-Score :  {f1:.4f}")

    # Classification report détaillé
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0))

    # --- Visualisations ---
    print("  [Visualisation] Courbes d'entraînement...")
    plot_training_history(history, profile_name)

    print("  [Visualisation] Matrice de confusion...")
    plot_confusion_matrix(y_true, y_pred, profile_name)

    print("  [Visualisation] Métriques par type...")
    plot_metrics_by_anomaly(y_true, y_pred, profile_name)

    # Sauvegarder le modèle
    model_path = OUTPUT_DIR / f'{profile_name}_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"  Modèle sauvegardé: {model_path}")

    all_results[profile_name] = {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'history': history, 'y_true': y_true, 'y_pred': y_pred,
    }

# =====================================================================
# 9. Résumé global de tous les profils
# =====================================================================
print("\n" + "=" * 70)
print("9. RÉSUMÉ GLOBAL — TOUS LES PROFILS")
print("=" * 70)

summary_data = []
for profile_name, res in all_results.items():
    summary_data.append({
        'Profil': profile_name,
        'Accuracy': f"{res['accuracy']:.4f}",
        'Precision': f"{res['precision']:.4f}",
        'Recall': f"{res['recall']:.4f}",
        'F1-Score': f"{res['f1']:.4f}",
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Barplot comparatif des profils
fig, ax = plt.subplots(figsize=(12, 6))
profiles_list = list(all_results.keys())
metrics = ['accuracy', 'precision', 'recall', 'f1']
x = np.arange(len(profiles_list))
width = 0.2

for i, m in enumerate(metrics):
    vals = [all_results[p][m] for p in profiles_list]
    ax.bar(x + i * width, vals, width, label=m.capitalize())

ax.set_xlabel('Profil')
ax.set_ylabel('Score')
ax.set_title('Comparaison des Métriques par Profil', fontsize=14, fontweight='bold')
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(profiles_list, rotation=15)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'global_profile_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Classification CWT+CNN terminée! Résultats sauvegardés dans:", OUTPUT_DIR)
