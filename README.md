# Time-Series Anomaly Detection — MLOps Pipeline

> Detect anomalies in industrial sensor time-series using three complementary
> deep-learning approaches, integrated into a reproducible MLOps pipeline with
> **MLflow**, **DVC**, **Deepchecks** and **GitHub Actions**.

---

## Architecture

```
Raw CSV data (InfluxDB format)
        │
        ▼
  ┌─────────────┐
  │  DVC Stage  │  data version control (dvc.yaml)
  └─────────────┘
        │
        ▼  sliding windows (128 pts, stride 64)
        │
  ┌──────────────────────────────────────────────────┐
  │                                                  │
  ▼                  ▼                               ▼
CWT + CNN         CWT + CNN                  Contrastive 1D
Classifier        Autoencoder                   Encoder
(supervised)      (unsupervised)           (self-supervised)
  │                  │                               │
  └──────────────────┴───────────────────────────────┘
                     │
            MLflow experiment tracking
            (metrics, params, artefacts, model registry)
                     │
            Deepchecks validation
            (data integrity + model performance)
                     │
            GitHub Actions CI/CD
            (lint → validate → train → report)
```

### Sensor Profiles (from EDA)

| Profile | Sensors | Behaviour |
|---------|---------|-----------|
| profil\_1\_stable  | I23, I61        | Stable / low noise |
| profil\_2\_bimodal | I42, I43        | ON/OFF bimodal     |
| profil\_3\_multi   | 52, I11, I113, I32 | Multi-level     |

### Anomaly Types (synthetic injection)

`spike` · `plateau` · `drift` · `variance` · `dropout` · `shape_shift`

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
# or
make install
```

### 2 — Run the full DVC pipeline

```bash
dvc repro          # reruns only changed stages
# or
make pipeline
```

### 3 — Train individual models

```bash
make train-classification   # CWT + CNN supervised classifier
make train-autoencoder      # CWT + CNN unsupervised autoencoder
make train-contrastive      # Contrastive self-supervised (1D)
```

### 4 — Run Deepchecks validation

```bash
make validate
# Reports are saved to reports/
```

### 5 — View MLflow UI

```bash
make mlflow-ui
# Open http://localhost:5000
```

---

## Project Structure

```
.
├── data/                          # Raw CSV sensor files (DVC-tracked)
├── src/
│   ├── __init__.py
│   ├── preprocess.py              # Shared data loading, windowing, anomaly injection
│   ├── train_classification.py   # CWT+CNN Classifier + MLflow
│   ├── train_autoencoder.py      # CWT+CNN Autoencoder + MLflow
│   ├── train_contrastive.py      # Contrastive 1D + MLflow
│   └── validate.py               # Deepchecks data + model validation
├── results_classification/        # Trained models & plots (DVC output)
├── results_autoencoder/           # Trained models & plots (DVC output)
├── results_contrastive/           # Trained models & plots (DVC output)
├── reports/                       # Deepchecks HTML reports (DVC output)
├── metrics/                       # JSON metrics consumed by DVC
├── .github/workflows/
│   ├── ci.yml                     # Lint + data validation on every push/PR
│   └── train.yml                  # Full pipeline on push to main
├── dvc.yaml                       # DVC pipeline definition
├── params.yaml                    # Centralised hyper-parameters
├── requirements.txt
├── Makefile
└── setup.cfg                      # flake8 configuration
```

---

## MLOps Tools

| Tool | Role |
|------|------|
| **MLflow** | Experiment tracking, metric logging, artefact storage, model registry |
| **DVC** | Reproducible ML pipeline, data versioning, metric comparison |
| **Deepchecks** | Automated data integrity checks & model performance validation |
| **GitHub Actions** | CI (lint + validate) and CD (full training on `main`) |

---

## Configuration

All hyper-parameters live in **`params.yaml`** and are read by every training
script.  DVC tracks changes to this file and re-runs the affected stages
automatically.

```yaml
# Example — change epochs for the classifier:
classification:
  epochs: 50          # was 30
```

Then run `dvc repro` — only the classification stage and the validation stage
will re-execute.

---

## CI / CD Pipelines

### `ci.yml` — triggered on every push / PR

1. **Lint** with `flake8`
2. **Data validation** with Deepchecks — reports uploaded as GitHub Artefacts

### `train.yml` — triggered on push to `main` (or manual dispatch)

1. Install dependencies
2. `dvc repro` — full pipeline (train × 3 + validate)
3. Upload results, metrics, MLflow runs, and Deepchecks reports as Artefacts

---

## Legacy Scripts

The original monolithic scripts are kept for reference:

| File | Description |
|------|-------------|
| `cwt_cnn_classification.py` | Original supervised classification script |
| `cwt_cnn_autoencoder.py` | Original autoencoder script |
| `contrastive.py` | Original contrastive learning script |
| `eda_adapter.ipynb` | Exploratory Data Analysis notebook |