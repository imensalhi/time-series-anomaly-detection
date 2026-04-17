# Time-Series Anomaly Detection Pipeline (Electrical Current)

This project implements an end-to-end MLOps pipeline for anomaly detection on industrial current signals exported from InfluxDB.

## What is implemented

- Preprocessing service:
  - parse Influx CSV format
  - detect null values and missing timestamps (gaps)
  - fill data with interpolation
  - detect run and idle periods
  - train only on run data
- CWT service:
  - compute wavelet scalograms for CWT-CNN autoencoder
- Clustering service:
  - KMeans grouping of machines from run-signal statistics
  - prediction for a new machine and new-group detection
- Training service:
  - model 1: CWT-CNN autoencoder trained on normal run windows
  - model 2: contrastive 1D encoder (without CWT) trained on normal run windows
- Evaluation service:
  - inject 6 synthetic anomalies in test split
  - optimize threshold for anomaly scores
  - confusion matrices, precision, recall, F1
- Fine-tuning service and transfer service
- MLflow integration for experiment tracking
- DVC pipeline orchestration
- Deepchecks validation reports
- GitHub Actions CI/CD

## Pipeline stages (DVC)

1. preprocess
2. clustering
3. train_autoencoder
4. train_contrastive
5. evaluate
6. validate

Run all stages:

```bash
dvc repro
```

## Main commands

```bash
make install
make train-all
make fine-tune
make transfer
make mlflow-ui
```

## Key folders

- data/: raw Influx CSV exports
- artifacts/preprocessed/: cleaned data + run/idle masks
- src/services/: modular service layer
- results_autoencoder/: autoencoder models
- results_contrastive/: contrastive models
- metrics/: JSON metrics for DVC and reporting
- reports/: confusion matrices and Deepchecks HTML reports

## Anomaly types injected in evaluation

- spike
- plateau
- drift
- variance
- dropout
- shape_shift