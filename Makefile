.PHONY: install lint train-classification train-autoencoder train-contrastive validate \
        train-all pipeline clean mlflow-ui

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# ──────────────────────────────────────────────────────────────────────────────
# Code quality
# ──────────────────────────────────────────────────────────────────────────────
lint:
	flake8 src/ --max-line-length=100 --ignore=E203,W503

# ──────────────────────────────────────────────────────────────────────────────
# Individual training stages
# ──────────────────────────────────────────────────────────────────────────────
train-classification:
	python src/train_classification.py

train-autoencoder:
	python src/train_autoencoder.py

train-contrastive:
	python src/train_contrastive.py

# ──────────────────────────────────────────────────────────────────────────────
# Validation (Deepchecks)
# ──────────────────────────────────────────────────────────────────────────────
validate:
	python src/validate.py

# ──────────────────────────────────────────────────────────────────────────────
# Run all training stages then validate
# ──────────────────────────────────────────────────────────────────────────────
train-all: train-classification train-autoencoder train-contrastive validate

# ──────────────────────────────────────────────────────────────────────────────
# DVC pipeline (recommended — reruns only changed stages)
# ──────────────────────────────────────────────────────────────────────────────
pipeline:
	dvc repro

# ──────────────────────────────────────────────────────────────────────────────
# Launch MLflow UI (http://localhost:5000)
# ──────────────────────────────────────────────────────────────────────────────
mlflow-ui:
	mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000

# ──────────────────────────────────────────────────────────────────────────────
# Clean generated outputs (keeps data/ and mlruns/)
# ──────────────────────────────────────────────────────────────────────────────
clean:
	rm -rf results_classification/ results_autoencoder/ results_contrastive/ \
	       reports/ metrics/ __pycache__ src/__pycache__
