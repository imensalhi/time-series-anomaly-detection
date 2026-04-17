.PHONY: install lint preprocess cluster train-autoencoder train-contrastive evaluate validate \
	fine-tune transfer train-all pipeline clean mlflow-ui

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
# Pipeline services
# ──────────────────────────────────────────────────────────────────────────────
preprocess:
	python src/preprocess.py

cluster:
	python src/cluster_machine.py

train-autoencoder:
	python src/train_autoencoder.py

train-contrastive:
	python src/train_contrastive.py

evaluate:
	python src/evaluate_models.py

# ──────────────────────────────────────────────────────────────────────────────
# Validation (Deepchecks)
# ──────────────────────────────────────────────────────────────────────────────
validate:
	python src/validate.py

# ──────────────────────────────────────────────────────────────────────────────
# Fine-tuning and transfer learning
# ──────────────────────────────────────────────────────────────────────────────
fine-tune:
	python src/fine_tune.py

transfer:
	python src/transfer.py

# ──────────────────────────────────────────────────────────────────────────────
# Run core pipeline services locally
# ──────────────────────────────────────────────────────────────────────────────
train-all: preprocess cluster train-autoencoder train-contrastive evaluate validate

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
	rm -rf results_autoencoder/ results_contrastive/ artifacts/ reports/ metrics/ \
	       __pycache__ src/__pycache__
