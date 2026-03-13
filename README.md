# CMAPSS Jet Engine Predictive Maintenance

This project predicts **Remaining Useful Life (RUL)** for jet engines using NASA CMAPSS turbofan degradation simulation data.

The repository supports:
- dataset preprocessing for both **tabular ML** and **sequence DL** workflows,
- model training with either **Random Forest** or **LSTM**,
- experiment tracking with **MLflow**,
- online inference through a **FastAPI** service.

---

## Project objective
Given historical sensor readings from each engine, estimate how many cycles remain before failure.

The target variable is `RUL`, computed per engine cycle and optionally clipped by `max_rul` during preprocessing.

---

## Repository structure

Files 'example_training_ML.ipynb' and 'example_training_TF.ipynb' provides sample notebooks for the training pipeline for the CV-tuned RF model and for the LSTM model with some useful images.
Those files are only for example purpose only.
The structure of the project is explained below.

- `src/data_preprocess/dataset_generation.py`  
  End-to-end dataset preparation (load raw files, compute RUL, feature engineering, save processed artifacts).
- `src/data_preprocess/data_generation_fcn.py`  
  Helper functions: loading CMAPSS text files, sorting, RUL generation, missing-value handling, and constant-feature dropping.
- `src/training.py`  
  Trains either `random_forest` (tabular) or `lstm` (sequence), evaluates MAE, and logs runs to MLflow.
- `app/api.py`  
  FastAPI inference API loading a trained TensorFlow `.keras` model + normalization artifacts.
- `src/test_api.py`  
  Client script to test the `/predict` endpoint with batched sequence inputs.
- `configs/*.template.json`  
  Templates for preprocessing and training configuration.

---

## Data and preprocessing

### Expected raw data
The preprocessing config expects CMAPSS raw files under:

- `data/CMAPSSData/train_FD00x.txt`
- `data/CMAPSSData/test_FD00x.txt`
- `data/CMAPSSData/RUL_FD00x.txt`

(These raw data files are not included in this repository and should be added locally.)

### Generate processed datasets
1. Create a local config from template:
   - copy `configs/dataset_generation_config.template.json` to `configs/dataset_generation_config.local.json`
2. Update paths and options in the local config.
3. Run preprocessing:

```bash
python src/data_preprocess/dataset_generation.py
```

Depending on `is_sequence_modeling`:
- `false` → tabular CSVs (`*_X_*.csv`, `*_y_*.csv`)
- `true` → sequence `.npz` files (`*_X_y_*.npz`)

Processed datasets are written under `data/processed/{tabular|sequence}/{dataset_version}`.

---

## Training

### Configure training
1. Create a local config from template:
   - copy `configs/training_config.template.json` to `configs/training_config.local.json`
2. Set:
   - `common.algorithm` to `"random_forest"` or `"lstm"`
   - `common.data_type` to match the algorithm (`tabular` for RF, `sequence` for LSTM)
   - dataset names and version paths under `common.training` / `common.test`

### Run training
```bash
python src/training.py
```

The training script logs parameters, metrics, dataset metadata, and model artifacts to MLflow.

---

## MLflow experiment tracking (port 8080)

Use MLflow on **port 8080**.

### Start local MLflow server
```bash
mlflow server \
  --host 0.0.0.0 \
  --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

### Point training to MLflow
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:8080
python src/training.py
```

Open the UI at:

- `http://127.0.0.1:8080`

> Note: `configs/training_config.template.json` already uses `"mlflow_server_uri": "http://127.0.0.1:8080"`.

---

## Inference API

Start the API server:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /` → health check and loaded model name
- `POST /predict` → RUL prediction for batches shaped `(batch_size, window_size, n_features)`

Quick API test:

```bash
python src/test_api.py
```

---

## Docker

You can also run the API with Docker Compose:

```bash
docker compose up --build
```

API will be available at `http://localhost:8000`.
