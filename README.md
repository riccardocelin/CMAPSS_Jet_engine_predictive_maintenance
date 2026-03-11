# CMAPSS_Jet_engine_predictive_maintenance
The objective of the project is to predict the number of remaining operational cycles before failure in the test set, i.e. the number of operational cycles after the last cycle that the engine will continue to operate.

## MLflow experiment tracking
`src/training.py` now logs experiment runs to MLflow for both supported algorithms (Random Forest and LSTM), including:
- hyperparameters from config,
- dataset metadata (paths, shape information),
- validation/test MAE metrics,
- trained model artifact.

### Start a local MLflow tracking server
```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Then point the training script to that server:
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python src/training.py
```

You can open the UI at: `http://127.0.0.1:5000`.
