from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import tensorflow as tf
import numpy as np
import json
import os 

####################################################################################
####################################################################################

model_name = 'RUL_lstm-0.1.1' # for now: only sequence .keras models are supported (LSTM trained with TF2.x)

####################################################################################
####################################################################################


# support function
def load_model_and_artifacts(model_fullfile, model_artifact_fullfile):
    model = tf.keras.models.load_model(model_fullfile)

    artifacts = ""
    with open(model_artifact_fullfile) as f:
        artifacts = json.load(f)

    return model, artifacts

# sensor data item
class JetEngineData(BaseModel):
    sequences : List[List[List[float]]]


# load TF model
actual_dir = os.path.dirname(os.path.realpath(__file__))
model_fullfile = actual_dir + '/models/' + model_name + '/' + model_name + '.keras'
model_artifact_fullfile = actual_dir + '/models/' + model_name + '/' + model_name + '.json' # TODO: explore MLflow for experiment tracking
model, artifacts = load_model_and_artifacts(model_fullfile, model_artifact_fullfile)

# artifact explicited
window_size = artifacts['model_input']['windows_size']
n_features = artifacts['model_input']['features_number']
X_m = artifacts['normalization_factors']['X_mean'] # shape (1, {N_FEATURES})
X_std = artifacts['normalization_factors']['X_std'] # shape (1, {N_FEATURES})
y_m = artifacts['normalization_factors']['y_mean'] # shape (1, 1)
y_std = artifacts['normalization_factors']['y_std'] # shape (1, 1)



# API
app = FastAPI()

@app.get("/")
def read_root():
    return{"health_status": "ok", "model_filename": model_name}

@app.post('/')
def predict_None_RUL():
    return {
        "predicted_RUL": None,
        "batch_size": None
    }

@app.post('/predict')
def predict_RUL(data: JetEngineData):
    
    batches = np.array(data.sequences, dtype=np.float32)

    # Validate dimensionality
    if batches.ndim != 3:
        raise HTTPException(status_code=400, detail="Input must be (batch_size, window_size, n_features)")

    # Validate shape (excluding batch dimension)
    if batches.shape[1:] != (window_size, n_features):
        raise HTTPException(
            status_code=400,
            detail=f"Each sequence must have shape ({window_size}, {n_features})"
        )

    # input normalization step
    batches_norm = (batches - X_m) / X_std

    # Model expects (batch_size, window_size, n_features)
    pred_norm = model.predict(batches_norm)

    # output normalization step
    pred = (pred_norm * y_std) + y_m

    return {
        "predicted_RUL": pred.tolist(),
        "batch_size": batches.shape[0]
    }