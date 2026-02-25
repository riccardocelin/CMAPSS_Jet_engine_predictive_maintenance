from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import tensorflow as tf
import numpy as np
import os 

# sensor data item
class JetEngineData(BaseModel):
    sequences : List[List[List[float]]]

# load TF model
actual_dir = os.path.dirname(os.path.realpath(__file__))
model_filename = 'RUL_lstm-0.1.0.keras'
model_fullfile = actual_dir + '/models/RUL_lstm-0.1.0/RUL_lstm-0.1.0.keras'
model = tf.keras.models.load_model(model_fullfile)

WINDOW_SIZE = 10
N_FEATURES = 18

# API
app = FastAPI()

@app.get("/")
def read_root():
    return{"health_status": "ok", "model_filename": model_filename}

@app.post('/')
def predict_None_RUL():
    return {
        "predicted_RUL": None,
        "batch_size": None
    }

@app.post('/predict')
def predict_RUL(data: JetEngineData):
    
    batch = np.array(data.sequences, dtype=np.float32)

    # Validate dimensionality
    if batch.ndim != 3:
        raise HTTPException(status_code=400, detail="Input must be (batch_size, window_size, n_features)")

    # Validate shape (excluding batch dimension)
    if batch.shape[1:] != (WINDOW_SIZE, N_FEATURES):
        raise HTTPException(
            status_code=400,
            detail=f"Each sequence must have shape ({WINDOW_SIZE}, {N_FEATURES})"
        )

    # Model expects (batch_size, window_size, n_features)
    predictions = model.predict(batch)

    return {
        "predicted_RUL": predictions.tolist(),
        "batch_size": batch.shape[0]
    }