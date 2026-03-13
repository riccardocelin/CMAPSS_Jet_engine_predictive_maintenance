from fastapi import FastAPI
import mlflow.pyfunc
import numpy as np
import pandas as pd


####################################################################################
####################################################################################

mlflow_uri = "http://127.0.0.1:8080"
model_name = 'RF_RUL_estimator' # for now: only sequence .keras models are supported (LSTM trained with TF2.x)
alias = "champion" # @champion for deployment (alias set in mlflow model registry)

####################################################################################
####################################################################################

mlflow.set_tracking_uri(mlflow_uri)

# Load the model version currently under the 'champion' alias
model_uri = f"models:/{model_name}@{alias}"
model = mlflow.pyfunc.load_model(model_uri)

app = FastAPI()

@app.get("/")
def read_root():
    return{"health_status": "ok"}

@app.post("/predict")
def predict(data: dict):
    x = data["inputs"]
    preds = model.predict(x)
    return {"predictions": preds.tolist()}