# NOTE: this is a workaround to avoid using local mlflow uri due to the fact that this demo project aim to deploy the model on cloud
import mlflow
import shutil
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = str(PROJECT_ROOT) + "/configs/download_model_to_deploy.local.yaml"  # remember to create a new file copy from configs/download_model_to_deploy.template.yaml

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

tracking_uri = config["mlflow"]["tracking_uri"]
model_name = config["model"]["name"]
alias = config["model"]["alias"]
output_dir = config["export"]["output_dir"]

mlflow.set_tracking_uri(tracking_uri)

model_uri = f"models:/{model_name}@{alias}"

local_path = mlflow.artifacts.download_artifacts(model_uri)

shutil.copytree(local_path, output_dir, dirs_exist_ok=True)

