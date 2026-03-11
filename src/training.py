import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = str(PROJECT_ROOT) + "/configs/training_config.local.json"  # remember to create a new file copy from configs/training_config.template.json


def flatten_dict(data, parent_key=""):
    flat = {}
    for key, value in data.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, new_key))
        else:
            flat[new_key] = value
    return flat


def log_params(params):
    for key, value in params.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            mlflow.log_param(key, json.dumps(list(value)))
        else:
            mlflow.log_param(key, value)


def print_mlflow_server_instructions():
    print("\n[MLFLOW] To start a local MLflow tracking server:")
    print(
        "  mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns"
    )
    print("[MLFLOW] Then set the tracking URI before running training:")
    print("  export MLFLOW_TRACKING_URI=http://127.0.0.1:5000\n")


def load_config(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tabular_data(cfg, is_training=True):

    label = ""
    if is_training is True:
        label = "training"
    else:
        label = "test"

    # load training dataset
    data_folder = cfg["data_base_folder"] + "/" + cfg["data_type"] + "/" + cfg["data_version"]
    dataset_X_name = cfg[label]["dataset_tabular_X_name"]
    dataset_y_name = cfg[label]["dataset_tabular_y_name"]

    data_paths = {
        "data_folder": data_folder,
        "dataset_X_name": dataset_X_name,
        "dataset_y_name": dataset_y_name,
    }

    X_with_engine = pd.read_csv(data_folder + "/" + dataset_X_name + ".csv")
    y = pd.read_csv(data_folder + "/" + dataset_y_name + ".csv")

    X = X_with_engine.loc[:, X_with_engine.columns != "engine_id"]

    return X, X_with_engine, y, data_paths


def load_sequence_data(cfg, is_training=True):

    label = ""
    if is_training is True:
        label = "training"
    else:
        label = "test"

    # load training dataset
    data_folder = cfg["data_base_folder"] + "/" + cfg["data_type"] + "/" + cfg["data_version"]
    dataset_X_y_name = cfg[label]["dataset_sequence_X_y_name"]

    data_paths = {"data_folder": data_folder, "dataset_X_y_name": dataset_X_y_name}

    X_y = np.load(data_folder + "/" + dataset_X_y_name + ".npz")

    X_with_engine = X_y["X"]
    y = X_y["y"]

    idx_engine_id = X_y["feature_names"].tolist().index("engine_id")

    f_name = X_y["feature_names"].tolist()
    f_name_no_engine_id = f_name.copy()
    f_name_no_engine_id.pop(idx_engine_id)

    X = np.delete(X_with_engine, idx_engine_id, axis=2)  # no engine_id feature

    variable_names = {
        "feature_names_X": f_name_no_engine_id,
        "feature_names_X_with_engine": f_name,
        "y_name": X_y["y_name"].tolist(),
    }

    return X, X_with_engine, y, idx_engine_id, data_paths, variable_names


def train_random_forest(cfg, cfg_rf):

    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # load datasets
    X, X_with_engine, y, training_dataset_paths = load_tabular_data(cfg, is_training=True)
    X_test, _, y_test, test_dataset_paths = load_tabular_data(cfg, is_training=False)

    experiment_name = cfg.get("mlflow_experiment_name", "CMAPSS_Training")
    run_name = cfg.get("mlflow_run_name", "random_forest")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        log_params(flatten_dict({"common": cfg, "random_forest": cfg_rf}))

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "rf",
                    RandomForestRegressor(
                        random_state=cfg["random_state"],
                        n_jobs=cfg_rf["n_jobs"],
                    ),
                ),
            ]
        )

        model = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())

        cv = GroupKFold(n_splits=cfg_rf["n_splits_cv"])
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=cfg_rf["hyperparameters"],
            cv=cv,
            scoring=cfg_rf["scoring"],
            n_jobs=cfg_rf["n_jobs"],
            verbose=cfg_rf["verbose"],
        )

        grid_search.fit(X, y, groups=X_with_engine["engine_id"])

        # metrics
        mae_cv_trn = abs(grid_search.best_score_)
        y_test_pred = grid_search.predict(X_test)
        mae_cv_tst = mean_absolute_error(y_test, y_test_pred)

        metrics_trn = {"Training set": training_dataset_paths, "MAE VAL": mae_cv_trn}
        metrics_tst = {"Test set": test_dataset_paths, "MAE TST": mae_cv_tst}

        # mlflow logs
        mlflow.log_metric("mae_val", mae_cv_trn)
        mlflow.log_metric("mae_test", mae_cv_tst)
        mlflow.log_param("best_params", json.dumps(grid_search.best_params_))
        mlflow.log_param("dataset_train_rows", X.shape[0])
        mlflow.log_param("dataset_test_rows", X_test.shape[0])
        mlflow.log_param("dataset_feature_count", X.shape[1])
        mlflow.log_param("dataset_train_path", json.dumps(training_dataset_paths))
        mlflow.log_param("dataset_test_path", json.dumps(test_dataset_paths))
        mlflow.sklearn.log_model(grid_search.best_estimator_, artifact_path="model")

        print(metrics_trn)
        print(metrics_tst)


def train_lstm(cfg, cfg_lstm):

    import tensorflow as tf
    from sklearn.model_selection import GroupShuffleSplit
    from tensorflow.keras import layers, models

    tf.random.set_seed(cfg["random_state"])

    X, X_with_engine, y, idx_engine_id, data_paths_training, variable_names = load_sequence_data(cfg, is_training=True)
    X_test, _, y_test, _, data_paths_test, _ = load_sequence_data(cfg, is_training=False)

    experiment_name = cfg.get("mlflow_experiment_name", "CMAPSS_Training")
    run_name = cfg.get("mlflow_run_name", "lstm")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        log_params(flatten_dict({"common": cfg, "lstm": cfg_lstm}))

        # GroupShuffleSplit per engine
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg_lstm["train_val_ratio"], random_state=cfg["random_state"])
        train_idx, val_idx = next(gss.split(X, y, groups=X_with_engine[:, idx_engine_id, 0]))

        # split
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        norm = layers.Normalization()
        norm.adapt(X_train)

        model = models.Sequential(
            [
                layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
                norm,
                layers.LSTM(cfg_lstm["layers"][0]["units"], return_sequences=True),
                layers.Dropout(cfg_lstm["layers"][0]["dropout"]),
                layers.LSTM(cfg_lstm["layers"][1]["units"], return_sequences=False),
                layers.Dense(1),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg_lstm["learning_rate"]),
            loss=cfg_lstm["loss"],
            metrics=cfg_lstm["metrics"],
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=cfg_lstm["epochs"],
            batch_size=cfg_lstm["batch_size"],
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor=cfg_lstm["early_stopping"]["monitor"],
                    patience=cfg_lstm["early_stopping"]["patience"],
                    restore_best_weights=cfg_lstm["early_stopping"]["restore_best_weights"],
                )
            ],
            verbose=cfg_lstm["verbose"],
        )

        y_val_pred = model.predict(X_val, verbose=0).flatten()
        y_test_pred = model.predict(X_test, verbose=0).flatten()

        mae_val = mean_absolute_error(y_val[:, -1], y_val_pred)
        mae_test = mean_absolute_error(y_test[:, -1], y_test_pred)

        metrics_trn = {"Training set": data_paths_training, "MAE VAL": mae_val}
        metrics_tst = {"Test set": data_paths_test, "MAE TST": mae_test}

        mlflow.log_metric("mae_val", mae_val)
        mlflow.log_metric("mae_test", mae_test)
        mlflow.log_param("dataset_train_sequences", X.shape[0])
        mlflow.log_param("dataset_test_sequences", X_test.shape[0])
        mlflow.log_param("dataset_timesteps", X.shape[1])
        mlflow.log_param("dataset_feature_count", X.shape[2])
        mlflow.log_param("feature_names", json.dumps(variable_names["feature_names_X"]))
        mlflow.log_param("dataset_train_path", json.dumps(data_paths_training))
        mlflow.log_param("dataset_test_path", json.dumps(data_paths_test))
        mlflow.log_param("epochs_trained", len(history.history.get("loss", [])))
        mlflow.tensorflow.log_model(model, artifact_path="model")

        print(metrics_trn)
        print(metrics_tst)


def main() -> None:
    print_mlflow_server_instructions()
    config = load_config(CONFIG_PATH)
    cfg = config["common"]  # common config

    algorithm = cfg["algorithm"].lower()

    if algorithm == "random_forest":

        if cfg["data_type"] == "sequence":
            raise ValueError("wrong data_type in config, for RF has to be 'tabular'")

        rf_cfg = config["random_forest"]
        train_random_forest(cfg, rf_cfg)

    elif algorithm == "lstm":

        if cfg["data_type"] == "tabular":
            raise ValueError("wrong data_type in config, for LSTM has to be 'sequence'")

        lstm_cfg = config["lstm"]
        train_lstm(cfg, lstm_cfg)

    else:
        raise ValueError("algorithm type not found in config")


if __name__ == "__main__":
    main()
