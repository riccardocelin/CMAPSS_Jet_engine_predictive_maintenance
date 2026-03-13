import os
import requests
import numpy as np
import pandas as pd

base_url = "http://127.0.0.1:8000"
endpoint = "/predict"

data_folder = "data/processed/sequence/v1/test_FD001_X_y_v1.npz"
data_folder = "data/processed/tabular/v1/test_FD001_X_v1.csv"
sequence_model = False # Sequence model: True -> test for LSTM models (sequence), model: False -> test for RF models (tabular)


def predict(url="http://127.0.0.1:8000/predict", data=None, verbose=True, timeout=50):
    """
    Make a POST request to get the prediction from the RUL model served via FastAPI.

    Args:
        url: URL that the request is sent to.
        data: model input
        verbose: print diagnostic information.
        timeout: request timeout in seconds.

    Returns:
        dict: parsed JSON response from the server.
    """

    if data is None:
        raise ValueError("'data' cannot be None.")

    # Accept numpy arrays directly and convert to JSON-serializable structure.
    #sequences = data.tolist() if isinstance(data, np.ndarray) else data
    sequences = None
    if sequence_model:
        sequences = data.tolist()
    else:
        sequences = data.values.tolist()

    payload = {"inputs": sequences}

    response = requests.post(url, json=payload, timeout=timeout)

    if verbose:
        print(f"POST {url} -> {response.status_code}")
        if response.ok:
            print("Prediction succeeded.")
        else:
            print(f"Request failed: {response.text}")

    # Raise a useful error if the API returned non-200.
    response.raise_for_status()
    return response.json()

def load_test_input(data_fullfile, sequence_flag=True):

    X = None

    # load test data for api prediction
    if sequence_flag:
        X_y = np.load(data_fullfile)

        X_with_engine = X_y["X"]
        y = X_y["y"]

        idx_engine_id = X_y["feature_names"].tolist().index("engine_id")

        f_name = X_y["feature_names"].tolist()
        f_name_no_engine_id = f_name.copy()
        f_name_no_engine_id.pop(idx_engine_id)

        X = np.delete(X_with_engine, idx_engine_id, axis=2)  # no engine_id feature

    else: # tabular data
        X_with_engine = pd.read_csv(data_fullfile)
        X = X_with_engine.loc[:, X_with_engine.columns != "engine_id"]

    return X


def main():

    url = base_url + endpoint
    actual_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.dirname(actual_dir) + "/" + data_folder

    X = load_test_input(data_dir, sequence_model)

    # test server connection
    test_response = requests.get(base_url + '/')
    print(f"Test response: {test_response}")

    response = predict(url, X)
    print(f"Predicted RUL: {response['predictions']}")

if __name__ == "__main__":
    main()
