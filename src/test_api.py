import os
import requests
import numpy as np

base_url = "http://127.0.0.1:8000"
endpoint = "/predict"
url = base_url + endpoint


def predict(url="http://127.0.0.1:8000/predict", data=None, verbose=True, timeout=10):
    """
    Make a POST request to get the prediction from the RUL model.

    Args:
        url (str): URL that the request is sent to.
        data (np.ndarray | list): input with shape (batch_size, window_size, n_features).
        verbose (bool): print diagnostic information.
        timeout (int): request timeout in seconds.

    Returns:
        dict: parsed JSON response from the server.
    """

    if data is None:
        raise ValueError("'data' cannot be None.")

    # Accept numpy arrays directly and convert to JSON-serializable structure.
    sequences = data.tolist() if isinstance(data, np.ndarray) else data
    payload = {"sequences": sequences}

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


def main():
    actual_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.dirname(actual_dir) + "/data"

    # load test data for api prediction
    data_batch = np.load(data_dir + "/data_test_api.npy")

    for data in data_batch:
        data_rsh = data[np.newaxis, ...]
        response = predict(url, data_rsh)
        print(f"Predicted RUL: {response['predicted_RUL']} | batch_size: {response['batch_size']}")


if __name__ == "__main__":
    main()
