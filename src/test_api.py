import requests
import numpy as np
import os

base_url = 'http://127.0.0.1:8000'
endpoint = '/predict'
url = base_url + endpoint


def predict(url="http://127.0.0.1:8000/", data=None, verbose=True):
    """
    Make a POST request to get the prediction from the RUL model
    
    Args:
        url (str): URL that the request is sent to.
        data (List[List[List[float]]]): data to upload, should be (batch_size, windows_size, features_number)

    Returns:
        requests.predicted_RUL, requests.batch_size: Response from the server.
    """
    
    response = requests.post(url, data)
    status_code = response.status_code

    if verbose:
        msg = "Prediction went good." if status_code == 200 else "There was an error when handling the request."
        print(msg)
    return response


def main():

    actual_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.dirname(actual_dir) + '/data'

    # load test data for api prediction
    data_batch = np.load(data_dir + '/data_test_api.npy')
    
    for data in data_batch:
        data_rsh = data[np.newaxis, ...]
        # response = predict(url, data_rsh)
        # print(f'Predicted RUL: {response.predicted_RUL}')


if __name__ == '__main__':
    main()