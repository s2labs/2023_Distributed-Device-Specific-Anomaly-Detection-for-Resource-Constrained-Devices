import ipaddress
import time

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing

# Local Import
import models
import preprocess

__path__ = "demo"  # Path to model and data folder


def process_dataset(layer: str, neurons: str, training_ratio: str, device: str, remote: bool = False,
                    glob: bool = False) -> (float, pd.DataFrame):
    """
    Processes dataset with pre-trained model specified by parameters
    :param layer: Amount of layers
    :param neurons: Amount of neurons of first hidden layer
    :param training_ratio: Ratio of training data
    :param device: Device ID for device to be predicted
    :param remote: True if model for a remote device is used, else use Ad model for local device
    :param glob: True: Load global model
    :return: Duration of processing, results, and losses
    """

    list_3 = [[47, 5, 2, 5, 1], [47, 10, 3, 10, 1], [47, 20, 6, 20, 1]]
    list_1 = [[47, 2, 1], [47, 3, 1], [47, 6, 1]]

    dict_translator = {'2': 0, '5': 0, '3': 1, '10': 1, '6': 2, '20': 2}

    params_dict = {3: list_3, 1: list_1}

    if device == 'default':
        device = int(ipaddress.IPv4Address("59.166.0.2"))

    # Load Model
    try:
        parameters_selected = params_dict[int(layer)][dict_translator[neurons]]

        print(f'{layer}/{parameters_selected[1]}/{training_ratio}/{device}')
        if remote:
            if glob:
                model = models.load_ffnn(
                    f'{__path__}/models/local/{layer}/{parameters_selected[1]}/{training_ratio}/models/0',
                    parameters_selected)

            else:
                model = models.load_ffnn(
                    f'{__path__}/models/remote/{layer}/{parameters_selected[1]}/{training_ratio}/models/{device}',
                    parameters_selected)
            # Select data
            dataset_raw = preprocess.get_dataset(f"{__path__}/data/remote/testlabel-{device}.csl")
        else:
            model = models.load_ffnn(
                f'{__path__}/models/local/{layer}/{parameters_selected[1]}/{training_ratio}/models/0',
                parameters_selected)
            # Select data
            dataset_raw = preprocess.get_dataset(f"{__path__}/data/local/testlabel-{device}.csl")

    except Exception as e:
        print(f'Error: {e}')
        return "Parameter error, unable to load model"

    test_dataset = dataset_raw.loc[:, ~dataset_raw.columns.isin([*['attack_cat', 'Label']])]

    # print(test_dataset)

    min_max_scaler = preprocessing.MinMaxScaler()
    test_dataset = min_max_scaler.fit_transform(test_dataset)

    test_dataset = torch.from_numpy(test_dataset.astype(np.float32))
    results = []

    # Process data
    start_time = time.time()
    for i, features in enumerate(test_dataset):
        outputs = model(features)
        results.append(outputs.detach().item())

    end_time = time.time()
    duration = end_time - start_time

    # return results and processing time
    return duration, pd.DataFrame(results, columns=['losses'], dtype=float)
