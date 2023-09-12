import time
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing

# Local Import
import models


def train_dffnn(key: int, train_frame_dict: pd.DataFrame, test_frame_dict: pd.DataFrame, dropfeatures: list = None,
                parameter: list = None) -> object:
    """
    Trains a dffnn and stores it together with traintimes, labels, and loss
    :param key: ID of device
    :param train_frame_dict: Testset
    :param test_frame_dict: Trainingset
    :param dropfeatures: Features to drop for the training
    :param parameter: List of modelparameters
    :return:
    """

    """
    Step 1: Load and prepare the dataset
    """

    # Remove labels and optional defined features
    if dropfeatures is None:
        dropfeatures = []
    train_dataset = train_frame_dict.loc[:, ~train_frame_dict.columns.isin([*['attack_cat', 'Label'], *dropfeatures])]
    label = train_frame_dict.loc[:, train_frame_dict.columns.isin(['Label'])].reset_index(drop=True)

    test_dataset = test_frame_dict.loc[:, ~test_frame_dict.columns.isin([*['attack_cat', 'Label'], *dropfeatures])]
    testlabel = test_frame_dict.loc[:, test_frame_dict.columns.isin(['Label'])].reset_index(drop=True)

    # Normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    train_dataset = min_max_scaler.fit_transform(train_dataset)
    test_dataset = min_max_scaler.fit_transform(test_dataset)

    train_dataset = torch.from_numpy(train_dataset.astype(np.float32))
    label = torch.from_numpy(label.to_numpy().astype(np.float32))

    test_dataset = torch.from_numpy(test_dataset.astype(np.float32))
    testlabel = torch.from_numpy(testlabel.to_numpy().astype(np.float32))

    """
    Step 2: Instantiate the parameters
    """

    num_epochs = 100

    # If no parameters are given use default
    if parameter is None:
        input_dim = 47
        hidden_dim1 = 10
        hidden_dim2 = 3
        hidden_dim3 = 10
        output_dim = 1

        model = models.FeedforwardNeuralNetModel(input_dim, hidden_dim1, hidden_dim2,
                                                 hidden_dim3, output_dim)

    # Parameterized
    else:
        print(f'Parameterized model with dimensions: {parameter}')
        model = models.ParameterFeedforwardNeuralNetModel(parameter)

    # Instantiate the MSE Loss
    criterion = torch.nn.MSELoss()

    learning_rate = 0.002

    # Instantiate the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.4, weight_decay=1e-6)

    """
    Step 3: Train the model
    """

    iteration = 0
    losses = []

    # Measure runtime
    starttime = time.time()

    for epoch in range(num_epochs):
        startepoch = time.time()
        print(f'Epoch {epoch}')

        for i, features in enumerate(train_dataset):
            # Clear gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            outputs = model(features)

            # Calculate loss
            loss = criterion(outputs, label[i])

            # Get gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            iteration += 1

            # Use detach to save memory
            losses.append(loss.detach())

        endepoch = time.time()
        print(
            'Iteration: {}. Loss: {}. Time {}'.format(iteration, loss.item(), endepoch - startepoch))
        print(f'Estimated time left {(num_epochs - epoch) * (endepoch - startepoch)}')

    endtime = time.time()

    # Calculate Accuracy
    correct = 0
    total = 0
    lossy = []
    # Iterate through test dataset
    for i, features in enumerate(test_dataset):
        outputs = model(features)
        lossy.append(outputs.detach().item())
        total += 1
        # Total correct predictions
        if testlabel[[i]] == outputs:
            correct += 1
    print(f'Correct: {correct}, Total: {total}')

    """
    Step 4: Save the model, labels, loss, and times
    """

    torch.save(model.state_dict(), f'models/{key}')

    np.savetxt(f'models/testlabel-{key}.csl', test_frame_dict, delimiter=",", fmt='%s')
    np.savetxt(f'models/loss-{key}.csv', lossy, delimiter=",", fmt='%s')

    with open(f'models/traintime-{key}.txt', 'a') as the_file:
        the_file.write(str(endtime - starttime))

    del model, lossy, train_dataset, label, test_dataset, testlabel, min_max_scaler, criterion, optimizer, loss, losses
    return


def do_training(key: int, train_frame_dict: pd.DataFrame, test_frame_dict: pd.DataFrame, parameter: list = None) -> int:
    """
    Checks if at least two samples are available for training and calls the training
    :rtype: int
    :param key: ID of device
    :param train_frame_dict: Testset
    :param test_frame_dict: Trainingset
    :param parameter: List of modelparameters
    :return: 1 for error, 0 for success
    """

    if len(train_frame_dict) < 2 or len(test_frame_dict) < 2:
        return 1
    print(f'Device:{key}')
    train_dffnn(key, train_frame_dict, test_frame_dict, [], parameter)

    return 0
