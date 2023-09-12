import torch


class FeedforwardNeuralNetModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2,
                 hidden_dim3, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()

        # Linear functions
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = torch.nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = torch.nn.Linear(hidden_dim3, output_dim)

        # Non-linearity
        self.tanh = torch.nn.Tanh()

        # Activation
        self.ReLu = torch.nn.ReLU()

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        out = self.fc1(x)
        out = self.tanh(out)

        out = self.fc2(out)
        out = self.tanh(out)

        out = self.fc3(out)
        out = self.tanh(out)

        out = self.fc4(out)
        out = self.tanh(out)

        # Apply sigmoid
        out = self.sigmoid(out)

        return out


class ParameterFeedforwardNeuralNetModel(torch.nn.Module):
    def __init__(self, dimensions):
        if len(dimensions) < 2:
            print(f'Invalid dimension for NN: {dimensions}')
            return
        super(ParameterFeedforwardNeuralNetModel, self).__init__()

        # Linear functions
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(dimensions[i - 1], dimensions[i]) for i in range(1, len(dimensions))])

        # Non-linearity
        self.tanh = torch.nn.Tanh()

        # Activation
        self.ReLu = torch.nn.ReLU()

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        for layer in self.linears:
            x = layer(x)
            x = self.tanh(x)

        x = self.sigmoid(x)

        return x


class ShallowFeedForwardNeuralNetModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2,
                 hidden_dim3, output_dim):
        super(ShallowFeedForwardNeuralNetModel, self).__init__()

        # Linear functions
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.fc2 = torch.nn.Linear(hidden_dim1, output_dim)

        # Non-linearity
        self.tanh = torch.nn.Tanh()

        # Activation
        self.ReLu = torch.nn.ReLU()

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        out = self.fc1(x)
        out = self.tanh(out)

        out = self.fc2(out)
        out = self.tanh(out)

        # Apply sigmoid
        out = self.sigmoid(out)

        return out


def load_ffnn(path: str, neurons: list = None) -> FeedforwardNeuralNetModel:
    """
    Loads stored ffnn
    :param path: Path to stored ffnn
    :param neurons: List of modelparameters
    :return: Loaded ffnn
    """
    if neurons is None:
        input_dim = 47
        hidden_dim1 = 10
        hidden_dim2 = 3
        hidden_dim3 = 10
        output_dim = 1

        model_loaded = FeedforwardNeuralNetModel(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim)

    else:
        model_loaded = ParameterFeedforwardNeuralNetModel(neurons)

    model_loaded.load_state_dict(torch.load(path))
    model_loaded.eval()

    return model_loaded

