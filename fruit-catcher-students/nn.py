import numpy as np

# NEURAL NETWORK

class NeuralNetwork:
    """
    A simple feedforward neural network with configurable hidden layers and activation functions.

    Attributes:
        input_size (int): Number of input features.
        hidden_architecture (tuple): Tuple with the number of neurons in each hidden layer.
        hidden_activation (function): Activation function for the hidden layers.
        output_activation (function): Activation function for the output layer.
    """

    def __init__(self, input_size, hidden_architecture, hidden_activation, output_activation):
        """
        Initializes the neural network with the given architecture and activation functions.

        Args:
            input_size (int): Number of input features.
            hidden_architecture (tuple): Hidden layer structure, e.g. (5, 2) means two layers with 5 and 2 neurons.
            hidden_activation (function): Activation function for hidden layers.
            output_activation (function): Activation function for the output layer.
        """
        self.input_size = input_size
        self.hidden_architecture = hidden_architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def compute_num_weights(self):
        """
        Computes the total number of weights (including biases) needed for the network.

        Returns:
            int: Total number of weights required.
        """
        total = 0
        input_size = self.input_size
        for n in self.hidden_architecture:
            total += (input_size + 1) * n  # weights + biases
            input_size = n
        total += (input_size + 1) * 1  # output layer (1 neuron)
        return total

    def load_weights(self, weights):
        """
        Loads the weights into the network and reshapes them according to the architecture.

        Args:
            weights (list[float]): A flat list of weights and biases to load into the network.
        """
        w = np.array(weights)

        self.hidden_weights = []
        self.hidden_biases = []

        start_w = 0
        input_size = self.input_size
        for n in self.hidden_architecture:
            end_w = start_w + (input_size + 1) * n
            self.hidden_biases.append(w[start_w:start_w+n])
            self.hidden_weights.append(w[start_w+n:end_w].reshape(input_size, n))
            start_w = end_w
            input_size = n

        self.output_bias = w[start_w]
        self.output_weights = w[start_w+1:]

    def forward(self, x):
        """
        Performs a forward pass through the neural network.

        Args:
            x (list[float] or np.ndarray): Input feature vector.

        Returns:
            float or int: Output of the network after applying all activations.
        """
        #print("Input to neural network:", x)
        x = np.array(x)
        for W, b in zip(self.hidden_weights, self.hidden_biases):
            z = x @ W + b
            x = self.hidden_activation(z)
        z = x @ self.output_weights + self.output_bias
        return self.output_activation(z)


def create_network_architecture(input_size):
    """
    Creates a predefined neural network architecture.

    Args:
        input_size (int): Number of input features.

    Returns:
        NeuralNetwork: A neural network instance with the specified architecture.
    """
    hidden_fn = lambda x: 1 / (1 + np.exp(-x))  # Sigmoid activation
    output_fn = lambda x: 1 if x > 0 else -1
    return NeuralNetwork(input_size, (10, 5), hidden_fn, output_fn)