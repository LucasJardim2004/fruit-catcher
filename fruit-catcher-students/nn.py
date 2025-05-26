import numpy as np  # Importing NumPy library for numerical computations

# --- Activation Functions ---

def sigmoid(x):
    """
    Sigmoid activation function.

    Args:
        x (float or np.ndarray): Input value(s).

    Returns:
        float or np.ndarray: Output after applying sigmoid.
    """
    return 1 / (1 + np.exp(-x))  # Applies sigmoid formula element-wise

def sigmoid_derivative(x):
    """
    Derivative of the sigmoid function.

    Args:
        x (float or np.ndarray): Input value(s).

    Returns:
        float or np.ndarray: Derivative of the sigmoid applied to x.
    """
    s = sigmoid(x)              # Compute sigmoid first
    return s * (1 - s)          # Derivative formula of sigmoid

# --- Neural Network Class ---

class NeuralNetwork:
    """
    Class representing a feedforward neural network with optional hidden layers.

    Attributes:
        input_size (int): Number of input neurons.
        hidden_architecture (tuple[int]): Number of neurons in each hidden layer.
        hidden_activation (Callable): Activation function for hidden layers.
        output_activation (Callable): Activation function for the output layer.
    """

    def __init__(self, input_size, hidden_architecture, hidden_activation=sigmoid, output_activation=sigmoid):
        """
        Initializes the neural network with the given architecture.

        Args:
            input_size (int): Number of input features.
            hidden_architecture (tuple[int]): Size of each hidden layer.
            hidden_activation (Callable): Activation for hidden layers (default: sigmoid).
            output_activation (Callable): Activation for output layer (default: sigmoid).
        """
        self.input_size = input_size                      # Store input size
        self.hidden_architecture = hidden_architecture    # Store hidden layer sizes
        self.hidden_activation = hidden_activation        # Store hidden activation function
        self.output_activation = output_activation        # Store output activation function

    def compute_num_weights(self):
        """
        Computes total number of weights (including biases) required for this architecture.

        Returns:
            int: Total number of weights.
        """
        num = 0
        in_size = self.input_size
        for h in self.hidden_architecture:
            num += (in_size + 1) * h     # +1 for the bias per neuron
            in_size = h                  # Set input size for next layer
        num += in_size + 1               # Output layer weights + 1 bias
        return num

    def load_weights(self, weights):
        """
        Loads the network weights from a flat list/array.

        Args:
            weights (list[float] or np.ndarray): Flat list of weights to assign to layers.
        """
        w = np.array(weights)            # Convert input to numpy array
        self.hidden_weights = []         # List of weight matrices per hidden layer
        self.hidden_biases  = []         # List of bias vectors per hidden layer
        start = 0
        in_size = self.input_size

        for h in self.hidden_architecture:
            end = start + (in_size + 1) * h                        # Total weights + biases
            self.hidden_biases.append(w[start:start + h])         # First 'h' values are biases
            self.hidden_weights.append(w[start + h:end].reshape(in_size, h))  # Reshape rest into weight matrix
            start = end                                           # Update start for next layer
            in_size = h                                           # Update input size

        self.output_weights = w[start:start + in_size]            # Output weights (1 per input)
        self.output_bias    = w[start + in_size]                  # Output bias

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Args:
            x (list[float] or np.ndarray): Input vector.

        Returns:
            float: Output value after feedforward.
        """
        a = np.array(x)                                           # Convert input to numpy array

        for W, b in zip(self.hidden_weights, self.hidden_biases):
            z = a.dot(W) + b                                      # Linear combination (z = Wx + b)
            a = self.hidden_activation(z)                         # Apply activation function

        z_out = a.dot(self.output_weights) + self.output_bias     # Final output layer computation
        return self.output_activation(z_out)                      # Apply output activation

# --- Network Architecture Builder ---

def create_network_architecture(input_size, mode="mlp"):
    """
    Creates a neural network based on the selected mode.

    Args:
        input_size (int): Number of input features.
        mode (str, optional): "perceptron" or "mlp". Default is "mlp".

    Returns:
        NeuralNetwork: An instance of the configured network.
    """
    if mode == "perceptron":
        return NeuralNetwork(input_size, (), sigmoid, lambda x: 1 if x >= 0 else -1)
        # Simple perceptron: no hidden layers, output is 1 or -1 (step function)
    else:
        hidden_architecture = (10,)      # 1 hidden layer with 10 neurons
        hidden_fn = sigmoid              # Use sigmoid activation in hidden layer
        output_fn = lambda x: 1 if x >= 0 else -1  # Output decision: 1 or -1
        return NeuralNetwork(input_size, hidden_architecture, hidden_fn, output_fn)