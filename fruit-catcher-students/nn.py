# nn.py
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class NeuralNetwork:
    def __init__(self, input_size, hidden_architecture, hidden_activation=sigmoid, output_activation=sigmoid):
        self.input_size = input_size
        self.hidden_architecture = hidden_architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def compute_num_weights(self):
        num = 0
        in_size = self.input_size
        for h in self.hidden_architecture:
            num += (in_size + 1) * h
            in_size = h
        num += in_size + 1
        return num

    def load_weights(self, weights):
        w = np.array(weights)
        self.hidden_weights = []
        self.hidden_biases  = []
        start = 0
        in_size = self.input_size
        for h in self.hidden_architecture:
            end = start + (in_size + 1) * h
            self.hidden_biases.append(w[start:start + h])
            self.hidden_weights.append(w[start + h:end].reshape(in_size, h))
            start = end
            in_size = h
        self.output_weights = w[start:start + in_size]
        self.output_bias    = w[start + in_size]

    def forward(self, x):
        a = np.array(x)
        for W, b in zip(self.hidden_weights, self.hidden_biases):
            z = a.dot(W) + b
            a = self.hidden_activation(z)
        z_out = a.dot(self.output_weights) + self.output_bias
        return self.output_activation(z_out)

def create_network_architecture(input_size):
    """
    Cria uma rede neural com:
      - 1 camada oculta de 10 neurónios (para aprender não-linearidades)
      - sigmóide nas camadas ocultas
      - threshold em 0 na saída para decidir -1/1
    """
    hidden_architecture = (10,)
    hidden_fn = sigmoid
    output_fn = lambda x: 1 if x >= 0 else -1
    return NeuralNetwork(input_size, hidden_architecture, hidden_fn, output_fn)