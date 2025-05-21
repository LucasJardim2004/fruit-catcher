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
        # Calcula o número total de pesos e vieses na rede
        num_weights = 0
        input_size = self.input_size

        for n in self.hidden_architecture:
            num_weights += (input_size + 1) * n  # +1 para os vieses
            input_size = n

        # Adiciona os pesos e vieses da camada de saída
        num_weights += input_size + 1
        return num_weights

    def load_weights(self, weights):
        w = np.array(weights)

        self.hidden_weights = []
        self.hidden_biases = []

        start_w = 0
        input_size = self.input_size
        # Carrega pesos e vieses de cada camada oculta
        for n in self.hidden_architecture:
            end_w = start_w + (input_size + 1) * n
            # Primeiro n valores são bias, os restantes são pesos
            self.hidden_biases.append(w[start_w:start_w + n])
            self.hidden_weights.append(
                w[start_w + n:end_w].reshape(input_size, n)
            )
            start_w = end_w
            input_size = n

        # Camada de saída: primeiro weights, depois bias
        self.output_weights = w[start_w:start_w + input_size]
        self.output_bias = w[start_w + input_size]

    def forward(self, x):
        # Propagação direta
        a = np.array(x)
        # Camadas ocultas
        for weights, biases in zip(self.hidden_weights, self.hidden_biases):
            z = np.dot(a, weights) + biases
            a = self.hidden_activation(z)

        # Camada de saída
        z = np.dot(a, self.output_weights) + self.output_bias
        output = self.output_activation(z)
        return output


def create_network_architecture(input_size):
    """
    Cria a arquitetura da rede neural com base no tamanho da entrada.
    """
    # Configuração padrão
    hidden_fn = sigmoid
    # Limiar no valor linear para equilibrar -1/+1
    output_fn = lambda x: 1 if x >= 0 else -1

    return NeuralNetwork(input_size, (10,5), hidden_fn, output_fn)