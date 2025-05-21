import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

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
        for n in self.hidden_architecture:
            end_w = start_w + (input_size + 1) * n
            self.hidden_biases.append(w[start_w:start_w + n])
            self.hidden_weights.append(w[start_w + n:end_w].reshape(input_size, n))
            print(f"Camada oculta: Pesos = {self.hidden_weights[-1]}, Biases = {self.hidden_biases[-1]}")
            start_w = end_w
            input_size = n

        self.output_bias = w[start_w]
        self.output_weights = w[start_w + 1:]
        print(f"Camada de saída: Pesos = {self.output_weights}, Bias = {self.output_bias}")

    def forward(self, x):
        # Propagação direta
        a = np.array(x)
        print(f"Entrada inicial: {a}")
        for i, (weights, biases) in enumerate(zip(self.hidden_weights, self.hidden_biases)):
            z = np.dot(a, weights) + biases
            print(f"Camada {i + 1}: Z = {z}")
            a = self.hidden_activation(z)
            print(f"Camada {i + 1}: Ativação = {a}")

        # Camada de saída
        z = np.dot(a, self.output_weights) + self.output_bias
        print(f"Saída: Z = {z}")
        output = self.output_activation(z)
        print(f"Saída: Ativação = {output}")
        return output

def create_network_architecture(input_size):
    """
    Cria a arquitetura da rede neural com base no tamanho da entrada.
    """
    # Configuração padrão
    hidden_fn = lambda x: 1 / (1 + np.exp(-x))  # Função sigmoide
    output_fn = lambda x: 1 if x >= 0.5 else -1  # Converte a saída do sigmoide para -1 ou 1

    # Retorna uma instância da classe NeuralNetwork
    return NeuralNetwork(input_size, (), hidden_fn, output_fn)