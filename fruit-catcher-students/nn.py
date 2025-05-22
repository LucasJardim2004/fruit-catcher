import numpy as np

def sigmoid(x):
    """
    Função de ativação sigmoide.

    Args:
        x (float ou np.ndarray): Valor de entrada.

    Returns:
        float ou np.ndarray: Resultado da função sigmoide.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivada da função sigmoide.

    Args:
        x (float ou np.ndarray): Valor de entrada.

    Returns:
        float ou np.ndarray: Derivada da função sigmoide aplicada a x.
    """
    s = sigmoid(x)
    return s * (1 - s)

class NeuralNetwork:
    """
    Classe que representa uma rede neural feedforward com uma ou mais camadas ocultas.

    A rede suporta funções de ativação personalizadas para as camadas ocultas e de saída.

    Attributes:
        input_size (int): Número de neurónios na camada de entrada.
        hidden_architecture (tuple[int]): Lista com o número de neurónios por camada oculta.
        hidden_activation (Callable): Função de ativação para as camadas ocultas.
        output_activation (Callable): Função de ativação da camada de saída.
    """

    def __init__(self, input_size, hidden_architecture, hidden_activation=sigmoid, output_activation=sigmoid):
        """
        Inicializa a rede neural com a arquitetura definida.

        Args:
            input_size (int): Número de atributos de entrada.
            hidden_architecture (tuple[int]): Número de neurónios por camada oculta.
            hidden_activation (Callable, optional): Função de ativação para as camadas ocultas. Default é sigmoide.
            output_activation (Callable, optional): Função de ativação da camada de saída. Default é sigmoide.
        """
        self.input_size = input_size
        self.hidden_architecture = hidden_architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def compute_num_weights(self):
        """
        Calcula o número total de pesos necessários para a arquitetura atual da rede.

        Returns:
            int: Número total de pesos (incluindo biases).
        """
        num = 0
        in_size = self.input_size
        for h in self.hidden_architecture:
            num += (in_size + 1) * h  # +1 devido ao bias
            in_size = h
        num += in_size + 1  # saída final com bias
        return num

    def load_weights(self, weights):
        """
        Carrega os pesos da rede a partir de uma lista/array linear.

        Os pesos devem estar organizados de forma sequencial:
        - bias da camada,
        - pesos da camada (flattened),
        - bias de saída,
        - pesos de saída.

        Args:
            weights (list[float] ou np.ndarray): Pesos da rede em formato linear.
        """
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
        """
        Propagação direta dos dados pela rede.

        Args:
            x (list[float] ou np.ndarray): Vetor de entrada.

        Returns:
            float: Saída produzida pela rede após a propagação.
        """
        a = np.array(x)
        for W, b in zip(self.hidden_weights, self.hidden_biases):
            z = a.dot(W) + b
            a = self.hidden_activation(z)
        z_out = a.dot(self.output_weights) + self.output_bias
        return self.output_activation(z_out)

def create_network_architecture(input_size, mode="mlp"):
    """
    Cria uma rede neural com base no modo selecionado.

    Args:
        input_size (int): Número de atributos de entrada.
        mode (str, optional): Tipo de rede a criar ("perceptron" ou "mlp"). Default é "mlp".

    Returns:
        NeuralNetwork: Instância da rede neural.
    """
    if mode == "perceptron":
        return NeuralNetwork(input_size, (), sigmoid, lambda x: 1 if x >= 0 else -1)
    else:
        hidden_architecture = (10,)
        hidden_fn = sigmoid
        output_fn = lambda x: 1 if x >= 0 else -1
        return NeuralNetwork(input_size, hidden_architecture, hidden_fn, output_fn)