import math
from collections import Counter

class DecisionTree:
    """
    Implementação simples de uma árvore de decisão para classificação.

    A árvore utiliza entropia para escolher os atributos e aplica uma profundidade
    máxima para evitar overfitting. Funciona bem para datasets com atributos categóricos.

    Attributes:
        threshold (float): Ganho mínimo de informação para continuar a dividir nós.
        max_depth (int): Profundidade máxima da árvore.
        n_features (int): Número de atributos em cada instância.
        tree (dict): Estrutura de árvore construída.
    """

    def __init__(self, X, y, threshold=0.01, max_depth=3):
        """
        Inicializa e treina a árvore de decisão com os dados fornecidos.

        Args:
            X (list[list[Any]]): Lista de exemplos de treino (atributos).
            y (list[Any]): Lista de rótulos correspondentes a cada exemplo.
            threshold (float, optional): Ganho mínimo de informação para dividir. Default é 0.01.
            max_depth (int, optional): Profundidade máxima permitida. Default é 3.
        """
        # Inicializa os parâmetros e treina a árvore
        self.threshold = threshold  # Define o ganho mínimo necessário para dividir um nó
        self.max_depth = max_depth  # Define a profundidade máxima da árvore
        self.n_features = len(X[0]) if X else 0  # Número de atributos por instância
        self.tree = self._build_tree(X, y, 0, list(range(self.n_features)))  # Constrói a árvore recursivamente

    def _entropy(self, labels):
        """
        Calcula a entropia de uma lista de rótulos.

        Args:
            labels (list[Any]): Rótulos.

        Returns:
            float: Valor da entropia.
        """
        # Calcula a entropia (grau de desorganização) dos rótulos
        total = len(labels)  # Número total de rótulos
        if total == 0:
            return 0.0  # Evita divisão por zero
        counts = Counter(labels)  # Conta a frequência de cada rótulo
        ent = 0.0
        for c in counts.values():  # Para cada classe
            p = c / total  # Calcula a probabilidade da classe
            ent -= p * math.log2(p)  # Soma -p*log2(p) à entropia
        return ent  # Retorna a entropia total

    def _majority(self, labels):
        """
        Devolve o rótulo mais comum na lista.

        Args:
            labels (list[Any]): Lista de rótulos.

        Returns:
            Any: Rótulo com maior frequência.
        """
        # Devolve o rótulo mais comum
        if not labels:
            return None  # Se a lista estiver vazia
        return Counter(labels).most_common(1)[0][0]  # Retorna o rótulo com maior frequência

    def _build_tree(self, X, y, depth, features):
        """
        Constrói recursivamente a árvore de decisão.

        Args:
            X (list[list[Any]]): Dados de treino.
            y (list[Any]): Rótulos de treino.
            depth (int): Profundidade atual da árvore.
            features (list[int]): Índices dos atributos disponíveis.

        Returns:
            dict: Representação da árvore (nó).
        """
        # Constrói recursivamente a árvore de decisão
        if not y or all(lbl == y[0] for lbl in y):
            return {'label': y[0] if y else None}  # Se todos os rótulos forem iguais, retorna um nó folha
        if not features or depth >= self.max_depth:
            return {'label': self._majority(y)}  # Se não há mais atributos ou atingiu profundidade máxima

        current_ent = self._entropy(y)  # Entropia atual dos dados
        best_gain = 0.0  # Melhor ganho de informação inicial
        best_feat = None  # Melhor atributo para dividir

        for feat in features:  # Para cada atributo disponível
            splits = {}  # Dicionário para dividir os dados
            for xi, yi in zip(X, y):  # Para cada exemplo e seu rótulo
                splits.setdefault(xi[feat], []).append(yi)  # Agrupa os rótulos por valor do atributo
            rem = sum(
                (len(sub) / len(y)) * self._entropy(sub) for sub in splits.values())  # Entropia ponderada das divisões
            gain = current_ent - rem  # Ganho de informação ao dividir por esse atributo
            if gain > best_gain:  # Se este ganho é o melhor até agora
                best_gain = gain
                best_feat = feat  # Atualiza o melhor atributo

        if best_gain < self.threshold or best_feat is None:
            return {'label': self._majority(y)}  # Se ganho for pequeno, retorna nó folha com rótulo mais comum

        node = {'feature': best_feat, 'children': {}, 'default': self._majority(y)}  # Cria o nó com o melhor atributo
        rem_feats = [f for f in features if f != best_feat]  # Remove o atributo usado da lista

        splits = {}  # Reorganiza os dados para a próxima divisão
        for xi, yi in zip(X, y):
            val = xi[best_feat]  # Valor do atributo para esse exemplo
            splits.setdefault(val, {'X': [], 'y': []})  # Inicializa caso necessário
            splits[val]['X'].append(xi)  # Adiciona o exemplo à divisão
            splits[val]['y'].append(yi)  # Adiciona o rótulo à divisão

        for val, data in splits.items():
            # Constrói recursivamente as sub-árvores para cada valor do atributo
            node['children'][val] = self._build_tree(data['X'], data['y'], depth + 1, rem_feats)

        return node  # Retorna o nó atual (subárvore)

    def predict(self, x):
        """
        Classifica uma nova instância utilizando a árvore treinada.

        Args:
            x (list[Any]): Exemplo a ser classificado.

        Returns:
            Any: Rótulo previsto.

        Raises:
            ValueError: Se o número de atributos for diferente do esperado.
        """
        # Classifica um novo exemplo usando a árvore treinada
        if len(x) != self.n_features:
            raise ValueError(
                f"Expected input with {self.n_features} features, got {len(x)}")  # Valida o número de atributos

        def walk(node, xi):
            if 'label' in node:
                return node['label']  # Se for um nó folha, retorna o rótulo
            feat = node['feature']  # Obtém o índice do atributo para dividir
            child = node['children'].get(xi[feat])  # Escolhe a subárvore com base no valor do atributo
            if child is None:
                return node['default']  # Se não houver ramo, retorna o valor padrão
            return walk(child, xi)  # Continua a percorrer a árvore

        return walk(self.tree, x)  # Inicia a predição a partir da raiz


def train_decision_tree(X, y, threshold=0.01, max_depth=3):
    """
    Função auxiliar para treinar uma árvore de decisão.

    Args:
        X (list[list[Any]]): Dados de treino.
        y (list[Any]): Rótulos de treino.
        threshold (float, optional): Ganho mínimo de informação. Default é 0.01.
        max_depth (int, optional): Profundidade máxima da árvore. Default é 3.

    Returns:
        DecisionTree: Instância da árvore treinada.
    """
    # Função auxiliar para treinar uma árvore de decisão
    return DecisionTree(X, y, threshold=threshold, max_depth=max_depth)  # Retorna uma instância da árvore treinada