import math
from collections import Counter

class DecisionTree:
    def __init__(self, X, y, threshold=1.0, max_depth=None):
        """
        Inicializa e treina a árvore de decisão usando ganho de informação (ID3).
        X: lista de amostras, cada amostra é lista de atributos (categóricos).
        y: lista de rótulos (1 ou -1).
        threshold: ganho de informação mínimo para divisão.
        max_depth: profundidade máxima da árvore.
        """
        self.threshold = threshold
        self.max_depth = max_depth
        self.n_features = len(X[0]) if X else 0
        # Constroi a árvore
        self.tree = self._build_tree(X, y, depth=0, features=list(range(self.n_features)))

    def _entropy(self, labels):
        """Calcula a entropia de uma lista de rótulos."""
        total = len(labels)
        if total == 0:
            return 0.0
        counts = Counter(labels)
        ent = 0.0
        for count in counts.values():
            p = count / total
            ent -= p * math.log2(p)
        return ent

    def _majority(self, labels):
        """Retorna o rótulo mais frequente na lista de labels."""
        if not labels:
            return None
        return Counter(labels).most_common(1)[0][0]

    def _build_tree(self, X, y, depth, features):
        # Se vazio ou todas as etiquetas iguais, cria folha
        if not y or all(label == y[0] for label in y):
            return {'label': y[0] if y else None}
        # Se sem mais features ou atingiu profundidade máxima
        if not features or (self.max_depth is not None and depth >= self.max_depth):
            return {'label': self._majority(y)}

        current_entropy = self._entropy(y)
        best_gain = 0.0
        best_feat = None

        # Escolhe melhor feature pelo ganho de informação
        for feat in features:
            subsets = {}
            for xi, yi in zip(X, y):
                subsets.setdefault(xi[feat], []).append(yi)
            remainder = 0.0
            for subset_labels in subsets.values():
                remainder += (len(subset_labels) / len(y)) * self._entropy(subset_labels)
            gain = current_entropy - remainder
            if gain > best_gain:
                best_gain = gain
                best_feat = feat

        # Se ganho insuficiente, cria folha
        if best_gain < self.threshold or best_feat is None:
            return {'label': self._majority(y)}

        # Nó de decisão
        node = {
            'feature': best_feat,
            'children': {},
            'default': self._majority(y)
        }
        remaining_feats = [f for f in features if f != best_feat]

        # Cria subárvores para cada valor da feature
        splits = {}
        for xi, yi in zip(X, y):
            val = xi[best_feat]
            splits.setdefault(val, {'X': [], 'y': []})
            splits[val]['X'].append(xi)
            splits[val]['y'].append(yi)
        for val, data in splits.items():
            node['children'][val] = self._build_tree(data['X'], data['y'], depth + 1, remaining_feats)

        return node

    def predict(self, x):
        """Prediz rótulo para uma amostra x."""
        def recurse(node, xi):
            if 'label' in node:
                return node['label']
            feat = node['feature']
            val = xi[feat]
            child = node['children'].get(val)
            if child is None:
                return node['default']
            return recurse(child, xi)
        return recurse(self.tree, x)


def train_decision_tree(X, y, threshold=1.0, max_depth=None):
    """
    Treina e retorna uma DecisionTree configurada.
    """
    return DecisionTree(X, y, threshold=threshold, max_depth=max_depth)