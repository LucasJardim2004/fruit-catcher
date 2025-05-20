import numpy as np
from collections import defaultdict, Counter

#DECISION TREE

class DecisionTree:

    def __init__(self, X, y, threshold=1.0, max_depth=None):
        self.max_depth = max_depth
        self.threshold = threshold
        self.tree = {}
        self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # Simples árvore de decisão para features categóricas: guarda contagem das classes para cada combinação
        data = defaultdict(list)
        for features, label in zip(X, y):
            key = tuple(features)
            data[key].append(label)

        # Para cada combinação de features, calcula a classe majoritária
        self.tree = {}
        for key, labels in data.items():
            counts = Counter(labels)
            majority_class, majority_count = counts.most_common(1)[0]
            self.tree[key] = majority_class

    def predict(self, x):
        key = tuple(x)
        return self.tree.get(key, None)  # Se não conhece, devolve None

def train_decision_tree(X, y):
    return DecisionTree(X, y)