# dt.py
import math
from collections import Counter

class DecisionTree:
    def __init__(self, X, y, threshold=0.01, max_depth=3):
        """
        Árvores de decisão baseadas em combinações de atributos como 'name' + 'color'.
        'color' e 'format' isoladamente não distinguem bem bombas de frutas.
        A profundidade 3 reduz overfitting no dataset pequeno atual.
        """
        self.threshold = threshold
        self.max_depth = max_depth
        self.n_features = len(X[0]) if X else 0
        self.tree = self._build_tree(X, y, 0, list(range(self.n_features)))

    def _entropy(self, labels):
        total = len(labels)
        if total == 0:
            return 0.0
        counts = Counter(labels)
        ent = 0.0
        for c in counts.values():
            p = c / total
            ent -= p * math.log2(p)
        return ent

    def _majority(self, labels):
        if not labels:
            return None
        return Counter(labels).most_common(1)[0][0]

    def _build_tree(self, X, y, depth, features):
        if not y or all(lbl == y[0] for lbl in y):
            return {'label': y[0] if y else None}
        if not features or depth >= self.max_depth:
            return {'label': self._majority(y)}

        current_ent = self._entropy(y)
        best_gain = 0.0
        best_feat = None
        for feat in features:
            splits = {}
            for xi, yi in zip(X, y):
                splits.setdefault(xi[feat], []).append(yi)
            rem = sum((len(sub) / len(y)) * self._entropy(sub) for sub in splits.values())
            gain = current_ent - rem
            if gain > best_gain:
                best_gain = gain
                best_feat = feat

        if best_gain < self.threshold or best_feat is None:
            return {'label': self._majority(y)}

        node = {'feature': best_feat, 'children': {}, 'default': self._majority(y)}
        rem_feats = [f for f in features if f != best_feat]
        splits = {}
        for xi, yi in zip(X, y):
            val = xi[best_feat]
            splits.setdefault(val, {'X': [], 'y': []})
            splits[val]['X'].append(xi)
            splits[val]['y'].append(yi)
        for val, data in splits.items():
            node['children'][val] = self._build_tree(data['X'], data['y'], depth + 1, rem_feats)
        return node

    def predict(self, x):
        if len(x) != self.n_features:
            raise ValueError(f"Expected input with {self.n_features} features, got {len(x)}")
        def walk(node, xi):
            if 'label' in node:
                return node['label']
            feat = node['feature']
            child = node['children'].get(xi[feat])
            if child is None:
                return node['default']
            return walk(child, xi)
        return walk(self.tree, x)

def train_decision_tree(X, y, threshold=0.01, max_depth=3):
    return DecisionTree(X, y, threshold=threshold, max_depth=max_depth)