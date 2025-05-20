import numpy as np
from collections import defaultdict, Counter

# DECISION TREE

class DecisionTree:
    """
    A simple decision tree for categorical features.

    This implementation maps each unique combination of input features
    to the most common class label found in the training data.

    Attributes:
        max_depth (int): Maximum depth of the tree (not used in current implementation).
        threshold (float): Minimum threshold for a split (not used in current implementation).
        tree (dict): A mapping from feature tuples to predicted labels.
    """

    def __init__(self, X, y, threshold=1.0, max_depth=None):
        """
        Initializes and trains the decision tree.

        Args:
            X (list[list[str]]): Feature matrix with categorical values.
            y (list[int]): List of class labels.
            threshold (float, optional): Not used in current version.
            max_depth (int, optional): Not used in current version.
        """
        self.max_depth = max_depth
        self.threshold = threshold
        self.tree = {}
        self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """
        Builds the decision tree by mapping feature combinations to majority labels.

        Args:
            X (list[list[str]]): Feature matrix.
            y (list[int]): Corresponding labels.
            depth (int): Current depth of the tree (not used).
        """
        data = defaultdict(list)
        for features, label in zip(X, y):
            key = tuple(features)
            data[key].append(label)

        self.tree = {}
        for key, labels in data.items():
            counts = Counter(labels)
            majority_class, majority_count = counts.most_common(1)[0]
            self.tree[key] = majority_class

    def predict(self, x):
        """
        Predicts the class for a given input using the trained decision tree.

        Args:
            x (list[str]): Input feature vector.

        Returns:
            int or None: Predicted class label, or None if unseen combination.
        """
        key = tuple(x)
        return self.tree.get(key, None)


def train_decision_tree(X, y):
    """
    Trains a DecisionTree classifier on the given dataset.

    Args:
        X (list[list[str]]): Feature matrix with categorical data.
        y (list[int]): List of class labels.

    Returns:
        DecisionTree: A trained decision tree classifier.
    """
    return DecisionTree(X, y)