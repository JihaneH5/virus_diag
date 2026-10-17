"""Abstract base class that every model must respect."""
import pickle
from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Contract: fit(X, y) trains, predict(X) returns probabilities in [0, 1].

    ClinicalPredictor only relies on predict(), so any subclass works."""

    @abstractmethod
    def fit(self, X, y, **kwargs):
        ...

    @abstractmethod
    def predict(self, X):
        """Return the probability of being infected for each row of X."""

    def predict_class(self, X, threshold=0.5):
        return (self.predict(X) >= threshold).astype(int)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_prob, eps=1e-12):
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))
