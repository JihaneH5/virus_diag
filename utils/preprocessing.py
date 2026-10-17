"""Preprocessing: imputation, standardization, train/test split."""
import numpy as np


class Preprocessor:
    """Fills missing values with the median and standardizes (z-score).
    Fitted on the training set only, to avoid data leakage."""

    def __init__(self):
        self.median = None
        self.mean = None
        self.std = None

    def fit(self, X):
        self.median = np.nanmedian(X, axis=0)
        X = self._impute(X)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def _impute(self, X):
        X = np.array(X, dtype=float, copy=True)
        mask = np.isnan(X)
        if mask.any():
            X[mask] = np.take(self.median, np.where(mask)[1])
        return X

    def transform(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return (self._impute(X) - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    n_test = int(len(X) * test_size)
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
