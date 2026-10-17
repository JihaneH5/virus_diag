"""Logistic regression written from scratch with numpy."""
import numpy as np

from core.model import BaseModel, sigmoid, binary_cross_entropy
from core.optimizer import SGD


class LogisticRegression(BaseModel):
    def __init__(self, optimizer=None, epochs=200, batch_size=64, l2=1e-3, seed=0):
        self.optimizer = optimizer or SGD(lr=0.1)
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed
        self.w = None
        self.b = None
        self.history = []

    def fit(self, X, y, verbose=False, **kwargs):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.w = rng.normal(0, 0.01, d)
        self.b = np.zeros(1)
        self.history = []

        for epoch in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                Xb, yb = X[idx], y[idx]
                err = sigmoid(Xb @ self.w + self.b) - yb
                grad_w = Xb.T @ err / len(idx) + self.l2 * self.w
                grad_b = np.array([err.mean()])
                self.optimizer.step([self.w, self.b], [grad_w, grad_b])

            loss = binary_cross_entropy(y, self.predict(X))
            self.history.append(loss)
            if verbose and (epoch + 1) % 50 == 0:
                print(f"  [logreg] epoch {epoch + 1:4d}  loss={loss:.4f}")
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        return sigmoid(X @ self.w + self.b)
