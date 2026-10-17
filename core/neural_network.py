"""Small neural network (1 hidden layer, ReLU) written from scratch with numpy."""
import numpy as np

from core.model import BaseModel, sigmoid, binary_cross_entropy
from core.optimizer import Adam


class NeuralNetwork(BaseModel):
    def __init__(self, hidden=16, optimizer=None, epochs=150, batch_size=64,
                 l2=1e-4, seed=0):
        self.hidden = hidden
        self.optimizer = optimizer or Adam(lr=0.01)
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed
        self.W1 = self.b1 = self.W2 = self.b2 = None
        self.history = []

    def _init_params(self, d):
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / d), (d, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / self.hidden), (self.hidden, 1))
        self.b2 = np.zeros(1)

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        out = sigmoid(a1 @ self.W2 + self.b2).ravel()
        return z1, a1, out

    def fit(self, X, y, verbose=False, **kwargs):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self._init_params(d)
        self.history = []

        for epoch in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                Xb, yb = X[idx], y[idx]
                m = len(idx)

                z1, a1, out = self._forward(Xb)
                d_out = (out - yb).reshape(-1, 1) / m          # dL/dz2
                gW2 = a1.T @ d_out + self.l2 * self.W2
                gb2 = d_out.sum(axis=0)
                d_a1 = d_out @ self.W2.T
                d_z1 = d_a1 * (z1 > 0)
                gW1 = Xb.T @ d_z1 + self.l2 * self.W1
                gb1 = d_z1.sum(axis=0)

                self.optimizer.step([self.W1, self.b1, self.W2, self.b2],
                                    [gW1, gb1, gW2, gb2])

            loss = binary_cross_entropy(y, self.predict(X))
            self.history.append(loss)
            if verbose and (epoch + 1) % 50 == 0:
                print(f"  [nn]     epoch {epoch + 1:4d}  loss={loss:.4f}")
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        return self._forward(X)[2]
