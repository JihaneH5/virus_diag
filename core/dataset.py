"""Dataset: generation of fictitious patient data, loading and splitting."""
import csv
import os

import numpy as np

FEATURES = [
    "age",
    "temperature",
    "cough",
    "fatigue",
    "loss_smell",
    "contact",
    "oxygen_saturation",
    "heart_rate",
    "wbc_count",
    "crp",
]
TARGET = "infected"


def generate_fictive_data(n=1000, seed=42):
    """Create fictitious patients. The label follows a hidden noisy rule,
    so the models have something real to learn."""
    rng = np.random.default_rng(seed)

    infected = rng.random(n) < 0.4  # 40% infected

    age = rng.integers(18, 90, n).astype(float)
    temperature = np.where(infected, rng.normal(38.4, 0.7, n), rng.normal(36.9, 0.4, n))
    cough = (rng.random(n) < np.where(infected, 0.75, 0.15)).astype(float)
    fatigue = (rng.random(n) < np.where(infected, 0.70, 0.25)).astype(float)
    loss_smell = (rng.random(n) < np.where(infected, 0.45, 0.03)).astype(float)
    contact = (rng.random(n) < np.where(infected, 0.65, 0.20)).astype(float)
    oxygen = np.where(infected, rng.normal(93.5, 2.5, n), rng.normal(97.5, 1.2, n))
    heart_rate = np.where(infected, rng.normal(98, 12, n), rng.normal(76, 9, n))
    wbc = np.where(infected, rng.normal(10.5, 3.0, n), rng.normal(7.0, 1.8, n))
    crp = np.where(infected, rng.gamma(4.0, 12.0, n), rng.gamma(2.0, 2.5, n))

    # Label noise: flip 4% of labels (real data is never perfect)
    flip = rng.random(n) < 0.04
    label = np.where(flip, ~infected, infected).astype(int)

    X = np.column_stack(
        [age, temperature, cough, fatigue, loss_smell, contact,
         np.clip(oxygen, 80, 100), heart_rate, np.clip(wbc, 1, None), crp]
    )
    return X, label


class Dataset:
    """Loads / creates patient_data.csv and exposes X, y."""

    def __init__(self, path="data/patient_data.csv"):
        self.path = path
        self.X = None
        self.y = None
        self.feature_names = list(FEATURES)

    def create_csv(self, n=1000, seed=42):
        X, y = generate_fictive_data(n, seed)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(FEATURES + [TARGET])
            for row, label in zip(X, y):
                writer.writerow([round(float(v), 2) for v in row] + [int(label)])
        return self.path

    def load(self):
        if not os.path.exists(self.path):
            print(f"[dataset] {self.path} not found -> generating fictitious data")
            self.create_csv()
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.X = np.array([[float(r[c]) if r[c] != "" else np.nan for c in FEATURES]
                           for r in rows])
        self.y = np.array([int(r[TARGET]) for r in rows])
        return self.X, self.y

    def __len__(self):
        return 0 if self.y is None else len(self.y)
