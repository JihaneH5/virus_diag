"""Trainer: handles training only. It never makes clinical diagnoses."""
import os
import pickle


class Trainer:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def train(self, X_train, y_train, verbose=True):
        X_scaled = self.preprocessor.fit_transform(X_train)
        self.model.fit(X_scaled, y_train, verbose=verbose)
        return self.model

    def save(self, directory, name):
        """Save the trained model and its preprocessor (needed at prediction time)."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "preprocessor": self.preprocessor}, f)
        return path

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return bundle["model"], bundle["preprocessor"]
