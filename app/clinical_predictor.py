"""ClinicalPredictor: uses an already-trained model to diagnose a patient.

It works with ANY model that has a predict() method (duck typing / polymorphism)
and knows nothing about training."""
import numpy as np

FEATURES = [
    "age", "temperature", "cough", "fatigue", "loss_smell",
    "contact", "oxygen_saturation", "heart_rate", "wbc_count", "crp",
]


class ClinicalPredictor:
    def __init__(self, model, preprocessor=None, threshold=0.5):
        self.model = model                # already trained, needs predict()
        self.preprocessor = preprocessor  # optional: same scaling as in training
        self.threshold = threshold

    def _to_array(self, patient_data):
        if isinstance(patient_data, dict):
            patient_data = [patient_data.get(name, np.nan) for name in FEATURES]
        x = np.atleast_2d(np.asarray(patient_data, dtype=float))
        if self.preprocessor is not None:
            x = self.preprocessor.transform(x)
        return x

    def probability(self, patient_data):
        prediction = self.model.predict(self._to_array(patient_data))
        return float(np.ravel(prediction)[0])

    def diagnose(self, patient_data):
        return "Infecté" if self.probability(patient_data) >= self.threshold else "Sain"
