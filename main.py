"""Full pipeline: data -> preprocessing -> training -> evaluation -> saved models -> diagnosis."""
from app.clinical_predictor import ClinicalPredictor
from core.dataset import Dataset
from core.logistic_regression import LogisticRegression
from core.neural_network import NeuralNetwork
from core.optimizer import Adam, SGD
from pipeline.evaluator import Evaluator
from pipeline.trainer import Trainer
from utils.preprocessing import Preprocessor, train_test_split


def main():
    # 1. Data (the CSV is created with fictitious patients if it does not exist)
    dataset = Dataset("data/patient_data.csv")
    X, y = dataset.load()
    print(f"Dataset: {len(dataset)} patients, {X.shape[1]} features, "
          f"{y.mean():.0%} infected")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # 2. Two interchangeable models
    models = {
        "logistic_regression": LogisticRegression(optimizer=SGD(lr=0.1), epochs=200),
        "neural_network": NeuralNetwork(hidden=16, optimizer=Adam(lr=0.01), epochs=150),
    }

    # 3. Train -> evaluate -> save each one
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        trainer = Trainer(model, Preprocessor())
        trainer.train(X_train, y_train, verbose=True)

        results = Evaluator(model, trainer.preprocessor).evaluate(X_test, y_test)
        Evaluator.report(name, results)

        path = trainer.save("models", name)
        print(f"  saved -> {path}")

    # 4. Diagnose with a saved model, without any training code
    print("\n--- Diagnosis with ClinicalPredictor ---")
    model, preprocessor = Trainer.load("models/neural_network.pkl")
    predictor = ClinicalPredictor(model, preprocessor)

    sick = {"age": 52, "temperature": 39.1, "cough": 1, "fatigue": 1, "loss_smell": 1,
            "contact": 1, "oxygen_saturation": 91, "heart_rate": 108, "wbc_count": 12.4,
            "crp": 62}
    healthy = {"age": 30, "temperature": 36.8, "cough": 0, "fatigue": 0, "loss_smell": 0,
               "contact": 0, "oxygen_saturation": 98, "heart_rate": 72, "wbc_count": 6.5,
               "crp": 2.1}
    for label, patient in [("Patient A", sick), ("Patient B", healthy)]:
        print(f"{label}: {predictor.diagnose(patient)} "
              f"(p = {predictor.probability(patient):.1%})")


if __name__ == "__main__":
    main()
