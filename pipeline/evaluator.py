"""Evaluator: measures a trained model on unseen data."""
from utils import metrics


class Evaluator:
    def __init__(self, model, preprocessor, threshold=0.5):
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = threshold

    def evaluate(self, X_test, y_test):
        prob = self.model.predict(self.preprocessor.transform(X_test))
        pred = (prob >= self.threshold).astype(int)
        tp, tn, fp, fn = metrics.confusion_matrix(y_test, pred)
        return {
            "accuracy": metrics.accuracy(y_test, pred),
            "precision": metrics.precision(y_test, pred),
            "recall": metrics.recall(y_test, pred),
            "specificity": metrics.specificity(y_test, pred),
            "f1": metrics.f1_score(y_test, pred),
            "auc": metrics.roc_auc(y_test, prob),
            "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        }

    @staticmethod
    def report(name, results):
        c = results["confusion"]
        print(f"\n=== {name} ===")
        print(f"  accuracy    : {results['accuracy']:.3f}")
        print(f"  precision   : {results['precision']:.3f}")
        print(f"  recall      : {results['recall']:.3f}  (infected patients detected)")
        print(f"  specificity : {results['specificity']:.3f}")
        print(f"  F1          : {results['f1']:.3f}")
        print(f"  AUC         : {results['auc']:.3f}")
        print(f"  confusion   : TP={c['TP']} TN={c['TN']} FP={c['FP']} FN={c['FN']}")
