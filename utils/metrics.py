"""Classification metrics, implemented with numpy."""
import numpy as np


def confusion_matrix(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred):
    tp, _, fp, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall(y_true, y_pred):
    """Sensitivity: share of infected patients that were detected."""
    tp, _, _, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) else 0.0


def specificity(y_true, y_pred):
    _, tn, fp, _ = confusion_matrix(y_true, y_pred)
    return tn / (tn + fp) if (tn + fp) else 0.0


def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def roc_auc(y_true, y_prob):
    """AUC via the rank statistic (Mann-Whitney U)."""
    order = np.argsort(y_prob)
    ranks = np.empty(len(y_prob))
    ranks[order] = np.arange(1, len(y_prob) + 1)
    n_pos = int(np.sum(y_true == 1))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
