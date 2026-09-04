"""Binary screening metrics.

Reports the numbers that matter for a rare positive class (cataract):
AUROC, AUPRC (average precision), and recall/precision for the cataract
class, plus the confusion-matrix counts. NOT accuracy - at ~10:1 imbalance
a 'predict everything normal' model scores ~91% accuracy and catches zero
cataracts, so accuracy is misleading.

Robust to a single-class batch: ranking metrics return NaN instead of
raising, so quick smoke tests never blow up.
"""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support, confusion_matrix)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    both = len(np.unique(y_true)) == 2
    auroc = roc_auc_score(y_true, y_prob) if both else float("nan")
    auprc = average_precision_score(y_true, y_prob) if both else float("nan")

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "auroc": auroc, "auprc": auprc,
        "recall_cataract": float(rec[1]),
        "precision_cataract": float(prec[1]),
        "f1_cataract": float(f1[1]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }