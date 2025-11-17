from typing import Dict

import numpy as np


def compute_confusion_elements(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
    y_pred_bin = (y_pred >= threshold).astype(np.uint8)
    y_true_bin = (y_true >= 0.5).astype(np.uint8)

    tp = np.logical_and(y_true_bin == 1, y_pred_bin == 1).sum()
    fp = np.logical_and(y_true_bin == 0, y_pred_bin == 1).sum()
    fn = np.logical_and(y_true_bin == 1, y_pred_bin == 0).sum()
    tn = np.logical_and(y_true_bin == 0, y_pred_bin == 0).sum()
    return tp, fp, fn, tn


def compute_metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    eps = 1e-7
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
    }
