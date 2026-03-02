from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
        "auprc": float(average_precision_score(y_true, y_prob)) if y_true.sum() else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    rank = 0.0
    if len(y_true) > 1:
        rank = float(np.corrcoef(np.argsort(np.argsort(y_true)), np.argsort(np.argsort(y_pred)))[0, 1])
    return {"mae": float(mae), "rmse": float(rmse), "spearman_proxy": rank}


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= left) & (y_prob < right if right < 1 else y_prob <= right)
        if not np.any(mask):
            continue
        confidence = y_prob[mask].mean()
        accuracy = y_true[mask].mean()
        ece += abs(confidence - accuracy) * (mask.sum() / len(y_true))
    return float(ece)
