from __future__ import annotations

import numpy as np


def conformal_interval(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray, float]:
    residuals = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    q = float(np.quantile(residuals, 1 - alpha))
    lower = np.asarray(y_pred) - q
    upper = np.asarray(y_pred) + q
    return lower, upper, q


def conformal_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    inside = (np.asarray(y_true) >= np.asarray(lower)) & (np.asarray(y_true) <= np.asarray(upper))
    return float(np.mean(inside))
