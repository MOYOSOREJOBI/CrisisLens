from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HawkesForecast:
    intensity: np.ndarray
    anomaly_score: np.ndarray


class SimpleHawkes:
    def __init__(self, alpha: float = 0.4, beta: float = 0.8, mu: float = 0.5) -> None:
        self.alpha = alpha
        self.beta = beta
        self.mu = mu

    def fit_predict(self, counts: pd.Series) -> HawkesForecast:
        values = counts.to_numpy(dtype=float)
        intensity = np.zeros_like(values)
        running = self.mu
        for i, value in enumerate(values):
            running = self.mu + self.alpha * value + np.exp(-self.beta) * running
            intensity[i] = running
        residual = values - intensity
        anomaly = (residual - residual.mean()) / (residual.std() + 1e-6)
        return HawkesForecast(intensity=intensity, anomaly_score=anomaly)
