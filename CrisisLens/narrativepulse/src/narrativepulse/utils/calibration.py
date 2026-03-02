from __future__ import annotations

import numpy as np


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=float).reshape(-1)
        labels = np.asarray(labels, dtype=float).reshape(-1)
        best_t = 1.0
        best_loss = float("inf")
        for t in np.linspace(0.5, 5.0, 46):
            probs = 1.0 / (1.0 + np.exp(-(logits / t)))
            probs = np.clip(probs, 1e-6, 1 - 1e-6)
            loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
            if loss < best_loss:
                best_loss = loss
                best_t = float(t)
        self.temperature = best_t
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=float)
        return 1.0 / (1.0 + np.exp(-(logits / self.temperature)))
