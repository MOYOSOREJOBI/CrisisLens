from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


@dataclass
class StackerOutputs:
    shock_probability: np.ndarray
    severity: np.ndarray


class MetaStacker:
    def __init__(self) -> None:
        self.classifier = LogisticRegression(max_iter=500, random_state=42)
        self.regressor = LinearRegression()

    def fit(self, x: np.ndarray, y_class: np.ndarray, y_reg: np.ndarray) -> "MetaStacker":
        self.classifier.fit(x, y_class)
        self.regressor.fit(x, y_reg)
        return self

    def predict(self, x: np.ndarray) -> StackerOutputs:
        prob = self.classifier.predict_proba(x)[:, 1]
        reg = self.regressor.predict(x)
        return StackerOutputs(shock_probability=prob, severity=reg)
