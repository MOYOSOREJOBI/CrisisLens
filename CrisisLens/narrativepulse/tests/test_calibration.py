from __future__ import annotations

import numpy as np

from narrativepulse.utils.calibration import TemperatureScaler
from narrativepulse.utils.conformal import conformal_coverage, conformal_interval


def test_temperature_scaler_and_conformal() -> None:
    logits = np.array([-2.0, -1.0, 0.5, 1.5, 2.0])
    labels = np.array([0, 0, 0, 1, 1])
    scaler = TemperatureScaler().fit(logits, labels)
    probs = scaler.transform(logits)
    assert probs.shape == logits.shape
    lower, upper, q = conformal_interval(np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 2.8]))
    coverage = conformal_coverage(np.array([1.0, 2.0, 3.0]), lower, upper)
    assert q >= 0.0
    assert 0.0 <= coverage <= 1.0
