from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LabelModelResult:
    probabilities: np.ndarray
    pseudo_labels: np.ndarray
    coverage: dict[str, float]


def build_labeling_functions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    lf_event_spike = (df["gdelt_event_count_total"] > df["gdelt_event_count_total"].rolling(7, min_periods=3).mean().fillna(0) * 1.5).astype(int).to_numpy()
    lf_goldstein_tail = (df["gdelt_goldstein_min"] < df["gdelt_goldstein_min"].quantile(0.05)).astype(int).to_numpy()
    lf_attention_spike = (df["attention_shock_index"] > df["attention_shock_index"].quantile(0.95)).astype(int).to_numpy()
    vix = df["macro_VIXCLS"] if "macro_VIXCLS" in df.columns else pd.Series(np.zeros(len(df)))
    lf_macro_regime = (vix.diff().fillna(0.0) > vix.diff().fillna(0.0).quantile(0.9)).astype(int).to_numpy()
    return {
        "event_spike": lf_event_spike,
        "goldstein_tail": lf_goldstein_tail,
        "attention_spike": lf_attention_spike,
        "macro_regime": lf_macro_regime,
    }


def fit_label_model(df: pd.DataFrame) -> LabelModelResult:
    lfs = build_labeling_functions(df)
    lf_matrix = np.stack(list(lfs.values()), axis=1)
    probabilities = lf_matrix.mean(axis=1)
    pseudo_labels = (probabilities >= 0.5).astype(int)
    coverage = {name: float(values.mean()) for name, values in lfs.items()}
    return LabelModelResult(probabilities=probabilities, pseudo_labels=pseudo_labels, coverage=coverage)
