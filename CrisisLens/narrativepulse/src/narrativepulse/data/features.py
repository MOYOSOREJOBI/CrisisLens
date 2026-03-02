from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=max(3, window // 3)).mean()
    std = series.rolling(window=window, min_periods=max(3, window // 3)).std().replace(0, np.nan)
    z = (series - mean) / std
    return z.fillna(0.0)


def build_attention_shock_index(pageviews: pd.Series, window: int) -> pd.Series:
    delta = np.log1p(pageviews).diff().fillna(0.0)
    return rolling_zscore(delta, window)


def build_risk_shock_index(vix: pd.Series, goldstein_mean: pd.Series, event_count: pd.Series, window: int) -> pd.Series:
    vix_delta_z = rolling_zscore(vix.diff().fillna(0.0), window)
    neg_goldstein_z = rolling_zscore((-goldstein_mean).clip(lower=0.0), window)
    event_z = rolling_zscore(event_count, window)
    return (vix_delta_z + neg_goldstein_z + event_z) / 3.0


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["date"])
    week = dt.dt.isocalendar().week.astype(int)
    month = dt.dt.month.astype(int)
    out["week_sin"] = np.sin(2 * np.pi * week / 52.0)
    out["week_cos"] = np.cos(2 * np.pi * week / 52.0)
    out["is_aw_season"] = month.isin([2, 3]).astype(int)
    out["is_ss_season"] = month.isin([9, 10]).astype(int)
    out["month"] = month
    return out
