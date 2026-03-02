from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from narrativepulse.data.features import (
    add_time_features,
    build_attention_shock_index,
    build_risk_shock_index,
)
from narrativepulse.utils.io import ensure_dir, write_parquet


@dataclass
class PanelBuildResult:
    panel: pd.DataFrame
    feature_columns: list[str]


def _merge_daily_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=["entity", "date"], how="outer")
    out = out.sort_values(["entity", "date"]).reset_index(drop=True)
    return out


def build_daily_panel(
    gdelt_df: pd.DataFrame,
    wiki_df: pd.DataFrame,
    fred_df: pd.DataFrame,
    shock_quantile: float,
    rolling_window: int,
) -> PanelBuildResult:
    if gdelt_df.empty or wiki_df.empty:
        raise ValueError("GDELT and Wikipedia inputs must be non-empty.")

    fred = fred_df.copy()
    if not fred.empty and "entity" not in fred.columns:
        entities = sorted(set(gdelt_df["entity"]))
        fred = fred.assign(key=1).merge(pd.DataFrame({"entity": entities, "key": 1}), on="key").drop(columns="key")

    frames = [gdelt_df.copy(), wiki_df.copy()]
    if not fred.empty:
        frames.append(fred)

    panel = _merge_daily_frames(frames).fillna(0.0)
    panel = add_time_features(panel)
    panel["attention_shock_index"] = panel.groupby("entity")["wikipedia_pageviews"].transform(
        lambda s: build_attention_shock_index(s, rolling_window)
    )
    vix = panel["macro_VIXCLS"] if "macro_VIXCLS" in panel.columns else pd.Series(np.zeros(len(panel)))
    panel["risk_shock_index"] = build_risk_shock_index(
        vix=vix,
        goldstein_mean=panel["gdelt_goldstein_mean"],
        event_count=panel["gdelt_event_count_total"],
        window=rolling_window,
    )
    threshold = np.nanquantile(
        np.maximum(panel["attention_shock_index"].values, panel["risk_shock_index"].values),
        shock_quantile,
    )
    panel["shock_label"] = (
        (panel["attention_shock_index"] > threshold) | (panel["risk_shock_index"] > threshold)
    ).astype(int)
    panel["severity_t_plus_1"] = panel.groupby("entity")[
        ["attention_shock_index", "risk_shock_index"]
    ].transform(lambda x: x.max(axis=1)).groupby(panel["entity"]).shift(-1).fillna(0.0)
    panel["severity_t_plus_3"] = panel.groupby("entity")["severity_t_plus_1"].shift(-2).fillna(0.0)
    panel["severity_t_plus_7"] = panel.groupby("entity")["severity_t_plus_1"].shift(-6).fillna(0.0)
    panel["event_intensity_t_plus_1"] = panel.groupby("entity")["gdelt_event_count_total"].shift(-1).fillna(0.0)

    feature_columns = [
        c
        for c in panel.columns
        if c
        not in {
            "entity",
            "date",
            "shock_label",
            "severity_t_plus_1",
            "severity_t_plus_3",
            "severity_t_plus_7",
            "event_intensity_t_plus_1",
        }
    ]
    return PanelBuildResult(panel=panel, feature_columns=feature_columns)


def save_panel(result: PanelBuildResult, output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    write_parquet(result.panel, output_path)
