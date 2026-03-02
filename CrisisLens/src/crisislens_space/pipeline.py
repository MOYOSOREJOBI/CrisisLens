from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crisislens_space.fred import fetch_fred_series
from crisislens_space.gdelt import fetch_gdelt_timeline
from crisislens_space.io import load_model_predictions, save_json_report
from crisislens_space.plotting import compare_figure, timeline_figure, world_map_figure
from crisislens_space.wiki import fetch_pageviews


def _filter_frame(frame: pd.DataFrame, entity: str, start_date: str, end_date: str) -> pd.DataFrame:
    out = frame[(frame["entity"] == entity) & (frame["date"] >= pd.to_datetime(start_date)) & (frame["date"] <= pd.to_datetime(end_date))].copy()
    if out.empty:
        raise ValueError(f"No rows available for {entity} in the selected range.")
    return out.sort_values("date").reset_index(drop=True)


def _apply_live_adjustments(frame: pd.DataFrame, entity: str, start_date: str, end_date: str) -> pd.DataFrame:
    out = frame.copy()
    wiki = fetch_pageviews(entity.replace(" ", "_"), start_date, end_date)
    if wiki is not None and not wiki.empty:
        wiki["date"] = pd.to_datetime(wiki["date"])
        out = out.merge(wiki, on="date", how="left")
        out["pageviews"] = out["pageviews"].fillna(method="ffill").fillna(0.0)
        delta = np.log1p(out["pageviews"]).diff().fillna(0.0)
        out["attention_index"] = ((delta - delta.mean()) / (delta.std() + 1e-6)).clip(-5, 5)
    fred = fetch_fred_series("VIXCLS", start_date, end_date)
    if fred is not None and not fred.empty:
        fred["date"] = pd.to_datetime(fred["date"])
        out = out.merge(fred, on="date", how="left")
        out["VIXCLS"] = out["VIXCLS"].fillna(method="ffill").fillna(0.0)
        risk = out["VIXCLS"].diff().fillna(0.0)
        out["risk_index"] = ((risk - risk.mean()) / (risk.std() + 1e-6)).clip(-5, 5)
    out["shock_prob"] = (0.45 * out["shock_prob"] + 0.35 * _sigmoid(out["attention_index"]) + 0.20 * _sigmoid(out["risk_index"])).clip(0, 1)
    out["severity_pred"] = (out["severity_pred"] + 0.2 * out["attention_index"].clip(lower=0)).clip(lower=0)
    return out


def _sigmoid(values: pd.Series | np.ndarray) -> pd.Series:
    arr = np.asarray(values, dtype=float)
    return pd.Series(1.0 / (1.0 + np.exp(-arr)), index=getattr(values, "index", None))


def run_inference_panel(entity: str, start_date: str, end_date: str, horizon: int, fast_mode: bool, root_dir: Path) -> tuple[pd.DataFrame, dict]:
    frame, mode = load_model_predictions(root_dir)
    filtered = _filter_frame(frame, entity, start_date, end_date)
    if not fast_mode:
        filtered = _apply_live_adjustments(filtered, entity, start_date, end_date)
        mode = f"{mode}+live"
    filtered["severity_pred"] = filtered["severity_pred"] * (1.0 + 0.05 * (horizon - 1))
    filtered["intensity_pred"] = filtered["intensity_pred"] * (1.0 + 0.03 * (horizon - 1))
    metadata = {
        "mode": mode,
        "timeline_plot": timeline_figure(filtered, entity),
        "entity": entity,
        "horizon": horizon,
    }
    return filtered, metadata


def build_json_report(frame: pd.DataFrame, entity: str, horizon: int, metadata: dict) -> str:
    payload = {
        "entity": entity,
        "horizon_days": horizon,
        "mode": metadata["mode"],
        "latest": frame.sort_values("date").iloc[-1].to_dict(),
        "top_shock_days": (
            frame.sort_values(["shock_prob", "severity_pred"], ascending=False)
            .loc[:, ["date", "shock_prob", "severity_pred", "attention_index", "risk_index"]]
            .head(10)
            .to_dict("records")
        ),
    }
    output = save_json_report(payload, Path("cache") / f"report_{entity.replace(' ', '_')}_{horizon}.json")
    return str(output)


def build_compare_outputs(entities: list[str], start_date: str, end_date: str, horizon: int, fast_mode: bool, root_dir: Path):
    frames = []
    for entity in entities:
        frame, _ = run_inference_panel(entity, start_date, end_date, horizon, fast_mode, root_dir)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    table = (
        merged.groupby("entity")
        .agg(max_prob=("shock_prob", "max"), mean_prob=("shock_prob", "mean"), shock_days=("shock_prob", lambda s: int((s >= 0.7).sum())))
        .reset_index()
    )
    return compare_figure(merged), table


def build_world_map_outputs(start_date: str, end_date: str, horizon: int, fast_mode: bool, root_dir: Path):
    frame, mode = load_model_predictions(root_dir)
    filtered = frame[(frame["date"] >= pd.to_datetime(start_date)) & (frame["date"] <= pd.to_datetime(end_date))].copy()
    countries = filtered[filtered["entity"].str.contains("Ukraine|United States|France|Germany|China|Russia|Japan|Brazil|India|Canada", case=False, regex=True)]
    if countries.empty:
        countries = filtered.copy()
    latest = countries.sort_values("date").groupby("entity").tail(1).sort_values("shock_prob", ascending=False)
    latest["severity_pred"] = latest["severity_pred"] * (1.0 + 0.05 * (horizon - 1))
    return world_map_figure(latest), latest.loc[:, ["entity", "date", "shock_prob", "severity_pred"]].head(10)
