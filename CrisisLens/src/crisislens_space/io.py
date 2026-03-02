from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from crisislens_space.hub import fetch_model_snapshot
from crisislens_space.schema import REQUIRED_COLUMNS


def load_entities(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["entity", "date"]).reset_index(drop=True)


def load_demo_predictions(root_dir: Path) -> pd.DataFrame:
    path = root_dir / "data" / "demo_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing demo parquet: {path}")
    try:
        frame = pd.read_parquet(path)
    except Exception:
        frame = pd.read_csv(path)
    return _validate(frame)


def load_model_predictions(root_dir: Path) -> tuple[pd.DataFrame, str]:
    snapshot = fetch_model_snapshot()
    if snapshot is None:
        return load_demo_predictions(root_dir), "demo"
    candidates = [
        snapshot / "demo_predictions.parquet",
        snapshot / "predictions.parquet",
        snapshot / "artifacts" / "predictions.parquet",
    ]
    for path in candidates:
        if path.exists():
            try:
                frame = pd.read_parquet(path)
            except Exception:
                frame = pd.read_csv(path)
            return _validate(frame), "hub"
    return load_demo_predictions(root_dir), "demo"


def save_json_report(payload: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output_path
