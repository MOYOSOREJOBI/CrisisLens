from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from narrativepulse.config import load_config
from narrativepulse.utils.io import ensure_dir, read_json, read_parquet, write_json


def load_panel_from_config(config_path: str) -> tuple[object, pd.DataFrame]:
    cfg = load_config(config_path)
    panel = read_parquet(Path(cfg.data_dir) / "processed" / "panel.parquet")
    return cfg, panel


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.7, valid_ratio: float = 0.15) -> dict[str, pd.DataFrame]:
    dates = sorted(pd.to_datetime(df["date"]).unique())
    n_dates = len(dates)
    train_end = max(1, int(n_dates * train_ratio))
    valid_end = max(train_end + 1, int(n_dates * (train_ratio + valid_ratio)))
    train_dates = set(dates[:train_end])
    valid_dates = set(dates[train_end:valid_end])
    test_dates = set(dates[valid_end:])
    dated = df.copy()
    dated["_date"] = pd.to_datetime(dated["date"])
    return {
        "train": dated[dated["_date"].isin(train_dates)].drop(columns="_date").reset_index(drop=True),
        "valid": dated[dated["_date"].isin(valid_dates)].drop(columns="_date").reset_index(drop=True),
        "test": dated[dated["_date"].isin(test_dates)].drop(columns="_date").reset_index(drop=True),
    }


def feature_matrix(df: pd.DataFrame, exclude: set[str]) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return df[cols].to_numpy(dtype=np.float32), cols


def save_torch_payload(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    torch.save(payload, path)


def save_split_manifest(path: Path, splits: dict[str, pd.DataFrame]) -> None:
    payload = {
        name: {
            "rows": int(frame.shape[0]),
            "dates": [str(x) for x in sorted(frame["date"].unique())],
        }
        for name, frame in splits.items()
    }
    write_json(path, payload)
