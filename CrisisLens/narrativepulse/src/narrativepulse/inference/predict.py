from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from narrativepulse.config import load_config
from narrativepulse.train.common import load_panel_from_config
from narrativepulse.utils.io import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--as_of_date", required=True)
    return parser.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    _, panel = load_panel_from_config(args.config)
    row = panel[(panel["entity"] == args.entity) & (panel["date"] == args.as_of_date)]
    if row.empty:
        raise ValueError(f"No panel row found for entity={args.entity} date={args.as_of_date}")
    row = row.iloc[0]
    model = read_json(Path(cfg.artifacts_dir) / "stacker_model.json")
    feature_manifest = read_json(Path(cfg.artifacts_dir) / "feature_manifest.json")
    feature_names = feature_manifest["meta_features"]
    x = np.array(
        [
            row["attention_shock_index"],
            row["risk_shock_index"],
            row["gdelt_event_count_total"],
            row["wikipedia_pageviews"],
            0.5,
        ],
        dtype=float,
    )
    logits = float(np.dot(x, np.asarray(model["classifier_coef"])) + model["classifier_intercept"][0])
    prob = float(sigmoid(np.array([logits / model["temperature"]]))[0])
    severity = float(np.dot(x, np.asarray(model["regressor_coef"])) + model["regressor_intercept"])
    half_width = float(model["conformal_half_width"])
    top_idx = int(np.argmax(np.abs(np.asarray(model["classifier_coef"]))))
    print(
        {
            "entity": args.entity,
            "as_of_date": args.as_of_date,
            "shock_probability": prob,
            "severity_forecast": severity,
            "severity_interval": [severity - half_width, severity + half_width],
            "top_feature": feature_names[top_idx],
        }
    )


if __name__ == "__main__":
    main()
