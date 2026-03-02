from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from narrativepulse.models.hawkes import SimpleHawkes
from narrativepulse.train.common import load_panel_from_config
from narrativepulse.utils.io import ensure_dir, write_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, panel = load_panel_from_config(args.config)
    outputs = []
    hawkes = SimpleHawkes()
    for entity, frame in panel.groupby("entity"):
        forecast = hawkes.fit_predict(frame["gdelt_event_count_total"])
        local = frame[["entity", "date"]].copy()
        local["hawkes_intensity"] = forecast.intensity
        local["hawkes_anomaly"] = forecast.anomaly_score
        outputs.append(local)
    out = pd.concat(outputs, ignore_index=True)
    write_parquet(out, ensure_dir(Path(cfg.artifacts_dir)) / "hawkes_features.parquet")


if __name__ == "__main__":
    main()
