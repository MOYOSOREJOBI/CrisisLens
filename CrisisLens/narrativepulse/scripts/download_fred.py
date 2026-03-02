from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from narrativepulse.config import load_config
from narrativepulse.data.sources import fetch_fred_series, synthetic_fred_frame
from narrativepulse.logging import get_logger
from narrativepulse.utils.dates import date_range
from narrativepulse.utils.io import ensure_dir, write_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("download_fred")
    out_dir = ensure_dir(Path(cfg.data_dir) / "raw" / "fred")
    api_key = os.getenv("FRED_API_KEY")
    if api_key:
        frames: list[pd.DataFrame] = []
        for series_id in cfg.fred_series:
            try:
                frame = fetch_fred_series(series_id, api_key, cfg.start_date, cfg.end_date)
                frames.append(frame)
            except Exception as exc:
                logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        if frames:
            merged = frames[0]
            for frame in frames[1:]:
                merged = merged.merge(frame, on="date", how="outer")
            write_parquet(merged.sort_values("date").reset_index(drop=True), out_dir / "fred.parquet")
            return
    logger.warning("Using synthetic FRED frame.")
    frame = synthetic_fred_frame(cfg.fred_series, date_range(cfg.start_date, cfg.end_date))
    write_parquet(frame, out_dir / "fred.parquet")


if __name__ == "__main__":
    main()
