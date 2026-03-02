from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from narrativepulse.config import load_config
from narrativepulse.data.sources import (
    cache_json,
    fallback_dates,
    fetch_gdelt_doc_timeline,
    synthetic_gdelt_frame,
)
from narrativepulse.logging import get_logger
from narrativepulse.utils.io import ensure_dir, write_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("download_gdelt")
    raw_dir = ensure_dir(Path(cfg.data_dir) / "raw" / "gdelt")
    for entity in cfg.entities:
        query = f'"{entity.replace("_", " ")}"'
        try:
            payload = fetch_gdelt_doc_timeline(query, cfg.start_date, cfg.end_date)
            cache_json(payload, raw_dir / f"{entity}.json")
            logger.info("Fetched GDELT timeline payload for %s", entity)
        except Exception as exc:
            logger.warning("Falling back to synthetic GDELT payload for %s: %s", entity, exc)
        frame = synthetic_gdelt_frame(entity.replace("_", " "), fallback_dates(cfg.start_date, cfg.end_date))
        write_parquet(frame, raw_dir / f"{entity}.parquet")


if __name__ == "__main__":
    main()
