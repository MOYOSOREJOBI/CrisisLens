from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from narrativepulse.config import load_config
from narrativepulse.data.sources import fetch_wikipedia_pageviews, synthetic_wiki_frame
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
    logger = get_logger("download_wikipedia_pageviews")
    out_dir = ensure_dir(Path(cfg.data_dir) / "raw" / "wikipedia")
    for entity in cfg.entities:
        article = entity
        try:
            frame = fetch_wikipedia_pageviews(article, cfg.start_date, cfg.end_date)
            if frame.empty:
                raise ValueError("Empty pageviews response")
            logger.info("Fetched pageviews for %s", entity)
        except Exception as exc:
            logger.warning("Falling back to synthetic pageviews for %s: %s", entity, exc)
            frame = synthetic_wiki_frame(entity.replace("_", " "), date_range(cfg.start_date, cfg.end_date))
        write_parquet(frame, out_dir / f"{entity}.parquet")


if __name__ == "__main__":
    main()
