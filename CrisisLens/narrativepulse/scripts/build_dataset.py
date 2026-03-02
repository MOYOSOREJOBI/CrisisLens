from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from narrativepulse.config import load_config
from narrativepulse.data.dataset_builder import build_daily_panel, save_panel
from narrativepulse.logging import get_logger
from narrativepulse.utils.io import read_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _load_entity_frames(base: Path, entities: list[str]) -> pd.DataFrame:
    frames = []
    for entity in entities:
        frame = read_parquet(base / f"{entity}.parquet")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("build_dataset")
    data_dir = Path(cfg.data_dir)
    gdelt_df = _load_entity_frames(data_dir / "raw" / "gdelt", cfg.entities)
    wiki_df = _load_entity_frames(data_dir / "raw" / "wikipedia", cfg.entities)
    fred_df = read_parquet(data_dir / "raw" / "fred" / "fred.parquet")
    result = build_daily_panel(
        gdelt_df=gdelt_df,
        wiki_df=wiki_df,
        fred_df=fred_df,
        shock_quantile=cfg.shock_quantile,
        rolling_window=cfg.rolling_window,
    )
    output_path = data_dir / "processed" / "panel.parquet"
    save_panel(result, output_path)
    logger.info("Saved panel to %s", output_path)


if __name__ == "__main__":
    main()
