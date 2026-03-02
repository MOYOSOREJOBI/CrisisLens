from __future__ import annotations

import argparse

from narrativepulse.config import load_config
from narrativepulse.train.common import load_panel_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    _, panel = load_panel_from_config(args.config)
    latest = panel.sort_values(["date", "entity"]).groupby("entity").tail(1)
    rows = latest[["entity", "date", "attention_shock_index", "risk_shock_index"]].to_dict("records")
    print({"mode": "realtime_demo", "entities": rows, "artifacts_dir": cfg.artifacts_dir})


if __name__ == "__main__":
    main()
