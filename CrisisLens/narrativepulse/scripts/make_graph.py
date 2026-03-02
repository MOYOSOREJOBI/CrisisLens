from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx

from narrativepulse.config import load_config
from narrativepulse.utils.io import read_parquet, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    panel = read_parquet(Path(cfg.data_dir) / "processed" / "panel.parquet")
    graph = nx.Graph()
    for date, frame in panel.groupby("date"):
        entities = sorted(frame["entity"].unique())
        for entity in entities:
            graph.add_node(entity)
        for i, left in enumerate(entities):
            for right in entities[i + 1 :]:
                if graph.has_edge(left, right):
                    graph[left][right]["weight"] += 1
                else:
                    graph.add_edge(left, right, weight=1)
    serialised = {
        "nodes": [{"id": node} for node in graph.nodes()],
        "edges": [
            {"source": left, "target": right, "weight": data["weight"]}
            for left, right, data in graph.edges(data=True)
        ],
    }
    write_json(Path(cfg.data_dir) / "processed" / "graph.json", serialised)


if __name__ == "__main__":
    main()
