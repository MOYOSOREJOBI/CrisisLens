from __future__ import annotations

import argparse
from pathlib import Path

from narrativepulse.models.graphsage import graph_embeddings_from_panel
from narrativepulse.train.common import feature_matrix, load_panel_from_config, save_torch_payload
from narrativepulse.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, panel = load_panel_from_config(args.config)
    x, cols = feature_matrix(panel, exclude={"shock_label", "severity_t_plus_1", "severity_t_plus_3", "severity_t_plus_7", "event_intensity_t_plus_1"})
    result = graph_embeddings_from_panel(panel["entity"].tolist(), x, hidden_dim=cfg.graph_hidden_size)
    artifacts_dir = ensure_dir(Path(cfg.artifacts_dir))
    write_json(
        artifacts_dir / "graph_features.json",
        {"entities": result.entity_to_index, "feature_columns": cols, "embedding_dim": int(result.embeddings.shape[1])},
    )
    save_torch_payload(
        artifacts_dir / "graph_embeddings.pt",
        {"entity_to_index": result.entity_to_index, "embeddings": result.embeddings},
    )


if __name__ == "__main__":
    main()
