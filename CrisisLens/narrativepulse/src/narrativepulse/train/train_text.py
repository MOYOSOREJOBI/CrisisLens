from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from narrativepulse.models.text_encoder import TextEncoderModel
from narrativepulse.train.common import chronological_split, load_panel_from_config, save_torch_payload
from narrativepulse.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, panel = load_panel_from_config(args.config)
    splits = chronological_split(panel)
    artifacts_dir = ensure_dir(Path(cfg.artifacts_dir))
    model = TextEncoderModel(model_name=cfg.text_model_name, hidden_dim=cfg.hidden_size)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device)
    texts = [f"{row.entity} {row.date}" for row in splits["train"].itertuples()]
    embeddings = model.aggregate_daily_embeddings(texts)
    # Lightweight weak supervision targets for the text branch.
    y_stance = torch.tensor((splits["train"]["risk_shock_index"] > 0).astype(int).to_numpy(), dtype=torch.long, device=device)
    y_topic = torch.tensor(np.mod(np.arange(len(texts)), 8), dtype=torch.long, device=device)
    batch = model.encode_texts(texts, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(max(1, int(cfg.epochs))):
        optimizer.zero_grad()
        out = model(**batch)
        loss = loss_fn(out.stance_logits, y_stance) + loss_fn(out.topic_logits, y_topic)
        loss.backward()
        optimizer.step()
    model.eval()
    save_torch_payload(
        artifacts_dir / "text_model.pt",
        {"state_dict": model.state_dict(), "model_name": cfg.text_model_name, "hidden_dim": cfg.hidden_size},
    )
    write_json(
        artifacts_dir / "text_features.json",
        {
            "train_rows": int(len(texts)),
            "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
        },
    )


if __name__ == "__main__":
    main()
