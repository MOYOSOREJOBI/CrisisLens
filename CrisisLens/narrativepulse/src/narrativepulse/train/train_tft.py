from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from narrativepulse.models.tft import SimpleTFT
from narrativepulse.train.common import chronological_split, feature_matrix, load_panel_from_config, save_torch_payload
from narrativepulse.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def to_sequences(x: np.ndarray, y: np.ndarray, seq_len: int = 14) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(seq_len, len(x)):
        xs.append(x[i - seq_len : i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def main() -> None:
    args = parse_args()
    cfg, panel = load_panel_from_config(args.config)
    splits = chronological_split(panel)
    x_train, cols = feature_matrix(splits["train"], exclude={"shock_label", "severity_t_plus_1", "severity_t_plus_3", "severity_t_plus_7", "event_intensity_t_plus_1"})
    y_train = splits["train"][["severity_t_plus_1", "severity_t_plus_3", "severity_t_plus_7"]].to_numpy(dtype=np.float32)
    if len(x_train) < 15:
        raise ValueError("Not enough training rows for TFT sequence construction.")
    seq_x, seq_y = to_sequences(x_train, y_train)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = SimpleTFT(input_dim=seq_x.shape[-1], hidden_dim=cfg.tft_hidden_size, horizons=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()
    x_tensor = torch.tensor(seq_x, device=device)
    y_tensor = torch.tensor(seq_y, device=device)
    model.train()
    for _ in range(max(1, int(cfg.epochs))):
        optimizer.zero_grad()
        pred = model(x_tensor)
        loss = loss_fn(pred, y_tensor)
        loss.backward()
        optimizer.step()
    save_torch_payload(
        ensure_dir(Path(cfg.artifacts_dir)) / "tft_model.pt",
        {"state_dict": model.state_dict(), "input_dim": seq_x.shape[-1], "hidden_dim": cfg.tft_hidden_size},
    )
    write_json(Path(cfg.artifacts_dir) / "tft_features.json", {"feature_columns": cols, "seq_len": 14})


if __name__ == "__main__":
    main()
