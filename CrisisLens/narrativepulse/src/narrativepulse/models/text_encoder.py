from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class TextBatchOutput:
    embeddings: torch.Tensor
    stance_logits: torch.Tensor
    topic_logits: torch.Tensor
    toxicity_logits: torch.Tensor | None


class TextEncoderModel(nn.Module):
    def __init__(
        self,
        model_name: str = "distilroberta-base",
        hidden_dim: int = 128,
        topic_classes: int = 8,
        use_toxicity_head: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name)
        base_hidden = int(self.encoder.config.hidden_size)
        self.proj = nn.Linear(base_hidden, hidden_dim)
        self.stance_head = nn.Linear(hidden_dim, 3)
        self.topic_head = nn.Linear(hidden_dim, topic_classes)
        self.toxicity_head = nn.Linear(hidden_dim, 1) if use_toxicity_head else None

    def encode_texts(self, texts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in batch.items()}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> TextBatchOutput:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = encoded.last_hidden_state[:, 0]
        emb = self.proj(pooled)
        toxicity = self.toxicity_head(emb) if self.toxicity_head is not None else None
        return TextBatchOutput(
            embeddings=emb,
            stance_logits=self.stance_head(emb),
            topic_logits=self.topic_head(emb),
            toxicity_logits=toxicity,
        )

    @torch.no_grad()
    def aggregate_daily_embeddings(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.proj.out_features), dtype=np.float32)
        device = next(self.parameters()).device
        chunks = []
        for i in range(0, len(texts), batch_size):
            batch = self.encode_texts(texts[i : i + batch_size], device=device)
            outputs = self.forward(**batch)
            chunks.append(outputs.embeddings.cpu().numpy())
        return np.concatenate(chunks, axis=0)
