from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from torch import nn


@dataclass
class GraphEmbeddingResult:
    entity_to_index: dict[str, int]
    embeddings: np.ndarray


class GraphSAGEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lin_self = nn.Linear(input_dim, hidden_dim)
        self.lin_neigh = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neigh = (adj @ x) / deg
        hidden = torch.relu(self.lin_self(x) + self.lin_neigh(neigh))
        return self.out(hidden)


def build_co_mention_graph(entities: list[str]) -> nx.Graph:
    graph = nx.Graph()
    for entity in entities:
        graph.add_node(entity)
    ordered = sorted(set(entities))
    for i, left in enumerate(ordered):
        for right in ordered[i + 1 :]:
            graph.add_edge(left, right, weight=1.0)
    return graph


def graph_embeddings_from_panel(panel_entities: list[str], features: np.ndarray, hidden_dim: int) -> GraphEmbeddingResult:
    unique_entities = sorted(set(panel_entities))
    index = {entity: i for i, entity in enumerate(unique_entities)}
    x = np.zeros((len(unique_entities), features.shape[1]), dtype=np.float32)
    counts = np.zeros(len(unique_entities), dtype=np.float32)
    for entity, feat in zip(panel_entities, features):
        idx = index[entity]
        x[idx] += feat
        counts[idx] += 1.0
    counts = np.clip(counts, 1.0, None)
    x = x / counts[:, None]
    graph = build_co_mention_graph(unique_entities)
    adj = np.zeros((len(unique_entities), len(unique_entities)), dtype=np.float32)
    for left, right, data in graph.edges(data=True):
        i = index[left]
        j = index[right]
        adj[i, j] = float(data["weight"])
        adj[j, i] = float(data["weight"])
    model = GraphSAGEEncoder(input_dim=x.shape[1], hidden_dim=hidden_dim)
    with torch.no_grad():
        emb = model(torch.tensor(x), torch.tensor(adj)).numpy()
    return GraphEmbeddingResult(entity_to_index=index, embeddings=emb)
