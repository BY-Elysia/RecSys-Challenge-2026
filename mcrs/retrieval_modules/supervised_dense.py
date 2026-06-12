"""Supervised dense retrieval with lightweight adapters over Qwen embeddings."""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from mcrs.retrieval_modules.qwen_dense import (
    DEFAULT_TASK_INSTRUCTION,
    QwenDenseRetriever,
)


class ResidualProjection(nn.Module):
    def __init__(self, dimension: int, bottleneck: int, dropout: float) -> None:
        super().__init__()
        self.down = nn.Linear(dimension, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, dimension, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(1.0))
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.dropout(F.gelu(self.down(values))))
        return F.normalize(values + self.residual_scale * residual, p=2, dim=-1)


class DualTowerProjection(nn.Module):
    def __init__(
        self,
        dimension: int = 1024,
        bottleneck: int = 128,
        dropout: float = 0.1,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.query_projection = ResidualProjection(dimension, bottleneck, dropout)
        self.track_projection = ResidualProjection(dimension, bottleneck, dropout)
        self.logit_scale = nn.Parameter(
            torch.tensor(float(np.log(1.0 / temperature)), dtype=torch.float32)
        )

    def encode_queries(self, values: torch.Tensor) -> torch.Tensor:
        return self.query_projection(values)

    def encode_tracks(self, values: torch.Tensor) -> torch.Tensor:
        return self.track_projection(values)

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)


def load_projection_model(
    checkpoint_dir: str,
    device: torch.device,
) -> tuple[DualTowerProjection, dict]:
    config_path = os.path.join(checkpoint_dir, "config.json")
    model_path = os.path.join(checkpoint_dir, "model.pt")
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    model = DualTowerProjection(
        dimension=int(config["dimension"]),
        bottleneck=int(config["bottleneck"]),
        dropout=float(config.get("dropout", 0.0)),
        temperature=float(config.get("temperature", 0.05)),
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config


class SupervisedDenseRetriever:
    """Retrieve tracks with a trained projection over frozen Qwen embeddings."""

    def __init__(
        self,
        checkpoint_dir: str,
        cache_dir: str = "./cache",
        device: str | None = None,
        query_batch_size: int = 16,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.cache_dir = cache_dir
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, self.config = load_projection_model(checkpoint_dir, self.device)
        self.query_batch_size = query_batch_size

        with open(
            os.path.join(checkpoint_dir, "track_ids.json"),
            encoding="utf-8",
        ) as file:
            self.track_ids: list[str] = json.load(file)
        corpus_path = os.path.join(checkpoint_dir, "projected_track_embeddings.npy")
        self.corpus_embeddings = np.load(corpus_path, mmap_mode="r")
        if len(self.track_ids) != len(self.corpus_embeddings):
            raise ValueError("Projected corpus rows do not match track IDs.")

        self.query_encoder = QwenDenseRetriever(
            embedding_field=self.config.get(
                "embedding_field",
                "metadata-qwen3_embedding_0.6b",
            ),
            model_name=self.config.get(
                "query_model_name",
                "Qwen/Qwen3-Embedding-0.6B",
            ),
            cache_dir=cache_dir,
            device=str(self.device),
            max_length=int(self.config.get("max_length", 512)),
            query_batch_size=query_batch_size,
            task_instruction=self.config.get(
                "task_instruction",
                DEFAULT_TASK_INSTRUCTION,
            ),
        )
        self.corpus_tensor: torch.Tensor | None = None

    def _load_corpus_tensor(self) -> None:
        if self.corpus_tensor is not None:
            return
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.corpus_tensor = torch.from_numpy(
            np.array(self.corpus_embeddings, copy=True)
        ).to(self.device, dtype=dtype)

    def batch_text_to_item_retrieval_with_scores(
        self,
        queries: list[str],
        topk: int,
    ) -> tuple[list[list[str]], list[list[float]]]:
        if topk <= 0:
            return ([[] for _ in queries], [[] for _ in queries])
        self._load_corpus_tensor()
        topk = min(topk, len(self.track_ids))
        dtype = self.corpus_tensor.dtype
        all_ids: list[list[str]] = []
        all_scores: list[list[float]] = []

        for start in tqdm(
            range(0, len(queries), self.query_batch_size),
            desc="Supervised dense retrieval",
        ):
            base = self.query_encoder.encode_queries(
                queries[start:start + self.query_batch_size]
            ).to(dtype=torch.float32)
            with torch.inference_mode():
                projected = self.model.encode_queries(base).to(dtype=dtype)
                scores, indices = torch.topk(
                    projected @ self.corpus_tensor.T,
                    k=topk,
                    dim=1,
                )
            for row_indices, row_scores in zip(
                indices.detach().cpu().tolist(),
                scores.detach().cpu().float().tolist(),
            ):
                all_ids.append([self.track_ids[index] for index in row_indices])
                all_scores.append(row_scores)
        return all_ids, all_scores

    def batch_text_to_item_retrieval(
        self,
        queries: list[str],
        topk: int,
    ) -> list[list[str]]:
        ids, _ = self.batch_text_to_item_retrieval_with_scores(queries, topk)
        return ids

    def unload(self) -> None:
        self.model = None
        self.corpus_tensor = None
        self.query_encoder.unload()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
