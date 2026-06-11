"""Retrieval with the official precomputed track embeddings."""

import json
import os
import re
from typing import Iterable

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm


KNOWN_DIMENSIONS = {
    "audio-laion_clap": 512,
    "image-siglip2": 768,
    "cf-bpr": 128,
    "attributes-qwen3_embedding_0.6b": 1024,
    "lyrics-qwen3_embedding_0.6b": 1024,
    "metadata-qwen3_embedding_0.6b": 1024,
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


class PrecomputedTrackEmbeddingIndex:
    """Cosine-similarity index over an official track embedding field."""

    def __init__(
        self,
        dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        split: str = "all_tracks",
        embedding_field: str = "image-siglip2",
        cache_dir: str = "./cache",
        device: str | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.embedding_field = embedding_field
        self.cache_dir = cache_dir
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        cache_name = f"{_safe_name(split)}__{_safe_name(embedding_field)}"
        self.index_dir = os.path.join(cache_dir, "dense", cache_name)
        self.track_ids_path = os.path.join(self.index_dir, "track_ids.json")
        self.embeddings_path = os.path.join(self.index_dir, "embeddings.npy")
        self.valid_path = os.path.join(self.index_dir, "valid.npy")

        if not all(os.path.exists(path) for path in [
            self.track_ids_path,
            self.embeddings_path,
            self.valid_path,
        ]):
            self._build_cache()

        with open(self.track_ids_path, encoding="utf-8") as file:
            self.track_ids: list[str] = json.load(file)
        self.track_to_index = {track_id: index for index, track_id in enumerate(self.track_ids)}
        self.embeddings = np.load(self.embeddings_path, mmap_mode="r")
        self.valid = np.load(self.valid_path, mmap_mode="r")
        if len(self.track_ids) != len(self.embeddings) or len(self.track_ids) != len(self.valid):
            raise ValueError(f"Inconsistent embedding cache at {self.index_dir}.")

        self.corpus_tensor: torch.Tensor | None = None
        self.valid_tensor: torch.Tensor | None = None

    def _build_cache(self) -> None:
        os.makedirs(self.index_dir, exist_ok=True)
        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
        ).select_columns(["track_id", self.embedding_field])
        dimension = KNOWN_DIMENSIONS.get(self.embedding_field)
        if dimension is None:
            dimension = max(len(vector) for vector in dataset[self.embedding_field])

        track_ids: list[str] = []
        vectors: list[np.ndarray] = []
        valid_rows: list[bool] = []
        for item in tqdm(dataset, desc=f"Caching {self.embedding_field}"):
            vector = np.asarray(item[self.embedding_field], dtype=np.float32)
            is_valid = vector.ndim == 1 and vector.size == dimension and np.linalg.norm(vector) > 0
            if is_valid:
                vector = vector / np.linalg.norm(vector)
            else:
                vector = np.zeros(dimension, dtype=np.float32)
            track_ids.append(str(item["track_id"]))
            vectors.append(vector)
            valid_rows.append(bool(is_valid))

        np.save(self.embeddings_path, np.stack(vectors).astype(np.float16))
        np.save(self.valid_path, np.asarray(valid_rows, dtype=np.bool_))
        with open(self.track_ids_path, "w", encoding="utf-8") as file:
            json.dump(track_ids, file, ensure_ascii=False)

    def _load_tensors(self) -> None:
        if self.corpus_tensor is not None:
            return
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.corpus_tensor = torch.from_numpy(np.array(self.embeddings, copy=True)).to(self.device, dtype=dtype)
        self.valid_tensor = torch.from_numpy(np.array(self.valid, copy=True)).to(self.device)

    def _history_query(self, track_ids: Iterable[str], aggregation: str) -> np.ndarray | None:
        indices = [
            self.track_to_index[track_id]
            for track_id in track_ids
            if track_id in self.track_to_index and self.valid[self.track_to_index[track_id]]
        ]
        if not indices:
            return None
        vectors = np.asarray(self.embeddings[indices], dtype=np.float32)
        if aggregation == "last":
            query = vectors[-1]
        elif aggregation == "mean":
            query = vectors.mean(axis=0)
        elif aggregation == "max":
            query = vectors.max(axis=0)
        else:
            raise ValueError(f"Unsupported history aggregation: {aggregation}")
        norm = float(np.linalg.norm(query))
        return query / norm if norm else None

    def batch_history_retrieval(
        self,
        histories: list[list[str]],
        topk: int,
        aggregation: str = "mean",
        batch_size: int = 32,
        exclude_seen: bool = True,
    ) -> tuple[list[list[str]], list[list[float]]]:
        """Retrieve tracks similar to each list of previously recommended tracks."""
        if topk <= 0:
            return ([[] for _ in histories], [[] for _ in histories])
        self._load_tensors()
        topk = min(topk, len(self.track_ids))
        dtype = self.corpus_tensor.dtype
        all_ids: list[list[str]] = []
        all_scores: list[list[float]] = []

        for start in tqdm(range(0, len(histories), batch_size), desc=f"{self.embedding_field} {aggregation}"):
            batch_histories = histories[start:start + batch_size]
            queries = []
            available = []
            for history in batch_histories:
                query = self._history_query(history, aggregation)
                available.append(query is not None)
                queries.append(
                    query if query is not None else np.zeros(self.embeddings.shape[1], dtype=np.float32)
                )

            query_tensor = torch.from_numpy(np.stack(queries)).to(self.device, dtype=dtype)
            with torch.inference_mode():
                similarities = query_tensor @ self.corpus_tensor.T
                similarities[:, ~self.valid_tensor] = -torch.inf
                if exclude_seen:
                    for row, history in enumerate(batch_histories):
                        seen_indices = [
                            self.track_to_index[track_id]
                            for track_id in history
                            if track_id in self.track_to_index
                        ]
                        if seen_indices:
                            similarities[row, seen_indices] = -torch.inf
                values, indices = torch.topk(similarities, k=topk, dim=1)

            for is_available, row_indices, row_values in zip(
                available,
                indices.detach().cpu().tolist(),
                values.detach().cpu().float().tolist(),
            ):
                if not is_available:
                    all_ids.append([])
                    all_scores.append([])
                    continue
                pairs = [
                    (self.track_ids[index], score)
                    for index, score in zip(row_indices, row_values)
                    if np.isfinite(score)
                ]
                all_ids.append([track_id for track_id, _ in pairs])
                all_scores.append([score for _, score in pairs])
        return all_ids, all_scores

    def unload(self) -> None:
        self.corpus_tensor = None
        self.valid_tensor = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
