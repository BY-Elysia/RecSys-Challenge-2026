"""Dense retrieval against the official precomputed Qwen track embeddings."""

import json
import os
import re
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_TASK_INSTRUCTION = (
    "Given a conversational music request and listening context, "
    "retrieve relevant music tracks."
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


class QwenDenseRetriever:
    """Retrieve tracks by cosine similarity to official Qwen metadata embeddings."""

    def __init__(
        self,
        dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        split: str = "all_tracks",
        embedding_field: str = "metadata-qwen3_embedding_0.6b",
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        cache_dir: str = "./cache",
        device: str | None = None,
        max_length: int = 512,
        query_batch_size: int = 8,
        task_instruction: str = DEFAULT_TASK_INSTRUCTION,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.embedding_field = embedding_field
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length
        self.query_batch_size = query_batch_size
        self.task_instruction = task_instruction

        cache_name = f"{_safe_name(split)}__{_safe_name(embedding_field)}"
        self.dense_cache_dir = os.path.join(cache_dir, "dense", cache_name)
        self.track_ids_path = os.path.join(self.dense_cache_dir, "track_ids.json")
        self.embeddings_path = os.path.join(self.dense_cache_dir, "embeddings.npy")

        if not (os.path.exists(self.track_ids_path) and os.path.exists(self.embeddings_path)):
            self._build_corpus_cache()

        with open(self.track_ids_path, "r", encoding="utf-8") as file:
            self.track_ids: list[str] = json.load(file)
        self.corpus_embeddings = np.load(self.embeddings_path, mmap_mode="r")
        if len(self.track_ids) != len(self.corpus_embeddings):
            raise ValueError("Dense cache is inconsistent: track IDs and embedding rows differ.")

        self.tokenizer = None
        self.model = None
        self.corpus_tensor = None

    def _build_corpus_cache(self) -> None:
        os.makedirs(self.dense_cache_dir, exist_ok=True)
        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True,
            cache_dir=self.cache_dir,
        ).select_columns(["track_id", self.embedding_field])

        track_ids: list[str] = []
        vectors: list[np.ndarray] = []
        zero_vectors = 0
        invalid_vectors = 0
        known_dimensions = {
            "audio-laion_clap": 512,
            "image-siglip2": 768,
            "cf-bpr": 128,
            "attributes-qwen3_embedding_0.6b": 1024,
            "lyrics-qwen3_embedding_0.6b": 1024,
            "metadata-qwen3_embedding_0.6b": 1024,
        }
        embedding_dimension: int | None = known_dimensions.get(self.embedding_field)
        for item in tqdm(dataset, desc=f"Caching dense corpus ({self.embedding_field})"):
            vector = np.asarray(item[self.embedding_field], dtype=np.float32)
            if embedding_dimension is None and vector.ndim == 1 and vector.size > 0:
                embedding_dimension = int(vector.size)
            if embedding_dimension is None:
                raise ValueError("Could not infer dense embedding dimension from the current record.")
            if vector.ndim != 1 or vector.size != embedding_dimension:
                invalid_vectors += 1
                vector = np.zeros(embedding_dimension, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                zero_vectors += 1
            else:
                vector = vector / norm
            track_ids.append(str(item["track_id"]))
            vectors.append(vector)

        if not vectors:
            raise ValueError(f"No embeddings found in {self.dataset_name}/{self.split}.")

        embeddings = np.stack(vectors).astype(np.float16)
        np.save(self.embeddings_path, embeddings)
        with open(self.track_ids_path, "w", encoding="utf-8") as file:
            json.dump(track_ids, file, ensure_ascii=False)
        print(
            f"Cached {len(track_ids)} dense track embeddings to {self.dense_cache_dir} "
            f"({zero_vectors} zero vectors retained, including {invalid_vectors} invalid shapes)."
        )

    def _load_query_encoder(self) -> None:
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            cache_dir=self.cache_dir,
        )
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            dtype=dtype,
            attn_implementation="eager",
        )
        self.model.to(self.device)
        self.model.eval()

    def _load_corpus_tensor(self) -> None:
        if self.corpus_tensor is not None:
            return
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        corpus = np.array(self.corpus_embeddings, copy=True)
        self.corpus_tensor = torch.from_numpy(corpus).to(device=self.device, dtype=dtype)

    def encode_queries(self, queries: Iterable[str]) -> torch.Tensor:
        self._load_query_encoder()
        instructed_queries = [
            f"Instruct: {self.task_instruction}\nQuery:{query}"
            for query in queries
        ]
        batch = self.tokenizer(
            instructed_queries,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.inference_mode():
            outputs = self.model(**batch)
            embeddings = _last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            return F.normalize(embeddings, p=2, dim=1)

    def batch_text_to_item_retrieval(self, queries: list[str], topk: int) -> list[list[str]]:
        if topk <= 0:
            return [[] for _ in queries]
        self._load_corpus_tensor()
        topk = min(topk, len(self.track_ids))

        results: list[list[str]] = []
        for start in tqdm(
            range(0, len(queries), self.query_batch_size),
            desc="Dense query retrieval",
        ):
            query_embeddings = self.encode_queries(queries[start:start + self.query_batch_size])
            with torch.inference_mode():
                indices = torch.topk(query_embeddings @ self.corpus_tensor.T, k=topk, dim=1).indices
            for row in indices.detach().cpu().tolist():
                results.append([self.track_ids[index] for index in row])
        return results

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.corpus_tensor = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
