"""Train lightweight supervised query/track adapters for dense music retrieval."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluate_final_turn_recall import DEFAULT_BLIND_TURN_WEIGHTS
from mcrs.retrieval_modules.bm25 import BM25_MODEL
from mcrs.retrieval_modules.qwen_dense import (
    DEFAULT_TASK_INSTRUCTION,
    QwenDenseRetriever,
)
from mcrs.retrieval_modules.supervised_dense import DualTowerProjection
from train_ltr_ranker import RankingTask, build_tasks
from train_reranker import track_to_text


@dataclass
class QueryCache:
    embeddings: np.ndarray
    target_indices: np.ndarray
    turns: np.ndarray
    session_ids: list[str]
    histories: list[list[str]]


class ContrastiveDataset(Dataset):
    def __init__(
        self,
        query_embeddings: np.ndarray,
        target_indices: np.ndarray,
        hard_negative_indices: np.ndarray,
        track_embeddings: np.ndarray,
    ) -> None:
        self.query_embeddings = query_embeddings
        self.target_indices = target_indices
        self.hard_negative_indices = hard_negative_indices
        self.track_embeddings = track_embeddings

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        target = int(self.target_indices[index])
        hard_indices = self.hard_negative_indices[index]
        return {
            "query": torch.from_numpy(
                np.asarray(self.query_embeddings[index], dtype=np.float32)
            ),
            "positive": torch.from_numpy(
                np.asarray(self.track_embeddings[target], dtype=np.float32)
            ),
            "hard_negatives": torch.from_numpy(
                np.asarray(self.track_embeddings[hard_indices], dtype=np.float32)
            ),
            "target_index": torch.tensor(target, dtype=torch.long),
        }


def select_tasks(
    split: str,
    args: argparse.Namespace,
    track_texts: dict[str, str],
) -> list[RankingTask]:
    dataset = load_dataset(
        args.dataset_name,
        split=split,
        cache_dir=args.cache_dir,
    )
    tasks = build_tasks(dataset, track_texts, args.history_turns, "all")
    if split == args.train_split:
        random.Random(args.seed).shuffle(tasks)
        if args.max_train_tasks:
            tasks = tasks[:min(args.max_train_tasks, len(tasks))]
    elif args.max_dev_tasks:
        tasks = tasks[:min(args.max_dev_tasks, len(tasks))]
    return tasks


def task_query(task: RankingTask, mode: str) -> str:
    if mode == "feedback":
        return task.feedback_query
    if mode == "legacy":
        return task.legacy_query
    if mode == "current":
        return task.current_request
    raise ValueError(mode)


def cache_paths(cache_dir: str) -> dict[str, str]:
    return {
        "embeddings": os.path.join(cache_dir, "query_embeddings.npy"),
        "target_indices": os.path.join(cache_dir, "target_indices.npy"),
        "turns": os.path.join(cache_dir, "turns.npy"),
        "metadata": os.path.join(cache_dir, "metadata.json"),
    }


def save_query_cache(
    cache_dir: str,
    embeddings: np.ndarray,
    target_indices: np.ndarray,
    tasks: list[RankingTask],
    settings: dict,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    paths = cache_paths(cache_dir)
    np.save(paths["embeddings"], embeddings.astype(np.float16))
    np.save(paths["target_indices"], target_indices.astype(np.int32))
    np.save(
        paths["turns"],
        np.asarray([task.turn_number for task in tasks], dtype=np.int8),
    )
    with open(paths["metadata"], "w", encoding="utf-8") as file:
        json.dump(
            {
                "settings": settings,
                "session_ids": [task.session_id for task in tasks],
                "histories": [task.history for task in tasks],
            },
            file,
            ensure_ascii=False,
        )


def load_query_cache(cache_dir: str) -> QueryCache:
    paths = cache_paths(cache_dir)
    with open(paths["metadata"], encoding="utf-8") as file:
        metadata = json.load(file)
    return QueryCache(
        embeddings=np.load(paths["embeddings"], mmap_mode="r"),
        target_indices=np.load(paths["target_indices"], mmap_mode="r"),
        turns=np.load(paths["turns"], mmap_mode="r"),
        session_ids=metadata["session_ids"],
        histories=metadata["histories"],
    )


def build_query_cache(
    tasks: list[RankingTask],
    cache_dir: str,
    dense: QwenDenseRetriever,
    track_to_index: dict[str, int],
    args: argparse.Namespace,
) -> QueryCache:
    paths = cache_paths(cache_dir)
    if (
        not args.rebuild_query_cache
        and all(os.path.exists(path) for path in paths.values())
    ):
        cached = load_query_cache(cache_dir)
        if len(cached.target_indices) == len(tasks):
            return cached

    embeddings = []
    queries = [task_query(task, args.query_mode) for task in tasks]
    for start in tqdm(
        range(0, len(queries), args.query_batch_size),
        desc=f"Encoding {os.path.basename(cache_dir)}",
    ):
        batch = dense.encode_queries(queries[start:start + args.query_batch_size])
        embeddings.append(batch.detach().cpu().float().numpy())
    matrix = np.concatenate(embeddings) if embeddings else np.empty((0, 1024))
    targets = np.asarray(
        [track_to_index.get(task.target, -1) for task in tasks],
        dtype=np.int32,
    )
    save_query_cache(
        cache_dir,
        matrix,
        targets,
        tasks,
        {
            "query_mode": args.query_mode,
            "history_turns": args.history_turns,
            "query_model_name": args.query_model_name,
            "task_instruction": args.task_instruction,
            "max_length": args.max_length,
        },
    )
    return load_query_cache(cache_dir)


def build_hard_negative_cache(
    query_cache: QueryCache,
    track_embeddings: np.ndarray,
    track_ids: list[str],
    output_path: str,
    args: argparse.Namespace,
) -> np.ndarray:
    if os.path.exists(output_path) and not args.rebuild_hard_negatives:
        cached = np.load(output_path, mmap_mode="r")
        if cached.shape == (len(query_cache.target_indices), args.hard_negatives):
            return cached

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    corpus = torch.from_numpy(np.array(track_embeddings, copy=True)).to(
        device,
        dtype=dtype,
    )
    track_to_index = {track_id: index for index, track_id in enumerate(track_ids)}
    rows = []
    search_k = min(
        len(track_ids),
        max(args.hard_negative_search_topk, args.hard_negatives + 16),
    )
    for start in tqdm(
        range(0, len(query_cache.target_indices), args.retrieval_batch_size),
        desc="Mining dense hard negatives",
    ):
        queries = torch.from_numpy(
            np.asarray(
                query_cache.embeddings[
                    start:start + args.retrieval_batch_size
                ],
                dtype=np.float32,
            )
        ).to(device, dtype=dtype)
        with torch.inference_mode():
            indices = torch.topk(queries @ corpus.T, k=search_k, dim=1).indices
        for local_row, candidates in enumerate(indices.detach().cpu().tolist()):
            row_index = start + local_row
            excluded = {
                int(query_cache.target_indices[row_index]),
                *[
                    track_to_index[track_id]
                    for track_id in query_cache.histories[row_index]
                    if track_id in track_to_index
                ],
            }
            negatives = [index for index in candidates if index not in excluded]
            if len(negatives) < args.hard_negatives:
                raise RuntimeError("Hard-negative search did not produce enough tracks.")
            rows.append(negatives[:args.hard_negatives])
    output = np.asarray(rows, dtype=np.int32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, output)
    del corpus
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.load(output_path, mmap_mode="r")


def contrastive_loss(
    model: DualTowerProjection,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    in_batch_weight: float,
    alignment_weight: float,
) -> torch.Tensor:
    base_queries = F.normalize(batch["query"].to(device), p=2, dim=-1)
    base_positives = F.normalize(batch["positive"].to(device), p=2, dim=-1)
    base_hard = F.normalize(batch["hard_negatives"].to(device), p=2, dim=-1)
    queries = model.encode_queries(base_queries)
    positives = model.encode_tracks(base_positives)
    hard = model.encode_tracks(base_hard)
    target_indices = batch["target_index"].to(device)

    positive_logits = torch.einsum("bd,bd->b", queries, positives)[:, None]
    hard_logits = torch.einsum("bd,bhd->bh", queries, hard)
    hard_loss = F.cross_entropy(
        torch.cat([positive_logits, hard_logits], dim=1) * model.scale(),
        torch.zeros(len(queries), dtype=torch.long, device=device),
    )
    loss = hard_loss

    if in_batch_weight > 0:
        in_batch_logits = queries @ positives.T * model.scale()
        duplicate_targets = target_indices[:, None] == target_indices[None, :]
        duplicate_targets.fill_diagonal_(False)
        in_batch_logits = in_batch_logits.masked_fill(
            duplicate_targets,
            -torch.inf,
        )
        in_batch_loss = F.cross_entropy(
            in_batch_logits,
            torch.arange(len(queries), device=device),
        )
        loss = loss + in_batch_weight * in_batch_loss

    if alignment_weight > 0:
        query_alignment = 1.0 - torch.einsum(
            "bd,bd->b",
            queries,
            base_queries,
        ).mean()
        loss = loss + alignment_weight * query_alignment
        if any(
            parameter.requires_grad
            for parameter in model.track_projection.parameters()
        ):
            track_alignment = 1.0 - torch.einsum(
                "bd,bd->b",
                positives,
                base_positives,
            ).mean()
            loss = loss + alignment_weight * track_alignment
    return loss


def project_corpus(
    model: DualTowerProjection,
    track_embeddings: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    rows = []
    model.eval()
    for start in range(0, len(track_embeddings), batch_size):
        batch = torch.from_numpy(
            np.asarray(
                track_embeddings[start:start + batch_size],
                dtype=np.float32,
            )
        ).to(device)
        with torch.inference_mode():
            rows.append(model.encode_tracks(batch).cpu().half().numpy())
    return np.concatenate(rows)


def retrieval_metrics(
    model: DualTowerProjection,
    query_cache: QueryCache,
    track_embeddings: np.ndarray,
    valid_tracks: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    projected_tracks = project_corpus(
        model,
        track_embeddings,
        device,
        args.projection_batch_size,
    )
    corpus = torch.from_numpy(projected_tracks).to(
        device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    valid_tensor = torch.from_numpy(np.asarray(valid_tracks)).to(device)
    cutoffs = sorted(set([1, 5, 20, 100, args.eval_topk]))
    ranks: list[int | None] = []

    for start in range(
        0,
        len(query_cache.target_indices),
        args.retrieval_batch_size,
    ):
        base = torch.from_numpy(
            np.asarray(
                query_cache.embeddings[
                    start:start + args.retrieval_batch_size
                ],
                dtype=np.float32,
            )
        ).to(device)
        with torch.inference_mode():
            queries = model.encode_queries(base).to(corpus.dtype)
            scores = queries @ corpus.T
            scores[:, ~valid_tensor] = -torch.inf
            indices = torch.topk(
                scores,
                k=min(args.eval_topk, len(projected_tracks)),
                dim=1,
            ).indices
        for row, candidates in enumerate(indices.detach().cpu().tolist()):
            target = int(query_cache.target_indices[start + row])
            try:
                ranks.append(candidates.index(target) + 1)
            except ValueError:
                ranks.append(None)

    def summarize(indices: list[int]) -> dict[str, float]:
        subset = [ranks[index] for index in indices]
        result = {
            f"recall@{cutoff}": sum(
                rank is not None and rank <= cutoff for rank in subset
            ) / max(1, len(subset))
            for cutoff in cutoffs
        }
        result["mrr"] = sum(
            1.0 / rank if rank is not None else 0.0 for rank in subset
        ) / max(1, len(subset))
        result["ndcg@20"] = sum(
            1.0 / math.log2(rank + 1)
            if rank is not None and rank <= 20
            else 0.0
            for rank in subset
        ) / max(1, len(subset))
        return result

    overall = summarize(list(range(len(ranks))))
    per_turn = {
        str(turn): summarize([
            index
            for index, value in enumerate(query_cache.turns)
            if int(value) == turn
        ])
        for turn in sorted(set(int(value) for value in query_cache.turns))
    }
    denominator = sum(DEFAULT_BLIND_TURN_WEIGHTS.values())
    weighted = {
        metric: sum(
            DEFAULT_BLIND_TURN_WEIGHTS[turn] * per_turn[str(turn)][metric]
            for turn in DEFAULT_BLIND_TURN_WEIGHTS
        ) / denominator
        for metric in overall
    }
    return {
        "overall": overall,
        "per_turn": per_turn,
        "blind_turn_weighted": weighted,
        "ranks": ranks,
        "projected_tracks": projected_tracks,
    }


def save_checkpoint(
    model: DualTowerProjection,
    output_dir: str,
    projected_tracks: np.ndarray,
    track_ids_path: str,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    np.save(
        os.path.join(output_dir, "projected_track_embeddings.npy"),
        projected_tracks.astype(np.float16),
    )
    shutil.copyfile(track_ids_path, os.path.join(output_dir, "track_ids.json"))
    config = {
        "dimension": int(projected_tracks.shape[1]),
        "bottleneck": args.bottleneck,
        "dropout": args.dropout,
        "temperature": args.temperature,
        "embedding_field": args.embedding_field,
        "query_model_name": args.query_model_name,
        "task_instruction": args.task_instruction,
        "max_length": args.max_length,
        "query_mode": args.query_mode,
        "history_turns": args.history_turns,
        "train_track_projection": args.train_track_projection,
        "in_batch_weight": args.in_batch_weight,
        "alignment_weight": args.alignment_weight,
        "best_epoch": epoch,
        "selection_metric": args.selection_metric,
        "selection_score": metrics["blind_turn_weighted"][args.selection_metric],
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    config = OmegaConf.load(f"config/{args.tid}.yaml")
    args.cache_dir = str(config.cache_dir)
    bm25 = BM25_MODEL(
        config.item_db_name,
        config.track_split_types,
        config.corpus_types,
        args.cache_dir,
    )
    track_texts = {
        track_id: track_to_text(metadata)
        for track_id, metadata in bm25.metadata_dict.items()
    }
    dense = QwenDenseRetriever(
        embedding_field=args.embedding_field,
        model_name=args.query_model_name,
        cache_dir=args.cache_dir,
        device=str(device),
        max_length=args.max_length,
        query_batch_size=args.query_batch_size,
        task_instruction=args.task_instruction,
    )
    track_ids = dense.track_ids
    track_to_index = {track_id: index for index, track_id in enumerate(track_ids)}
    track_embeddings = np.asarray(dense.corpus_embeddings, dtype=np.float32)
    valid_tracks = np.linalg.norm(track_embeddings, axis=1) > 0

    train_tasks = select_tasks(args.train_split, args, track_texts)
    dev_tasks = select_tasks(args.dev_split, args, track_texts)
    train_cache = build_query_cache(
        train_tasks,
        args.train_query_cache_dir,
        dense,
        track_to_index,
        args,
    )
    dev_cache = build_query_cache(
        dev_tasks,
        args.dev_query_cache_dir,
        dense,
        track_to_index,
        args,
    )
    dense.unload()

    train_mask = np.asarray([
        target >= 0 and valid_tracks[target]
        for target in train_cache.target_indices
    ])
    if not np.all(train_mask):
        train_cache = QueryCache(
            embeddings=np.asarray(train_cache.embeddings[train_mask]),
            target_indices=np.asarray(train_cache.target_indices[train_mask]),
            turns=np.asarray(train_cache.turns[train_mask]),
            session_ids=[
                value
                for value, keep in zip(train_cache.session_ids, train_mask)
                if keep
            ],
            histories=[
                value
                for value, keep in zip(train_cache.histories, train_mask)
                if keep
            ],
        )

    hard_negatives = build_hard_negative_cache(
        train_cache,
        track_embeddings,
        track_ids,
        args.hard_negative_cache,
        args,
    )
    train_dataset = ContrastiveDataset(
        train_cache.embeddings,
        train_cache.target_indices,
        hard_negatives,
        track_embeddings,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )

    model = DualTowerProjection(
        dimension=track_embeddings.shape[1],
        bottleneck=args.bottleneck,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)
    if not args.train_track_projection:
        model.track_projection.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    baseline = retrieval_metrics(
        model,
        dev_cache,
        track_embeddings,
        valid_tracks,
        device,
        args,
    )
    print(json.dumps({
        "baseline": {
            "overall": baseline["overall"],
            "blind_turn_weighted": baseline["blind_turn_weighted"],
        },
        "train_tasks": len(train_dataset),
        "dev_tasks": len(dev_cache.target_indices),
    }, indent=2))
    best_score = float(
        baseline["blind_turn_weighted"][args.selection_metric]
    )
    save_checkpoint(
        model,
        args.output_dir,
        baseline["projected_tracks"],
        dense.track_ids_path,
        args,
        0,
        baseline,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(loader, desc=f"Training dense epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            loss = contrastive_loss(
                model,
                batch,
                device,
                args.in_batch_weight,
                args.alignment_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite contrastive loss encountered.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = retrieval_metrics(
            model,
            dev_cache,
            track_embeddings,
            valid_tracks,
            device,
            args,
        )
        score = float(metrics["blind_turn_weighted"][args.selection_metric])
        record = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(1, len(losses)),
            "overall": metrics["overall"],
            "blind_turn_weighted": metrics["blind_turn_weighted"],
            "selection_score": score,
        }
        history.append(record)
        print(json.dumps(record, indent=2))
        if score > best_score:
            best_score = score
            save_checkpoint(
                model,
                args.output_dir,
                metrics["projected_tracks"],
                dense.track_ids_path,
                args,
                epoch,
                metrics,
            )

    report = {
        "settings": vars(args),
        "train_tasks": len(train_dataset),
        "dev_tasks": len(dev_cache.target_indices),
        "valid_track_count": int(np.count_nonzero(valid_tracks)),
        "baseline": {
            "overall": baseline["overall"],
            "blind_turn_weighted": baseline["blind_turn_weighted"],
        },
        "history": history,
        "best_score": best_score,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "report.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({
        "output_dir": args.output_dir,
        "best_score": best_score,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tid", default="bm25_tags_doubao_blindset_A")
    parser.add_argument(
        "--dataset_name",
        default="talkpl-ai/TalkPlayData-Challenge-Dataset",
    )
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--dev_split", default="test")
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument(
        "--output_dir",
        default="exp/dense/supervised_qwen_adapter_10k",
    )
    parser.add_argument(
        "--train_query_cache_dir",
        default="cache/dense_queries/train10k_feedback_seed13",
    )
    parser.add_argument(
        "--dev_query_cache_dir",
        default="cache/dense_queries/dev_all_feedback",
    )
    parser.add_argument(
        "--hard_negative_cache",
        default="cache/dense_queries/train10k_feedback_seed13/hard_negatives_32.npy",
    )
    parser.add_argument(
        "--embedding_field",
        default="metadata-qwen3_embedding_0.6b",
    )
    parser.add_argument(
        "--query_model_name",
        default="Qwen/Qwen3-Embedding-0.6B",
    )
    parser.add_argument("--task_instruction", default=DEFAULT_TASK_INSTRUCTION)
    parser.add_argument(
        "--query_mode",
        choices=["feedback", "legacy", "current"],
        default="feedback",
    )
    parser.add_argument("--history_turns", type=int, default=0)
    parser.add_argument("--max_train_tasks", type=int, default=10000)
    parser.add_argument("--max_dev_tasks", type=int, default=None)
    parser.add_argument("--query_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hard_negatives", type=int, default=32)
    parser.add_argument("--hard_negative_search_topk", type=int, default=128)
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--train_track_projection",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--in_batch_weight", type=float, default=0.05)
    parser.add_argument("--alignment_weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--retrieval_batch_size", type=int, default=64)
    parser.add_argument("--projection_batch_size", type=int, default=1024)
    parser.add_argument("--eval_topk", type=int, default=200)
    parser.add_argument(
        "--selection_metric",
        choices=["recall@20", "recall@100", "recall@200", "ndcg@20"],
        default="recall@100",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rebuild_query_cache", action="store_true")
    parser.add_argument("--rebuild_hard_negatives", action="store_true")
    main(parser.parse_args())
