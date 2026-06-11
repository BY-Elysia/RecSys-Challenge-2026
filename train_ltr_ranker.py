"""Train a LambdaRank model over multi-channel music retrieval features."""

import argparse
import json
import math
import os
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
from datasets import load_dataset

from evaluate_final_turn_recall import (
    DEFAULT_BLIND_TURN_WEIGHTS,
    build_metadata_indexes,
    feedback_rich_query,
    filter_seen,
    reciprocal_rank_fusion,
    structural_candidates,
)
from mcrs.retrieval_modules.bm25 import BM25_MODEL
from mcrs.retrieval_modules.precomputed_embeddings import PrecomputedTrackEmbeddingIndex
from mcrs.retrieval_modules.qwen_dense import DEFAULT_TASK_INSTRUCTION, QwenDenseRetriever
from train_reranker import build_query_text, get_turn_target, track_to_text


HISTORY_EMBEDDING_CHANNELS = [
    ("image-siglip2", "last"),
    ("image-siglip2", "mean"),
    ("metadata-qwen3_embedding_0.6b", "last"),
    ("metadata-qwen3_embedding_0.6b", "mean"),
    ("cf-bpr", "last"),
    ("cf-bpr", "mean"),
]
QUERY_DENSE_CHANNEL = "query-qwen3"
USER_CF_CHANNEL = "user-cf"
CHANNEL_NAMES = [
    "bm25_legacy",
    "bm25_feedback",
    "structure",
    QUERY_DENSE_CHANNEL,
    USER_CF_CHANNEL,
    *[f"{field}:{aggregation}" for field, aggregation in HISTORY_EMBEDDING_CHANNELS],
]
SCORED_CHANNELS = [
    QUERY_DENSE_CHANNEL,
    USER_CF_CHANNEL,
    *[f"{field}:{aggregation}" for field, aggregation in HISTORY_EMBEDDING_CHANNELS],
]
GOAL_CATEGORIES = list("ABCDEFGHIJK")
SPECIFICITIES = ["LL", "LH", "HL", "HH"]


@dataclass
class RankingTask:
    session_id: str
    user_id: str
    turn_number: int
    target: str
    history: list[str]
    current_request: str
    legacy_query: str
    feedback_query: str
    goal_category: str
    specificity: str


@dataclass
class FeatureDataset:
    features: np.ndarray
    labels: np.ndarray
    groups: list[int]
    group_task_indices: list[int]
    task_count: int
    task_turns: list[int]
    baseline_ranks: dict[str, list[int | None]]
    feature_names: list[str]


@dataclass
class InferenceFeatureDataset:
    features: np.ndarray
    groups: list[int]
    candidates: list[list[str]]
    feature_names: list[str]


def normalize_text(value: Any) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def metadata_matches(query: str, metadata: dict[str, Any], field: str) -> float:
    values = metadata.get(field, [])
    if not isinstance(values, list):
        values = [values]
    return float(any(
        len(normalized) >= 3 and normalized in query
        for normalized in (normalize_text(value) for value in values)
    ))


def release_year(value: Any) -> float:
    match = re.search(r"\d{4}", str(value))
    return (int(match.group()) - 1900) / 150.0 if match else 0.0


def build_tasks(
    dataset,
    track_texts: dict[str, str],
    history_turns: int,
    turn_mode: str,
) -> list[RankingTask]:
    tasks: list[RankingTask] = []
    for item in dataset:
        target_turns = sorted({
            turn["turn_number"]
            for turn in item["conversations"]
            if turn["role"] == "music"
        })
        if turn_mode == "final":
            target_turns = target_turns[-1:]
        for target_turn in target_turns:
            user_turn = next(
                turn
                for turn in item["conversations"]
                if turn["role"] == "user" and turn["turn_number"] == target_turn
            )
            history = [
                turn["content"]
                for turn in item["conversations"]
                if turn["role"] == "music" and turn["turn_number"] < target_turn
            ]
            tasks.append(RankingTask(
                session_id=item["session_id"],
                user_id=item["user_id"],
                turn_number=target_turn,
                target=get_turn_target(item["conversations"], target_turn),
                history=history,
                current_request=str(user_turn["content"]),
                legacy_query=build_query_text(
                    item,
                    target_turn,
                    track_texts,
                    query_mode="legacy",
                    history_turns=history_turns,
                ),
                feedback_query=feedback_rich_query(
                    item,
                    target_turn,
                    track_texts,
                    history_turns,
                ),
                goal_category=str(item.get("conversation_goal", {}).get("category") or ""),
                specificity=str(item.get("conversation_goal", {}).get("specificity") or ""),
            ))
    return tasks


def load_user_cf_embeddings(dataset_name: str, cache_dir: str) -> dict[str, np.ndarray]:
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    embeddings: dict[str, np.ndarray] = {}
    for split in dataset.values():
        for item in split:
            vector = np.asarray(item["cf-bpr"], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if vector.size == 128 and norm:
                embeddings[item["user_id"]] = vector / norm
    return embeddings


def feature_names() -> list[str]:
    names = []
    for channel in CHANNEL_NAMES:
        safe_channel = channel.replace(":", "__")
        names.extend([f"{safe_channel}__reciprocal_rank", f"{safe_channel}__present"])
    for channel in SCORED_CHANNELS:
        names.append(f"{channel.replace(':', '__')}__score")
    names.extend([
        "same_last_artist",
        "same_any_artist",
        "same_last_album",
        "same_any_album",
        "popularity",
        "release_year",
        "query_track_match",
        "query_artist_match",
        "query_album_match",
        "turn_number",
        "history_size",
        "user_track_cf_cosine",
        "user_track_cf_available",
    ])
    names.extend(f"goal_category__{category}" for category in GOAL_CATEGORIES)
    names.extend(f"specificity__{specificity}" for specificity in SPECIFICITIES)
    return names


def build_candidate_feature_rows(
    task: RankingTask,
    candidates: list[str],
    task_channels: dict[str, list[str]],
    score_maps: dict[str, dict[str, float]],
    bm25: BM25_MODEL,
    user_cf: dict[str, np.ndarray],
    track_cf: PrecomputedTrackEmbeddingIndex,
    names: list[str],
) -> np.ndarray:
    rank_maps = {
        name: {track_id: rank for rank, track_id in enumerate(task_channels[name], start=1)}
        for name in CHANNEL_NAMES
    }
    history_metadata = [
        bm25.metadata_dict[track_id]
        for track_id in task.history
        if track_id in bm25.metadata_dict
    ]
    last_metadata = history_metadata[-1] if history_metadata else {}
    any_artists = {
        str(artist_id)
        for item in history_metadata
        for artist_id in item.get("artist_id", [])
    }
    any_albums = {
        str(album_id)
        for item in history_metadata
        for album_id in item.get("album_id", [])
    }
    last_artists = {str(value) for value in last_metadata.get("artist_id", [])}
    last_albums = {str(value) for value in last_metadata.get("album_id", [])}
    normalized_query = normalize_text(task.current_request)
    user_vector = user_cf.get(task.user_id)

    rows = []
    for track_id in candidates:
        metadata = bm25.metadata_dict[track_id]
        row: list[float] = []
        for name in CHANNEL_NAMES:
            rank = rank_maps[name].get(track_id)
            row.extend([1.0 / rank if rank else 0.0, float(rank is not None)])
        for channel in SCORED_CHANNELS:
            row.append(score_maps[channel].get(track_id, 0.0))

        artists = {str(value) for value in metadata.get("artist_id", [])}
        albums = {str(value) for value in metadata.get("album_id", [])}
        row.extend([
            float(bool(artists & last_artists)),
            float(bool(artists & any_artists)),
            float(bool(albums & last_albums)),
            float(bool(albums & any_albums)),
            float(metadata.get("popularity") or 0) / 100.0,
            release_year(metadata.get("release_date")),
            metadata_matches(normalized_query, metadata, "track_name"),
            metadata_matches(normalized_query, metadata, "artist_name"),
            metadata_matches(normalized_query, metadata, "album_name"),
            task.turn_number / 8.0,
            len(task.history) / 7.0,
        ])

        track_index = track_cf.track_to_index.get(track_id)
        cf_available = (
            user_vector is not None
            and track_index is not None
            and bool(track_cf.valid[track_index])
        )
        cf_score = (
            float(np.dot(user_vector, np.asarray(track_cf.embeddings[track_index], dtype=np.float32)))
            if cf_available
            else 0.0
        )
        row.extend([cf_score, float(cf_available)])
        row.extend(float(task.goal_category == category) for category in GOAL_CATEGORIES)
        row.extend(float(task.specificity == specificity) for specificity in SPECIFICITIES)
        if len(row) != len(names):
            raise RuntimeError(f"Feature length mismatch: {len(row)} != {len(names)}")
        rows.append(row)

    return np.asarray(rows, dtype=np.float32)


def build_channel_candidates(
    tasks: list[RankingTask],
    bm25: BM25_MODEL,
    artist_tracks: dict[str, list[str]],
    album_tracks: dict[str, list[str]],
    channel_topk: int,
    embedding_batch_size: int,
    text_retrieval_batch_size: int,
    user_cf: dict[str, np.ndarray],
    cache_dir: str,
    device: str | None,
    args,
) -> tuple[dict[str, list[list[str]]], dict[str, list[list[float]]]]:
    channels: dict[str, list[list[str]]] = {}
    channel_scores: dict[str, list[list[float]]] = {}
    extra = max((len(task.history) for task in tasks), default=0)

    def retrieve_in_batches(queries: list[str]) -> list[list[str]]:
        batch_size = text_retrieval_batch_size or len(queries)
        results = []
        for start in range(0, len(queries), batch_size):
            results.extend(bm25.batch_text_to_item_retrieval(
                queries[start:start + batch_size],
                topk=channel_topk + extra,
            ))
        return results

    legacy = retrieve_in_batches([task.legacy_query for task in tasks])
    feedback = retrieve_in_batches([task.feedback_query for task in tasks])
    channels["bm25_legacy"] = [
        filter_seen(candidates, task.history, channel_topk)
        for task, candidates in zip(tasks, legacy)
    ]
    channels["bm25_feedback"] = [
        filter_seen(candidates, task.history, channel_topk)
        for task, candidates in zip(tasks, feedback)
    ]
    channels["structure"] = [
        structural_candidates(
            task.history,
            bm25.metadata_dict,
            artist_tracks,
            album_tracks,
            channel_topk,
        )
        for task in tasks
    ]
    channel_scores["bm25_legacy"] = [[] for _ in tasks]
    channel_scores["bm25_feedback"] = [[] for _ in tasks]
    channel_scores["structure"] = [[] for _ in tasks]

    if args.enable_query_dense:
        dense = QwenDenseRetriever(
            embedding_field=args.query_dense_embedding_field,
            model_name=args.query_dense_model_name,
            cache_dir=cache_dir,
            device=device,
            max_length=args.query_dense_max_length,
            query_batch_size=args.query_dense_batch_size,
            task_instruction=args.query_dense_instruction,
        )
        dense_ids, dense_scores = dense.batch_text_to_item_retrieval_with_scores(
            [task.feedback_query for task in tasks],
            topk=channel_topk + extra,
        )
        channels[QUERY_DENSE_CHANNEL] = [
            filter_seen(candidates, task.history, channel_topk)
            for task, candidates in zip(tasks, dense_ids)
        ]
        channel_scores[QUERY_DENSE_CHANNEL] = [
            [
                score
                for track_id, score in zip(candidates, scores)
                if track_id not in set(task.history)
            ][:channel_topk]
            for task, candidates, scores in zip(tasks, dense_ids, dense_scores)
        ]
        dense.unload()
    else:
        channels[QUERY_DENSE_CHANNEL] = [[] for _ in tasks]
        channel_scores[QUERY_DENSE_CHANNEL] = [[] for _ in tasks]

    if args.enable_cf_retrieval:
        cf_index = PrecomputedTrackEmbeddingIndex(
            embedding_field="cf-bpr",
            cache_dir=cache_dir,
            device=device,
        )
        user_cf_ids, user_cf_scores = cf_index.batch_vector_retrieval(
            [user_cf.get(task.user_id) for task in tasks],
            topk=channel_topk,
            batch_size=embedding_batch_size,
            exclude_track_ids=[task.history for task in tasks],
        )
        channels[USER_CF_CHANNEL] = user_cf_ids
        channel_scores[USER_CF_CHANNEL] = user_cf_scores
        cf_index.unload()
    else:
        channels[USER_CF_CHANNEL] = [[] for _ in tasks]
        channel_scores[USER_CF_CHANNEL] = [[] for _ in tasks]

    fields = list(dict.fromkeys(field for field, _ in HISTORY_EMBEDDING_CHANNELS))
    for field in fields:
        if field == "cf-bpr" and not args.enable_cf_retrieval:
            for _, aggregation in [
                pair for pair in HISTORY_EMBEDDING_CHANNELS if pair[0] == field
            ]:
                name = f"{field}:{aggregation}"
                channels[name] = [[] for _ in tasks]
                channel_scores[name] = [[] for _ in tasks]
            continue
        index = PrecomputedTrackEmbeddingIndex(
            embedding_field=field,
            cache_dir=cache_dir,
            device=device,
        )
        for _, aggregation in [pair for pair in HISTORY_EMBEDDING_CHANNELS if pair[0] == field]:
            ids, scores = index.batch_history_retrieval(
                [task.history for task in tasks],
                topk=channel_topk,
                aggregation=aggregation,
                batch_size=embedding_batch_size,
                exclude_seen=True,
            )
            name = f"{field}:{aggregation}"
            channels[name] = ids
            channel_scores[name] = scores
        index.unload()
    return channels, channel_scores


def build_feature_dataset(
    tasks: list[RankingTask],
    bm25: BM25_MODEL,
    artist_tracks: dict[str, list[str]],
    album_tracks: dict[str, list[str]],
    user_cf: dict[str, np.ndarray],
    track_cf: PrecomputedTrackEmbeddingIndex,
    args,
) -> FeatureDataset:
    channels, channel_scores = build_channel_candidates(
        tasks,
        bm25,
        artist_tracks,
        album_tracks,
        args.channel_topk,
        args.embedding_batch_size,
        args.text_retrieval_batch_size,
        user_cf,
        args.cache_dir,
        args.device,
        args,
    )
    names = feature_names()
    feature_groups: list[np.ndarray] = []
    label_groups: list[np.ndarray] = []
    groups: list[int] = []
    group_task_indices: list[int] = []
    baseline_ranks = {name: [] for name in CHANNEL_NAMES + ["rrf_all"]}

    for task_index, task in enumerate(tasks):
        task_channels = {name: channels[name][task_index] for name in CHANNEL_NAMES}
        score_maps = {
            name: dict(zip(channels[name][task_index], channel_scores[name][task_index]))
            for name in CHANNEL_NAMES
        }
        for name in CHANNEL_NAMES:
            try:
                baseline_ranks[name].append(task_channels[name].index(task.target) + 1)
            except ValueError:
                baseline_ranks[name].append(None)
        rrf = reciprocal_rank_fusion(
            [task_channels[name] for name in CHANNEL_NAMES],
            args.channel_topk * len(CHANNEL_NAMES),
            args.rrf_k,
        )
        try:
            baseline_ranks["rrf_all"].append(rrf.index(task.target) + 1)
        except ValueError:
            baseline_ranks["rrf_all"].append(None)

        candidates = list(dict.fromkeys(
            track_id
            for name in CHANNEL_NAMES
            for track_id in task_channels[name]
        ))
        if task.target not in candidates:
            continue

        rows = build_candidate_feature_rows(
            task,
            candidates,
            task_channels,
            score_maps,
            bm25,
            user_cf,
            track_cf,
            names,
        )
        feature_groups.append(rows)
        label_groups.append(np.asarray(
            [int(track_id == task.target) for track_id in candidates],
            dtype=np.int8,
        ))
        groups.append(len(candidates))
        group_task_indices.append(task_index)

    return FeatureDataset(
        features=np.concatenate(feature_groups) if feature_groups else np.empty((0, len(names)), dtype=np.float32),
        labels=np.concatenate(label_groups) if label_groups else np.empty(0, dtype=np.int8),
        groups=groups,
        group_task_indices=group_task_indices,
        task_count=len(tasks),
        task_turns=[task.turn_number for task in tasks],
        baseline_ranks=baseline_ranks,
        feature_names=names,
    )


def build_inference_feature_dataset(
    tasks: list[RankingTask],
    bm25: BM25_MODEL,
    artist_tracks: dict[str, list[str]],
    album_tracks: dict[str, list[str]],
    user_cf: dict[str, np.ndarray],
    track_cf: PrecomputedTrackEmbeddingIndex,
    args,
) -> InferenceFeatureDataset:
    channels, channel_scores = build_channel_candidates(
        tasks,
        bm25,
        artist_tracks,
        album_tracks,
        args.channel_topk,
        args.embedding_batch_size,
        args.text_retrieval_batch_size,
        user_cf,
        args.cache_dir,
        args.device,
        args,
    )
    names = feature_names()
    feature_groups: list[np.ndarray] = []
    groups: list[int] = []
    candidates_by_task: list[list[str]] = []

    for task_index, task in enumerate(tasks):
        task_channels = {name: channels[name][task_index] for name in CHANNEL_NAMES}
        score_maps = {
            name: dict(zip(channels[name][task_index], channel_scores[name][task_index]))
            for name in CHANNEL_NAMES
        }
        candidates = list(dict.fromkeys(
            track_id
            for name in CHANNEL_NAMES
            for track_id in task_channels[name]
        ))
        if not candidates:
            raise RuntimeError(f"No candidates for session {task.session_id}.")
        feature_groups.append(build_candidate_feature_rows(
            task,
            candidates,
            task_channels,
            score_maps,
            bm25,
            user_cf,
            track_cf,
            names,
        ))
        groups.append(len(candidates))
        candidates_by_task.append(candidates)

    return InferenceFeatureDataset(
        features=np.concatenate(feature_groups),
        groups=groups,
        candidates=candidates_by_task,
        feature_names=names,
    )


def save_feature_dataset(dataset: FeatureDataset, cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "features.npy"), dataset.features)
    np.save(os.path.join(cache_dir, "labels.npy"), dataset.labels)
    metadata = {
        "format_version": 1,
        "groups": dataset.groups,
        "group_task_indices": dataset.group_task_indices,
        "task_count": dataset.task_count,
        "task_turns": dataset.task_turns,
        "baseline_ranks": dataset.baseline_ranks,
        "feature_names": dataset.feature_names,
        "feature_shape": list(dataset.features.shape),
        "label_shape": list(dataset.labels.shape),
    }
    with open(os.path.join(cache_dir, "metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False)


def load_feature_dataset(cache_dir: str, mmap_mode: str | None = None) -> FeatureDataset:
    with open(os.path.join(cache_dir, "metadata.json"), encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata.get("format_version") != 1:
        raise ValueError(f"Unsupported feature cache version in {cache_dir}.")
    features = np.load(os.path.join(cache_dir, "features.npy"), mmap_mode=mmap_mode)
    labels = np.load(os.path.join(cache_dir, "labels.npy"), mmap_mode=mmap_mode)
    if list(features.shape) != metadata["feature_shape"]:
        raise ValueError(f"Feature shape mismatch in {cache_dir}.")
    if list(labels.shape) != metadata["label_shape"]:
        raise ValueError(f"Label shape mismatch in {cache_dir}.")
    return FeatureDataset(
        features=features,
        labels=labels,
        groups=[int(value) for value in metadata["groups"]],
        group_task_indices=[int(value) for value in metadata["group_task_indices"]],
        task_count=int(metadata["task_count"]),
        task_turns=[int(value) for value in metadata["task_turns"]],
        baseline_ranks=metadata["baseline_ranks"],
        feature_names=metadata["feature_names"],
    )


def ranks_from_predictions(dataset: FeatureDataset, predictions: np.ndarray) -> list[int | None]:
    ranks: list[int | None] = [None] * dataset.task_count
    offset = 0
    for group_size, task_index in zip(dataset.groups, dataset.group_task_indices):
        group_predictions = predictions[offset:offset + group_size]
        group_labels = dataset.labels[offset:offset + group_size]
        positive_indices = np.flatnonzero(group_labels)
        if len(positive_indices):
            positive_index = int(positive_indices[0])
            positive_score = group_predictions[positive_index]
            ranks[task_index] = int(np.count_nonzero(group_predictions > positive_score) + 1)
        offset += group_size
    return ranks


def summarize_ranks(ranks: list[int | None], turns: list[int]) -> dict[str, Any]:
    def summarize(indices: list[int]) -> dict[str, float | int]:
        selected = [ranks[index] for index in indices]
        return {
            "turns": len(indices),
            "candidate_recall": sum(rank is not None for rank in selected) / max(1, len(selected)),
            "recall@1": sum(rank == 1 for rank in selected) / max(1, len(selected)),
            "recall@5": sum(rank is not None and rank <= 5 for rank in selected) / max(1, len(selected)),
            "recall@20": sum(rank is not None and rank <= 20 for rank in selected) / max(1, len(selected)),
            "ndcg@20": sum(
                1.0 / math.log2(rank + 1) if rank is not None and rank <= 20 else 0.0
                for rank in selected
            ) / max(1, len(selected)),
        }

    result = {
        "overall": summarize(list(range(len(ranks)))),
        "turn1": summarize([index for index, turn in enumerate(turns) if turn == 1]),
        "turn2plus": summarize([index for index, turn in enumerate(turns) if turn >= 2]),
        "per_turn": {
            str(turn): summarize([index for index, value in enumerate(turns) if value == turn])
            for turn in sorted(set(turns))
        },
    }
    if set(DEFAULT_BLIND_TURN_WEIGHTS).issubset(set(turns)):
        per_turn = result["per_turn"]
        denominator = sum(DEFAULT_BLIND_TURN_WEIGHTS.values())
        result["blind_turn_weighted"] = {
            metric: sum(
                DEFAULT_BLIND_TURN_WEIGHTS[turn] * float(per_turn[str(turn)][metric])
                for turn in DEFAULT_BLIND_TURN_WEIGHTS
            ) / denominator
            for metric in ["candidate_recall", "recall@1", "recall@5", "recall@20", "ndcg@20"]
        }
    return result


def main(args) -> None:
    rng = random.Random(args.seed)
    dev = load_dataset(args.dataset_name, split="test", cache_dir=args.cache_dir)
    if args.max_dev_sessions:
        dev = dev.select(range(min(args.max_dev_sessions, len(dev))))

    bm25 = BM25_MODEL(
        args.track_metadata_name,
        ["all_tracks"],
        args.corpus_types,
        args.cache_dir,
    )
    track_texts = {track_id: track_to_text(item) for track_id, item in bm25.metadata_dict.items()}
    artist_tracks, album_tracks = build_metadata_indexes(bm25.metadata_dict)
    user_cf = load_user_cf_embeddings(args.user_embedding_name, args.cache_dir)
    track_cf = PrecomputedTrackEmbeddingIndex(
        embedding_field="cf-bpr",
        cache_dir=args.cache_dir,
        device="cpu",
    )

    dev_tasks = build_tasks(dev, track_texts, args.history_turns, args.dev_turn_mode)

    train_features = None
    if not args.model_path:
        if args.train_feature_cache_dir and os.path.exists(
            os.path.join(args.train_feature_cache_dir, "metadata.json")
        ):
            print(f"Loading train features from {args.train_feature_cache_dir}...")
            train_features = load_feature_dataset(args.train_feature_cache_dir)
        else:
            train = load_dataset(args.dataset_name, split="train", cache_dir=args.cache_dir)
            if args.max_train_sessions:
                indices = list(range(len(train)))
                rng.shuffle(indices)
                train = train.select(indices[:min(args.max_train_sessions, len(train))])
            train_tasks = build_tasks(train, track_texts, args.history_turns, args.train_turn_mode)
            rng.shuffle(train_tasks)
            if args.max_train_tasks:
                train_tasks = train_tasks[:min(args.max_train_tasks, len(train_tasks))]
            print(f"Building train features for {len(train_tasks)} tasks...")
            train_features = build_feature_dataset(
                train_tasks,
                bm25,
                artist_tracks,
                album_tracks,
                user_cf,
                track_cf,
                args,
            )
            if args.train_feature_cache_dir:
                print(f"Saving train features to {args.train_feature_cache_dir}...")
                save_feature_dataset(train_features, args.train_feature_cache_dir)

    if args.dev_feature_cache_dir and os.path.exists(
        os.path.join(args.dev_feature_cache_dir, "metadata.json")
    ):
        print(f"Loading dev features from {args.dev_feature_cache_dir}...")
        dev_features = load_feature_dataset(args.dev_feature_cache_dir)
    else:
        print(f"Building dev features for {len(dev_tasks)} tasks...")
        dev_features = build_feature_dataset(
            dev_tasks,
            bm25,
            artist_tracks,
            album_tracks,
            user_cf,
            track_cf,
            args,
        )
        if args.dev_feature_cache_dir:
            print(f"Saving dev features to {args.dev_feature_cache_dir}...")
            save_feature_dataset(dev_features, args.dev_feature_cache_dir)
    if not dev_features.groups:
        raise RuntimeError("No evaluation ranking groups were built.")
    if args.cache_only:
        print(json.dumps({
            "train": (
                {
                    "tasks": train_features.task_count,
                    "retrieved_groups": len(train_features.groups),
                    "rows": int(len(train_features.labels)),
                }
                if train_features is not None
                else None
            ),
            "dev": {
                "tasks": dev_features.task_count,
                "retrieved_groups": len(dev_features.groups),
                "rows": int(len(dev_features.labels)),
            },
            "train_feature_cache_dir": args.train_feature_cache_dir,
            "dev_feature_cache_dir": args.dev_feature_cache_dir,
        }, indent=2))
        return

    if args.model_path:
        model = lgb.Booster(model_file=args.model_path)
        best_iteration = (
            model.best_iteration
            if model.best_iteration > 0
            else model.current_iteration()
        )
    else:
        if train_features is None or not train_features.groups:
            raise RuntimeError("No trainable ranking groups were built.")
        train_set = lgb.Dataset(
            train_features.features,
            label=train_features.labels,
            group=train_features.groups,
            feature_name=train_features.feature_names,
            free_raw_data=False,
        )
        dev_set = lgb.Dataset(
            dev_features.features,
            label=dev_features.labels,
            group=dev_features.groups,
            feature_name=dev_features.feature_names,
            reference=train_set,
            free_raw_data=False,
        )
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [1, 5, 20],
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "min_data_in_leaf": args.min_data_in_leaf,
            "feature_fraction": args.feature_fraction,
            "bagging_fraction": args.bagging_fraction,
            "bagging_freq": 1,
            "label_gain": [0, 1],
            "seed": args.seed,
            "verbosity": -1,
            "num_threads": args.num_threads,
        }
        model = lgb.train(
            params,
            train_set,
            num_boost_round=args.num_boost_round,
            valid_sets=[dev_set],
            valid_names=["dev"],
            callbacks=[
                lgb.early_stopping(args.early_stopping_rounds),
                lgb.log_evaluation(args.log_period),
            ],
        )
        best_iteration = model.best_iteration

    predictions = model.predict(
        dev_features.features,
        num_iteration=best_iteration,
    )
    ltr_ranks = ranks_from_predictions(dev_features, predictions)
    report = {
        "settings": vars(args),
        "train": (
            {
                "tasks": train_features.task_count,
                "retrieved_groups": len(train_features.groups),
                "rows": int(len(train_features.labels)),
            }
            if train_features is not None
            else None
        ),
        "dev": {
            "tasks": dev_features.task_count,
            "retrieved_groups": len(dev_features.groups),
            "rows": int(len(dev_features.labels)),
        },
        "best_iteration": best_iteration,
        "ltr": summarize_ranks(ltr_ranks, dev_features.task_turns),
        "baselines": {
            name: summarize_ranks(ranks, dev_features.task_turns)
            for name, ranks in dev_features.baseline_ranks.items()
        },
        "feature_importance_gain": dict(sorted(
            zip(
                dev_features.feature_names,
                model.feature_importance(importance_type="gain").tolist(),
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "model.txt")
    report_path = os.path.join(args.output_dir, "report.json")
    feature_path = os.path.join(args.output_dir, "feature_names.json")
    model.save_model(model_path, num_iteration=best_iteration)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    with open(feature_path, "w", encoding="utf-8") as file:
        json.dump(dev_features.feature_names, file, indent=2, ensure_ascii=False)
    print(json.dumps({
        "best_iteration": best_iteration,
        "ltr": report["ltr"],
        "baselines": {
            name: value["overall"]
            for name, value in report["baselines"].items()
        },
        "top_features": list(report["feature_importance_gain"].items())[:15],
        "model_path": model_path,
        "report_path": report_path,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multi-channel LambdaRank reranker.")
    parser.add_argument("--dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--track_metadata_name", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--user_embedding_name", default="talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument("--output_dir", default="exp/ltr/multichannel_v1")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--train_feature_cache_dir", default=None)
    parser.add_argument("--dev_feature_cache_dir", default=None)
    parser.add_argument("--cache_only", action="store_true")
    parser.add_argument(
        "--corpus_types",
        nargs="+",
        default=["track_name", "artist_name", "album_name", "release_date", "tag_list"],
    )
    parser.add_argument("--channel_topk", type=int, default=50)
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--text_retrieval_batch_size", type=int, default=5000)
    parser.add_argument(
        "--enable_query_dense",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--enable_cf_retrieval",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--query_dense_embedding_field",
        default="metadata-qwen3_embedding_0.6b",
    )
    parser.add_argument("--query_dense_model_name", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--query_dense_max_length", type=int, default=512)
    parser.add_argument("--query_dense_batch_size", type=int, default=16)
    parser.add_argument("--query_dense_instruction", default=DEFAULT_TASK_INSTRUCTION)
    parser.add_argument("--history_turns", type=int, default=0)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--train_turn_mode", choices=["all", "final"], default="all")
    parser.add_argument("--dev_turn_mode", choices=["all", "final"], default="final")
    parser.add_argument("--max_train_sessions", type=int, default=None)
    parser.add_argument("--max_train_tasks", type=int, default=10000)
    parser.add_argument("--max_dev_sessions", type=int, default=None)
    parser.add_argument("--num_boost_round", type=int, default=500)
    parser.add_argument("--early_stopping_rounds", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--min_data_in_leaf", type=int, default=50)
    parser.add_argument("--feature_fraction", type=float, default=0.9)
    parser.add_argument("--bagging_fraction", type=float, default=0.9)
    parser.add_argument("--num_threads", type=int, default=0)
    parser.add_argument("--log_period", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None)
    main(parser.parse_args())
