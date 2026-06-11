"""Evaluate high-precision constraint-gated model replacement on Dev."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter

import lightgbm as lgb
import numpy as np
from datasets import load_dataset

from constraint_gating import CONSTRAINT_CATEGORIES, strict_constraint_categories
from evaluate_cross_model_fusion import model_active_rows
from run_inference_ltr_blindset import load_removed_channels
from train_ltr_ranker import FeatureDataset, load_feature_dataset, summarize_ranks


def dev_requests(dataset) -> list[str]:
    requests = []
    for item in dataset:
        target_turns = sorted({
            turn["turn_number"]
            for turn in item["conversations"]
            if turn["role"] == "music"
        })
        for target_turn in target_turns:
            user_turn = next(
                turn
                for turn in item["conversations"]
                if turn["role"] == "user" and turn["turn_number"] == target_turn
            )
            requests.append(str(user_turn["content"]))
    return requests


def load_predictions(
    dataset: FeatureDataset,
    model_path: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    model = lgb.Booster(model_file=model_path)
    names = model.feature_name()
    missing = [name for name in names if name not in dataset.feature_names]
    if missing:
        raise RuntimeError(f"Model is missing cached features: {missing}")
    indices = [dataset.feature_names.index(name) for name in names]
    iteration = model.best_iteration if model.best_iteration > 0 else model.current_iteration()
    predictions = model.predict(dataset.features[:, indices], num_iteration=iteration)
    removed = load_removed_channels(model_path)
    active = model_active_rows(dataset, names, removed)
    return predictions, active, {
        "path": model_path,
        "iteration": iteration,
        "features": len(names),
        "removed_channels": removed,
    }


def gated_ranks(
    dataset: FeatureDataset,
    anchor_scores: np.ndarray,
    anchor_active: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_active: np.ndarray,
    selected_tasks: set[int],
    lock_prefix: int,
) -> list[int | None]:
    ranks: list[int | None] = [None] * dataset.task_count
    offset = 0
    for group_size, task_index in zip(dataset.groups, dataset.group_task_indices):
        labels = dataset.labels[offset:offset + group_size]
        positive_indices = np.flatnonzero(labels)
        if not positive_indices.size:
            offset += group_size
            continue
        positive = int(positive_indices[0])
        anchor_group = anchor_scores[offset:offset + group_size]
        anchor_indices = np.flatnonzero(anchor_active[offset:offset + group_size])
        anchor_order = anchor_indices[np.argsort(-anchor_group[anchor_indices], kind="stable")]

        if task_index not in selected_tasks:
            order = anchor_order
        else:
            candidate_group = candidate_scores[offset:offset + group_size]
            candidate_indices = np.flatnonzero(candidate_active[offset:offset + group_size])
            candidate_order = candidate_indices[
                np.argsort(-candidate_group[candidate_indices], kind="stable")
            ]
            locked = list(anchor_order[:lock_prefix])
            order = locked + [
                int(index)
                for index in candidate_order
                if int(index) not in locked
            ]
            order.extend(
                int(index)
                for index in anchor_order
                if int(index) not in set(order)
            )
        matches = np.flatnonzero(np.asarray(order) == positive)
        ranks[task_index] = int(matches[0] + 1) if matches.size else None
        offset += group_size
    return ranks


def ndcg_delta(candidate: list[int | None], anchor: list[int | None], indices: set[int]) -> float:
    def score(rank: int | None) -> float:
        return 1.0 / math.log2(rank + 1) if rank is not None and rank <= 20 else 0.0

    return sum(score(candidate[index]) - score(anchor[index]) for index in indices) / max(1, len(indices))


def compact(metrics: dict) -> dict[str, float]:
    return {
        "overall_ndcg@20": metrics["overall"]["ndcg@20"],
        "turn1_ndcg@20": metrics["turn1"]["ndcg@20"],
        "turn2plus_ndcg@20": metrics["turn2plus"]["ndcg@20"],
        "blind_weighted_ndcg@20": metrics["blind_turn_weighted"]["ndcg@20"],
    }


def main(args: argparse.Namespace) -> None:
    cache = load_feature_dataset(args.dev_feature_cache_dir, mmap_mode="r")
    raw_dataset = load_dataset(args.dataset_name, split="test", cache_dir=args.cache_dir)
    requests = dev_requests(raw_dataset)
    if len(requests) != cache.task_count:
        raise RuntimeError(f"Request/task mismatch: {len(requests)} != {cache.task_count}")
    categories = [strict_constraint_categories(request) for request in requests]

    anchor_scores, anchor_active, anchor_meta = load_predictions(cache, args.anchor_model_path)
    candidate_scores, candidate_active, candidate_meta = load_predictions(
        cache, args.candidate_model_path
    )
    no_gate = gated_ranks(
        cache, anchor_scores, anchor_active, candidate_scores, candidate_active, set(), 0
    )
    reports = []
    for lock_prefix in args.lock_prefixes:
        selections = {
            category: {index for index, values in enumerate(categories) if category in values}
            for category in CONSTRAINT_CATEGORIES
        }
        selections["all_strict"] = {
            index for index, values in enumerate(categories) if values
        }
        for name, selected in selections.items():
            ranks = gated_ranks(
                cache,
                anchor_scores,
                anchor_active,
                candidate_scores,
                candidate_active,
                selected,
                lock_prefix,
            )
            metrics = summarize_ranks(ranks, cache.task_turns)
            reports.append({
                "gate": name,
                "lock_prefix": lock_prefix,
                "selected_tasks": len(selected),
                "changed_rank_tasks": sum(ranks[index] != no_gate[index] for index in selected),
                "selected_mean_ndcg_delta": ndcg_delta(ranks, no_gate, selected),
                **compact(metrics),
            })

    reports.sort(
        key=lambda value: (
            value["blind_weighted_ndcg@20"],
            value["overall_ndcg@20"],
        ),
        reverse=True,
    )
    report = {
        "anchor": anchor_meta,
        "candidate": candidate_meta,
        "category_task_counts": Counter(
            category for values in categories for category in values
        ),
        "anchor_metrics": compact(summarize_ranks(no_gate, cache.task_turns)),
        "results": reports,
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument("--dev_feature_cache_dir", required=True)
    parser.add_argument("--anchor_model_path", required=True)
    parser.add_argument("--candidate_model_path", required=True)
    parser.add_argument("--lock_prefixes", nargs="+", type=int, default=[1, 5])
    parser.add_argument(
        "--output_path",
        default="exp/ltr/constraint_gating/report.json",
    )
    main(parser.parse_args())
