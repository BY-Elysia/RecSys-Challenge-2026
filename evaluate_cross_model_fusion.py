"""Evaluate fusion between models with different feature sets on one LTR cache."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from typing import Any

import lightgbm as lgb
import numpy as np

from run_inference_ltr_blindset import load_removed_channels
from train_ltr_ranker import FeatureDataset, load_feature_dataset, summarize_ranks


def model_active_rows(
    dataset: FeatureDataset,
    model_feature_names: list[str],
    removed_channels: list[str],
) -> np.ndarray:
    channels = [
        name.removesuffix("__present")
        for name in model_feature_names
        if name.endswith("__present")
        and name.removesuffix("__present") not in removed_channels
    ]
    columns = [dataset.feature_names.index(f"{channel}__present") for channel in channels]
    return np.any(dataset.features[:, columns] > 0, axis=1)


def masked_groupwise_scores(
    values: np.ndarray,
    active: np.ndarray,
    groups: list[int],
    mode: str,
    rrf_k: float,
) -> np.ndarray:
    normalized = np.zeros(len(values), dtype=np.float64)
    offset = 0
    for group_size in groups:
        group_values = values[offset:offset + group_size]
        group_active = active[offset:offset + group_size]
        active_indices = np.flatnonzero(group_active)
        if active_indices.size:
            selected = group_values[active_indices]
            if mode == "minmax":
                low = float(np.min(selected))
                high = float(np.max(selected))
                scores = (selected - low) / (high - low) if high > low else np.ones_like(selected)
            elif mode == "rrf":
                order = np.argsort(-selected, kind="stable")
                ranks = np.empty(len(selected), dtype=np.int32)
                ranks[order] = np.arange(1, len(selected) + 1)
                scores = 1.0 / (rrf_k + ranks)
            else:
                raise ValueError(mode)
            normalized[offset + active_indices] = scores
        offset += group_size
    return normalized


def ranks_with_locked_anchor_prefix(
    dataset: FeatureDataset,
    anchor_scores: np.ndarray,
    anchor_active: np.ndarray,
    fused_scores: np.ndarray,
    union_active: np.ndarray,
    lock_prefix: int,
) -> list[int | None]:
    ranks: list[int | None] = [None] * dataset.task_count
    offset = 0
    for group_size, task_index in zip(dataset.groups, dataset.group_task_indices):
        labels = dataset.labels[offset:offset + group_size]
        positive = np.flatnonzero(labels)
        if not positive.size:
            offset += group_size
            continue
        positive_index = int(positive[0])
        group_anchor = anchor_scores[offset:offset + group_size]
        group_anchor_active = anchor_active[offset:offset + group_size]
        group_fused = fused_scores[offset:offset + group_size]
        group_union_active = union_active[offset:offset + group_size]

        anchor_indices = np.flatnonzero(group_anchor_active)
        anchor_order = anchor_indices[
            np.argsort(-group_anchor[anchor_indices], kind="stable")
        ]
        locked = set(int(index) for index in anchor_order[:lock_prefix])
        if positive_index in locked:
            ranks[task_index] = list(anchor_order[:lock_prefix]).index(positive_index) + 1
        elif group_union_active[positive_index]:
            eligible = group_union_active.copy()
            if locked:
                eligible[list(locked)] = False
            higher = np.count_nonzero(eligible & (group_fused > group_fused[positive_index]))
            ranks[task_index] = int(lock_prefix + higher + 1)
        offset += group_size
    return ranks


def compact(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "overall_ndcg@20": metrics["overall"]["ndcg@20"],
        "turn1_ndcg@20": metrics["turn1"]["ndcg@20"],
        "turn2plus_ndcg@20": metrics["turn2plus"]["ndcg@20"],
        "blind_weighted_ndcg@20": metrics["blind_turn_weighted"]["ndcg@20"],
        "blind_weighted_recall@20": metrics["blind_turn_weighted"]["recall@20"],
    }


def load_model_predictions(
    dataset: FeatureDataset,
    spec: str,
) -> tuple[str, dict[str, Any], np.ndarray, np.ndarray]:
    name, path = spec.split("=", 1)
    model = lgb.Booster(model_file=path)
    feature_names = model.feature_name()
    missing = [feature for feature in feature_names if feature not in dataset.feature_names]
    if missing:
        raise RuntimeError(f"{name} is missing cache features: {missing}")
    indices = [dataset.feature_names.index(feature) for feature in feature_names]
    iteration = model.best_iteration if model.best_iteration > 0 else model.current_iteration()
    predictions = model.predict(dataset.features[:, indices], num_iteration=iteration)
    removed_channels = load_removed_channels(path)
    active = model_active_rows(dataset, feature_names, removed_channels)
    metadata = {
        "path": path,
        "iteration": iteration,
        "feature_count": len(feature_names),
        "removed_channels": removed_channels,
        "active_fraction": float(np.mean(active)),
    }
    return name, metadata, predictions, active


def main(args: argparse.Namespace) -> None:
    dataset = load_feature_dataset(args.dev_feature_cache_dir, mmap_mode="r")
    models = {}
    raw_predictions = {}
    active_rows = {}
    for spec in args.models:
        name, metadata, predictions, active = load_model_predictions(dataset, spec)
        models[name] = metadata
        raw_predictions[name] = predictions
        active_rows[name] = active
    if args.anchor_model not in models:
        raise RuntimeError(f"Anchor model {args.anchor_model} is not loaded.")

    normalized = {
        name: masked_groupwise_scores(
            raw_predictions[name],
            active_rows[name],
            dataset.groups,
            args.score_mode,
            args.rrf_k,
        )
        for name in models
    }
    union_active = np.any(np.stack(list(active_rows.values())), axis=0)
    other_names = [name for name in models if name != args.anchor_model]
    ranked = []
    methods = {}
    for weights in itertools.product(args.weights, repeat=len(other_names)):
        fused = args.anchor_weight * normalized[args.anchor_model].copy()
        for name, weight in zip(other_names, weights):
            fused += weight * normalized[name]
        ranks = ranks_with_locked_anchor_prefix(
            dataset,
            raw_predictions[args.anchor_model],
            active_rows[args.anchor_model],
            fused,
            union_active,
            args.lock_prefix,
        )
        metrics = summarize_ranks(ranks, dataset.task_turns)
        method = (
            f"{args.score_mode}_anchor{args.anchor_weight:g}_lock{args.lock_prefix}"
            + "".join(f"__{name}{weight:g}" for name, weight in zip(other_names, weights))
        )
        methods[method] = metrics
        ranked.append({"method": method, **compact(metrics)})
    ranked.sort(
        key=lambda item: (item["blind_weighted_ndcg@20"], item["overall_ndcg@20"]),
        reverse=True,
    )
    report = {
        "settings": vars(args),
        "models": models,
        "ranking": ranked,
        "methods": methods,
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({"top": ranked[:20], "output_path": args.output_path}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev_feature_cache_dir", default="cache/ltr/dev_all_top100_cf_v2")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--anchor_model", required=True)
    parser.add_argument("--anchor_weight", type=float, default=1.0)
    parser.add_argument("--weights", nargs="+", type=float, default=[0, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 1.5, 2])
    parser.add_argument("--score_mode", choices=["rrf", "minmax"], default="rrf")
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument("--lock_prefix", type=int, default=1)
    parser.add_argument("--output_path", default="exp/ltr/cross_model_fusion/report.json")
    main(parser.parse_args())
