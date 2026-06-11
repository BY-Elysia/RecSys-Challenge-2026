"""Evaluate score fusion while keeping the v2 top-1 item fixed."""

from __future__ import annotations

import argparse
import itertools
import json
import os

import lightgbm as lgb
import numpy as np

from train_ltr_cached_ablation import VARIANTS, kept_feature_indices, prepare_dataset
from train_ltr_ranker import load_feature_dataset, summarize_ranks


def groupwise_minmax(values: np.ndarray, groups: list[int]) -> np.ndarray:
    normalized = np.empty_like(values, dtype=np.float64)
    offset = 0
    for group_size in groups:
        group = values[offset:offset + group_size]
        low = float(np.min(group))
        high = float(np.max(group))
        normalized[offset:offset + group_size] = (
            (group - low) / (high - low)
            if high > low
            else 0.0
        )
        offset += group_size
    return normalized


def groupwise_rrf(values: np.ndarray, groups: list[int], k: float = 60.0) -> np.ndarray:
    scores = np.empty_like(values, dtype=np.float64)
    offset = 0
    for group_size in groups:
        group = values[offset:offset + group_size]
        order = np.argsort(-group, kind="stable")
        ranks = np.empty(group_size, dtype=np.int32)
        ranks[order] = np.arange(1, group_size + 1)
        scores[offset:offset + group_size] = 1.0 / (k + ranks)
        offset += group_size
    return scores


def ranks_locked_top1(
    labels: np.ndarray,
    groups: list[int],
    group_task_indices: list[int],
    anchor_scores: np.ndarray,
    fused_scores: np.ndarray,
    task_count: int,
) -> list[int | None]:
    ranks: list[int | None] = [None] * task_count
    offset = 0
    for group_size, task_index in zip(groups, group_task_indices):
        group_anchor = anchor_scores[offset:offset + group_size]
        group_fused = fused_scores[offset:offset + group_size]
        group_labels = labels[offset:offset + group_size]
        positive = np.flatnonzero(group_labels)
        if positive.size == 0:
            offset += group_size
            continue
        positive_index = int(positive[0])
        anchor_top1 = int(np.argmax(group_anchor))
        if positive_index == anchor_top1:
            ranks[task_index] = 1
        else:
            positive_score = group_fused[positive_index]
            eligible = np.ones(group_size, dtype=bool)
            eligible[anchor_top1] = False
            higher = np.count_nonzero(eligible & (group_fused > positive_score))
            ranks[task_index] = int(higher + 2)
        offset += group_size
    return ranks


def ranks_locked_top1_anchor_top20(
    labels: np.ndarray,
    groups: list[int],
    group_task_indices: list[int],
    anchor_scores: np.ndarray,
    fused_scores: np.ndarray,
    task_count: int,
) -> list[int | None]:
    ranks: list[int | None] = [None] * task_count
    offset = 0
    for group_size, task_index in zip(groups, group_task_indices):
        group_anchor = anchor_scores[offset:offset + group_size]
        group_fused = fused_scores[offset:offset + group_size]
        group_labels = labels[offset:offset + group_size]
        positive = np.flatnonzero(group_labels)
        if positive.size == 0:
            offset += group_size
            continue
        positive_index = int(positive[0])
        anchor_order = np.argsort(-group_anchor, kind="stable")
        anchor_top20 = set(int(index) for index in anchor_order[:20])
        if positive_index not in anchor_top20:
            offset += group_size
            continue
        anchor_top1 = int(anchor_order[0])
        if positive_index == anchor_top1:
            ranks[task_index] = 1
        else:
            positive_score = group_fused[positive_index]
            eligible = np.zeros(group_size, dtype=bool)
            eligible[list(anchor_top20)] = True
            eligible[anchor_top1] = False
            higher = np.count_nonzero(eligible & (group_fused > positive_score))
            ranks[task_index] = int(higher + 2)
        offset += group_size
    return ranks


def compact(metrics: dict) -> dict:
    return {
        "overall_ndcg@20": metrics["overall"]["ndcg@20"],
        "overall_recall@20": metrics["overall"]["recall@20"],
        "turn1_ndcg@20": metrics["turn1"]["ndcg@20"],
        "turn2plus_ndcg@20": metrics["turn2plus"]["ndcg@20"],
        "blind_weighted_ndcg@20": metrics["blind_turn_weighted"]["ndcg@20"],
        "blind_weighted_recall@20": metrics["blind_turn_weighted"]["recall@20"],
    }


def load_predictions(dataset, model_specs: list[str]) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    models = {}
    predictions = {}
    for spec in model_specs:
        name, path = spec.split("=", 1)
        model = lgb.Booster(model_file=path)
        if model.feature_name() != dataset.feature_names:
            raise RuntimeError(f"Feature mismatch for model {name}.")
        iteration = model.best_iteration if model.best_iteration > 0 else model.current_iteration()
        models[name] = {"path": path, "iteration": iteration}
        predictions[name] = model.predict(dataset.features, num_iteration=iteration)
    return models, predictions


def main(args: argparse.Namespace) -> None:
    cache = load_feature_dataset(args.dev_feature_cache_dir, mmap_mode="r")
    definition = VARIANTS[args.variant]
    feature_indices = kept_feature_indices(
        cache.feature_names,
        definition["remove_prefixes"],
    )
    dataset = prepare_dataset(
        cache,
        feature_indices,
        definition["remove_channels"],
        require_positive=False,
    )
    models, raw_predictions = load_predictions(dataset, args.models)
    if args.anchor_model not in raw_predictions:
        raise RuntimeError(f"Anchor model {args.anchor_model} was not loaded.")

    if args.score_mode == "minmax":
        normalized = {
            name: groupwise_minmax(values, dataset.groups)
            for name, values in raw_predictions.items()
        }
    elif args.score_mode == "rrf":
        normalized = {
            name: groupwise_rrf(values, dataset.groups, args.rrf_k)
            for name, values in raw_predictions.items()
        }
    else:
        raise ValueError(args.score_mode)

    names = [name for name in raw_predictions if name != args.anchor_model]
    weight_values = [float(value) for value in args.weights]
    methods = {}
    tested = 0
    for weights in itertools.product(weight_values, repeat=len(names)):
        fused = args.anchor_weight * normalized[args.anchor_model].copy()
        for name, weight in zip(names, weights):
            fused += weight * normalized[name]
        if args.candidate_pool == "all":
            ranks = ranks_locked_top1(
                dataset.labels,
                dataset.groups,
                dataset.group_task_indices,
                raw_predictions[args.anchor_model],
                fused,
                dataset.task_count,
            )
        elif args.candidate_pool == "anchor_top20":
            ranks = ranks_locked_top1_anchor_top20(
                dataset.labels,
                dataset.groups,
                dataset.group_task_indices,
                raw_predictions[args.anchor_model],
                fused,
                dataset.task_count,
            )
        else:
            raise ValueError(args.candidate_pool)
        method = (
            f"locked_{args.anchor_model}_{args.candidate_pool}_{args.score_mode}"
            + "".join(f"__{name}{weight:g}" for name, weight in zip(names, weights))
        )
        methods[method] = summarize_ranks(ranks, dataset.task_turns)
        tested += 1

    ranked = sorted(
        ({"method": name, **compact(metrics)} for name, metrics in methods.items()),
        key=lambda item: (
            item["blind_weighted_ndcg@20"],
            item["overall_ndcg@20"],
        ),
        reverse=True,
    )
    report = {
        "models": models,
        "anchor_model": args.anchor_model,
        "variant": args.variant,
        "score_mode": args.score_mode,
        "candidate_pool": args.candidate_pool,
        "anchor_weight": args.anchor_weight,
        "weights": weight_values,
        "tested": tested,
        "methods": methods,
        "ranking": ranked,
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({
        "top": ranked[:20],
        "output_path": args.output_path,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--anchor_model", default="v2")
    parser.add_argument("--variant", default="no_metadata_cf_popularity")
    parser.add_argument("--dev_feature_cache_dir", default="cache/ltr/dev_all_top100_v1")
    parser.add_argument("--score_mode", choices=["minmax", "rrf"], default="rrf")
    parser.add_argument("--candidate_pool", choices=["all", "anchor_top20"], default="all")
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument("--anchor_weight", type=float, default=1.0)
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
    )
    parser.add_argument(
        "--output_path",
        default="exp/ltr/locked_top1_fusion/report.json",
    )
    main(parser.parse_args())
