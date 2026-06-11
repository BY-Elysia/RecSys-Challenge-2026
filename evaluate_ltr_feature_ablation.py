"""Evaluate grouped feature and retrieval-channel ablations for a saved LTR model."""

import argparse
import json
import math
import os
from typing import Any

import lightgbm as lgb
import numpy as np

from train_ltr_ranker import (
    DEFAULT_BLIND_TURN_WEIGHTS,
    FeatureDataset,
    load_feature_dataset,
    summarize_ranks,
)


ABLATION_GROUPS = {
    "bm25_legacy": {
        "prefixes": ["bm25_legacy__"],
        "retrieval_channels": ["bm25_legacy"],
    },
    "bm25_feedback": {
        "prefixes": ["bm25_feedback__"],
        "retrieval_channels": ["bm25_feedback"],
    },
    "structure": {
        "prefixes": [
            "structure__",
            "same_last_artist",
            "same_any_artist",
            "same_last_album",
            "same_any_album",
        ],
        "retrieval_channels": ["structure"],
    },
    "image": {
        "prefixes": ["image-siglip2__"],
        "retrieval_channels": ["image-siglip2__last", "image-siglip2__mean"],
    },
    "metadata_dense": {
        "prefixes": ["metadata-qwen3_embedding_0.6b__"],
        "retrieval_channels": [
            "metadata-qwen3_embedding_0.6b__last",
            "metadata-qwen3_embedding_0.6b__mean",
        ],
    },
    "query_dense": {
        "prefixes": ["query-qwen3__"],
        "retrieval_channels": ["query-qwen3"],
    },
    "cf_retrieval": {
        "prefixes": ["user-cf__", "cf-bpr__"],
        "retrieval_channels": ["user-cf", "cf-bpr__last", "cf-bpr__mean"],
    },
    "cf": {
        "prefixes": ["user_track_cf_"],
        "retrieval_channels": [],
    },
    "query_match": {
        "prefixes": ["query_track_match", "query_artist_match", "query_album_match"],
        "retrieval_channels": [],
    },
    "popularity_release": {
        "prefixes": ["popularity", "release_year"],
        "retrieval_channels": [],
    },
    "conversation_context": {
        "prefixes": ["turn_number", "history_size", "goal_category__", "specificity__"],
        "retrieval_channels": [],
    },
}

RETRIEVAL_CHANNELS = [
    "bm25_legacy",
    "bm25_feedback",
    "structure",
    "image-siglip2__last",
    "image-siglip2__mean",
    "metadata-qwen3_embedding_0.6b__last",
    "metadata-qwen3_embedding_0.6b__mean",
    "query-qwen3",
    "user-cf",
    "cf-bpr__last",
    "cf-bpr__mean",
]


def matching_columns(feature_names: list[str], prefixes: list[str]) -> list[int]:
    return [
        index
        for index, name in enumerate(feature_names)
        if any(name == prefix or name.startswith(prefix) for prefix in prefixes)
    ]


def active_candidate_mask(
    features: np.ndarray,
    feature_names: list[str],
    removed_channels: list[str],
) -> np.ndarray:
    if not removed_channels:
        return np.ones(len(features), dtype=bool)
    available_channels = [
        name.removesuffix("__present")
        for name in feature_names
        if name.endswith("__present")
    ]
    remaining = [
        channel
        for channel in available_channels
        if channel not in removed_channels
    ]
    present_columns = [
        feature_names.index(f"{channel}__present")
        for channel in remaining
    ]
    return np.any(features[:, present_columns] > 0, axis=1)


def ranks_from_masked_predictions(
    dataset: FeatureDataset,
    predictions: np.ndarray,
    active_rows: np.ndarray,
) -> list[int | None]:
    ranks: list[int | None] = [None] * dataset.task_count
    offset = 0
    for group_size, task_index in zip(dataset.groups, dataset.group_task_indices):
        group_predictions = predictions[offset:offset + group_size]
        group_labels = dataset.labels[offset:offset + group_size]
        group_active = active_rows[offset:offset + group_size]
        positive_index = int(np.flatnonzero(group_labels)[0])
        if group_active[positive_index]:
            positive_score = group_predictions[positive_index]
            ranks[task_index] = int(
                np.count_nonzero(
                    group_active & (group_predictions > positive_score)
                ) + 1
            )
        offset += group_size
    return ranks


def metric_delta(
    ablated: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float]:
    return {
        key: float(ablated[key]) - float(baseline[key])
        for key in ["candidate_recall", "recall@1", "recall@5", "recall@20", "ndcg@20"]
    }


def main(args) -> None:
    dataset = load_feature_dataset(args.feature_cache_dir, mmap_mode="r")
    model = lgb.Booster(model_file=args.model_path)
    if model.feature_name() != dataset.feature_names:
        raise RuntimeError("Model feature names do not match cached features.")
    iteration = model.best_iteration if model.best_iteration > 0 else model.current_iteration()

    baseline_predictions = model.predict(dataset.features, num_iteration=iteration)
    all_active = np.ones(len(dataset.features), dtype=bool)
    baseline_ranks = ranks_from_masked_predictions(
        dataset,
        baseline_predictions,
        all_active,
    )
    baseline = summarize_ranks(baseline_ranks, dataset.task_turns)
    results = {}

    selected_groups = args.groups or list(ABLATION_GROUPS)
    for group_name in selected_groups:
        definition = ABLATION_GROUPS[group_name]
        columns = matching_columns(dataset.feature_names, definition["prefixes"])
        if not columns:
            raise RuntimeError(f"No feature columns matched ablation group {group_name}.")
        masked_features = np.array(dataset.features, copy=True)
        masked_features[:, columns] = 0
        active_rows = active_candidate_mask(
            dataset.features,
            dataset.feature_names,
            definition["retrieval_channels"],
        )
        predictions = model.predict(masked_features, num_iteration=iteration)
        ranks = ranks_from_masked_predictions(dataset, predictions, active_rows)
        summary = summarize_ranks(ranks, dataset.task_turns)
        results[group_name] = {
            "masked_features": [dataset.feature_names[index] for index in columns],
            "removed_retrieval_channels": definition["retrieval_channels"],
            "active_candidate_fraction": float(np.mean(active_rows)),
            "metrics": summary,
            "delta": {
                "overall": metric_delta(summary["overall"], baseline["overall"]),
                "turn1": metric_delta(summary["turn1"], baseline["turn1"]),
                "turn2plus": metric_delta(summary["turn2plus"], baseline["turn2plus"]),
                "blind_turn_weighted": metric_delta(
                    summary["blind_turn_weighted"],
                    baseline["blind_turn_weighted"],
                ),
            },
        }

    report = {
        "model_path": args.model_path,
        "feature_cache_dir": args.feature_cache_dir,
        "model_iteration": iteration,
        "tasks": dataset.task_count,
        "retrieved_groups": len(dataset.groups),
        "rows": int(len(dataset.labels)),
        "blind_turn_weights": DEFAULT_BLIND_TURN_WEIGHTS,
        "baseline": baseline,
        "ablations": results,
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    compact = [
        {
            "group": name,
            "overall_ndcg": value["metrics"]["overall"]["ndcg@20"],
            "overall_delta": value["delta"]["overall"]["ndcg@20"],
            "blind_weighted_ndcg": value["metrics"]["blind_turn_weighted"]["ndcg@20"],
            "blind_weighted_delta": value["delta"]["blind_turn_weighted"]["ndcg@20"],
            "candidate_recall_delta": value["delta"]["overall"]["candidate_recall"],
        }
        for name, value in results.items()
    ]
    compact.sort(key=lambda item: item["blind_weighted_delta"])
    print(json.dumps({
        "baseline_overall": baseline["overall"],
        "baseline_blind_turn_weighted": baseline["blind_turn_weighted"],
        "ablations": compact,
        "output_path": args.output_path,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        default="exp/ltr/multichannel_v1_10k_top100/model.txt",
    )
    parser.add_argument(
        "--feature_cache_dir",
        default="cache/ltr/dev_all_top100_v1",
    )
    parser.add_argument(
        "--output_path",
        default="exp/ltr/multichannel_v1_10k_top100_ablation/report.json",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=list(ABLATION_GROUPS),
        default=None,
    )
    main(parser.parse_args())
