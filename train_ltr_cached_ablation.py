"""Retrain controlled LTR ablations from cached candidate feature matrices."""

import argparse
import gc
import json
import os
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np

from evaluate_ltr_feature_ablation import RETRIEVAL_CHANNELS
from train_ltr_ranker import (
    FeatureDataset,
    load_feature_dataset,
    ranks_from_predictions,
    summarize_ranks,
)


VARIANTS = {
    "baseline": {
        "remove_prefixes": [],
        "remove_channels": [],
    },
    "no_cf": {
        "remove_prefixes": ["user_track_cf_"],
        "remove_channels": [],
    },
    "no_query_dense": {
        "remove_prefixes": ["query-qwen3__"],
        "remove_channels": ["query-qwen3"],
    },
    "no_cf_retrieval": {
        "remove_prefixes": ["user-cf__", "cf-bpr__"],
        "remove_channels": ["user-cf", "cf-bpr__last", "cf-bpr__mean"],
    },
    "no_user_cf_retrieval": {
        "remove_prefixes": ["user-cf__"],
        "remove_channels": ["user-cf"],
    },
    "no_history_cf_retrieval": {
        "remove_prefixes": ["cf-bpr__"],
        "remove_channels": ["cf-bpr__last", "cf-bpr__mean"],
    },
    "no_new_retrieval": {
        "remove_prefixes": ["query-qwen3__", "user-cf__", "cf-bpr__"],
        "remove_channels": [
            "query-qwen3",
            "user-cf",
            "cf-bpr__last",
            "cf-bpr__mean",
        ],
    },
    "no_popularity_release": {
        "remove_prefixes": ["popularity", "release_year"],
        "remove_channels": [],
    },
    "no_cf_popularity_release": {
        "remove_prefixes": ["user_track_cf_", "popularity", "release_year"],
        "remove_channels": [],
    },
    "no_metadata_channel": {
        "remove_prefixes": ["metadata-qwen3_embedding_0.6b__"],
        "remove_channels": [
            "metadata-qwen3_embedding_0.6b__last",
            "metadata-qwen3_embedding_0.6b__mean",
        ],
    },
    "no_metadata_cf_popularity": {
        "remove_prefixes": [
            "metadata-qwen3_embedding_0.6b__",
            "user_track_cf_",
            "popularity",
            "release_year",
        ],
        "remove_channels": [
            "metadata-qwen3_embedding_0.6b__last",
            "metadata-qwen3_embedding_0.6b__mean",
        ],
    },
    "lean_history_cf": {
        "remove_prefixes": [
            "metadata-qwen3_embedding_0.6b__",
            "user_track_cf_",
            "user-cf__",
            "popularity",
            "release_year",
        ],
        "remove_channels": [
            "metadata-qwen3_embedding_0.6b__last",
            "metadata-qwen3_embedding_0.6b__mean",
            "user-cf",
        ],
    },
}


def kept_feature_indices(
    feature_names: list[str],
    remove_prefixes: list[str],
) -> list[int]:
    return [
        index
        for index, name in enumerate(feature_names)
        if not any(name == prefix or name.startswith(prefix) for prefix in remove_prefixes)
    ]


def active_rows_for_channels(
    dataset: FeatureDataset,
    remove_channels: list[str],
) -> np.ndarray:
    if not remove_channels:
        return np.ones(len(dataset.labels), dtype=bool)
    available_channels = [
        name.removesuffix("__present")
        for name in dataset.feature_names
        if name.endswith("__present")
    ]
    remaining_channels = [
        channel
        for channel in available_channels
        if channel not in remove_channels
    ]
    present_columns = [
        dataset.feature_names.index(f"{channel}__present")
        for channel in remaining_channels
    ]
    return np.any(dataset.features[:, present_columns] > 0, axis=1)


def prepare_dataset(
    dataset: FeatureDataset,
    feature_indices: list[int],
    remove_channels: list[str],
    allowed_turns: set[int] | None = None,
    require_positive: bool = False,
) -> FeatureDataset:
    active_rows = active_rows_for_channels(dataset, remove_channels)
    kept_rows = np.zeros(len(dataset.labels), dtype=bool)
    groups = []
    group_task_indices = []
    offset = 0

    for group_size, task_index in zip(dataset.groups, dataset.group_task_indices):
        group_active = active_rows[offset:offset + group_size]
        group_labels = dataset.labels[offset:offset + group_size]
        allowed = allowed_turns is None or dataset.task_turns[task_index] in allowed_turns
        positive_active = bool(np.any(group_active & (group_labels > 0)))
        if allowed and np.any(group_active) and (positive_active or not require_positive):
            kept_rows[offset:offset + group_size] = group_active
            groups.append(int(np.count_nonzero(group_active)))
            group_task_indices.append(task_index)
        offset += group_size

    row_indices = np.flatnonzero(kept_rows)
    features = np.asarray(
        dataset.features[np.ix_(row_indices, feature_indices)],
        dtype=np.float32,
    )
    labels = np.asarray(dataset.labels[row_indices], dtype=np.int8)
    return FeatureDataset(
        features=features,
        labels=labels,
        groups=groups,
        group_task_indices=group_task_indices,
        task_count=dataset.task_count,
        task_turns=dataset.task_turns,
        baseline_ranks={},
        feature_names=[dataset.feature_names[index] for index in feature_indices],
    )


def train_variant(
    name: str,
    definition: dict[str, list[str]],
    train_cache: FeatureDataset,
    dev_cache: FeatureDataset,
    args,
) -> dict[str, Any]:
    feature_indices = kept_feature_indices(
        train_cache.feature_names,
        definition["remove_prefixes"],
    )
    print(f"[{name}] Preparing train matrix...")
    train_data = prepare_dataset(
        train_cache,
        feature_indices,
        definition["remove_channels"],
        require_positive=True,
    )
    print(f"[{name}] Preparing Turn 8 validation matrix...")
    valid_data = prepare_dataset(
        dev_cache,
        feature_indices,
        definition["remove_channels"],
        allowed_turns={8},
        require_positive=True,
    )
    print(f"[{name}] Preparing full Dev matrix...")
    dev_data = prepare_dataset(
        dev_cache,
        feature_indices,
        definition["remove_channels"],
        require_positive=False,
    )

    train_set = lgb.Dataset(
        train_data.features,
        label=train_data.labels,
        group=train_data.groups,
        feature_name=train_data.feature_names,
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        valid_data.features,
        label=valid_data.labels,
        group=valid_data.groups,
        feature_name=valid_data.feature_names,
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
        valid_sets=[valid_set],
        valid_names=["turn8"],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds),
            lgb.log_evaluation(args.log_period),
        ],
    )
    best_iteration = model.best_iteration
    predictions = model.predict(
        dev_data.features,
        num_iteration=best_iteration,
    )
    ranks = ranks_from_predictions(dev_data, predictions)
    metrics = summarize_ranks(ranks, dev_data.task_turns)

    output_dir = os.path.join(args.output_dir, name)
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.txt")
    model.save_model(model_path, num_iteration=best_iteration)
    report = {
        "variant": name,
        "removed_prefixes": definition["remove_prefixes"],
        "removed_channels": definition["remove_channels"],
        "best_iteration": best_iteration,
        "feature_count": len(feature_indices),
        "train": {
            "groups": len(train_data.groups),
            "rows": int(len(train_data.labels)),
        },
        "turn8_validation": {
            "groups": len(valid_data.groups),
            "rows": int(len(valid_data.labels)),
        },
        "dev": {
            "groups": len(dev_data.groups),
            "rows": int(len(dev_data.labels)),
        },
        "metrics": metrics,
        "feature_importance_gain": dict(sorted(
            zip(
                dev_data.feature_names,
                model.feature_importance(importance_type="gain").tolist(),
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )),
        "model_path": model_path,
    }
    with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    del train_set, valid_set, model, predictions
    del train_data, valid_data, dev_data
    gc.collect()
    return report


def main(args) -> None:
    train_cache = load_feature_dataset(args.train_feature_cache_dir, mmap_mode="r")
    dev_cache = load_feature_dataset(args.dev_feature_cache_dir, mmap_mode="r")
    if train_cache.feature_names != dev_cache.feature_names:
        raise RuntimeError("Train and Dev feature names do not match.")

    selected = args.variants or list(VARIANTS)
    reports = {}
    for name in selected:
        reports[name] = train_variant(
            name,
            VARIANTS[name],
            train_cache,
            dev_cache,
            args,
        )

    baseline_ndcg = reports.get("baseline", {}).get("metrics", {})
    baseline_overall = baseline_ndcg.get("overall", {}).get("ndcg@20")
    baseline_blind = baseline_ndcg.get("blind_turn_weighted", {}).get("ndcg@20")
    summary = []
    for name, report in reports.items():
        overall = report["metrics"]["overall"]["ndcg@20"]
        blind = report["metrics"]["blind_turn_weighted"]["ndcg@20"]
        summary.append({
            "variant": name,
            "best_iteration": report["best_iteration"],
            "feature_count": report["feature_count"],
            "train_groups": report["train"]["groups"],
            "overall_ndcg@20": overall,
            "blind_weighted_ndcg@20": blind,
            "overall_delta_vs_baseline": (
                overall - baseline_overall if baseline_overall is not None else None
            ),
            "blind_delta_vs_baseline": (
                blind - baseline_blind if baseline_blind is not None else None
            ),
        })
    summary.sort(key=lambda item: item["blind_weighted_ndcg@20"], reverse=True)
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps({"summary": summary, "summary_path": summary_path}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_feature_cache_dir",
        default="cache/ltr/train10k_seed13_top100_v1",
    )
    parser.add_argument(
        "--dev_feature_cache_dir",
        default="cache/ltr/dev_all_top100_v1",
    )
    parser.add_argument(
        "--output_dir",
        default="exp/ltr/cached_ablation_10k_top100",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=None,
    )
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
    main(parser.parse_args())
