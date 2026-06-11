"""Evaluate score ensembles and turn-aware gates for compatible LTR models."""

import argparse
import itertools
import json
import os

import lightgbm as lgb
import numpy as np

from train_ltr_cached_ablation import VARIANTS, kept_feature_indices, prepare_dataset
from train_ltr_ranker import load_feature_dataset, ranks_from_predictions, summarize_ranks


def groupwise_normalize(
    predictions: np.ndarray,
    groups: list[int],
    mode: str,
) -> np.ndarray:
    normalized = np.empty_like(predictions, dtype=np.float64)
    offset = 0
    for group_size in groups:
        values = predictions[offset:offset + group_size]
        if mode == "minmax":
            low = float(np.min(values))
            high = float(np.max(values))
            normalized[offset:offset + group_size] = (
                (values - low) / (high - low)
                if high > low
                else 0.0
            )
        elif mode == "rrf":
            order = np.argsort(-values, kind="stable")
            ranks = np.empty(group_size, dtype=np.int32)
            ranks[order] = np.arange(1, group_size + 1)
            normalized[offset:offset + group_size] = 1.0 / (60.0 + ranks)
        else:
            raise ValueError(mode)
        offset += group_size
    return normalized


def turn_gate_predictions(
    dataset,
    predictions_by_name: dict[str, np.ndarray],
    turn1_model: str,
    later_model: str,
) -> np.ndarray:
    combined = np.empty(len(dataset.labels), dtype=np.float64)
    offset = 0
    for group_size, task_index in zip(dataset.groups, dataset.group_task_indices):
        model_name = turn1_model if dataset.task_turns[task_index] == 1 else later_model
        combined[offset:offset + group_size] = predictions_by_name[model_name][
            offset:offset + group_size
        ]
        offset += group_size
    return combined


def evaluate(dataset, predictions: np.ndarray) -> dict:
    ranks = ranks_from_predictions(dataset, predictions)
    return summarize_ranks(ranks, dataset.task_turns)


def compact(metrics: dict) -> dict:
    return {
        "overall_ndcg@20": metrics["overall"]["ndcg@20"],
        "overall_recall@20": metrics["overall"]["recall@20"],
        "turn1_ndcg@20": metrics["turn1"]["ndcg@20"],
        "turn2plus_ndcg@20": metrics["turn2plus"]["ndcg@20"],
        "blind_weighted_ndcg@20": metrics["blind_turn_weighted"]["ndcg@20"],
        "blind_weighted_recall@20": metrics["blind_turn_weighted"]["recall@20"],
    }


def main(args) -> None:
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

    models = {}
    predictions = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        model = lgb.Booster(model_file=path)
        if model.feature_name() != dataset.feature_names:
            raise RuntimeError(f"Feature mismatch for model {name}.")
        iteration = model.best_iteration if model.best_iteration > 0 else model.current_iteration()
        models[name] = {"path": path, "iteration": iteration}
        predictions[name] = model.predict(dataset.features, num_iteration=iteration)

    methods = {
        name: evaluate(dataset, values)
        for name, values in predictions.items()
    }
    names = list(predictions)
    for size in range(2, len(names) + 1):
        for selected_names in itertools.combinations(names, size):
            selected_names = list(selected_names)
            label = "_".join(selected_names)
            raw_mean = np.mean([predictions[name] for name in selected_names], axis=0)
            methods[f"raw_mean__{label}"] = evaluate(dataset, raw_mean)
            minmax_mean = np.mean([
                groupwise_normalize(predictions[name], dataset.groups, "minmax")
                for name in selected_names
            ], axis=0)
            methods[f"minmax_mean__{label}"] = evaluate(dataset, minmax_mean)
            rrf = np.sum([
                groupwise_normalize(predictions[name], dataset.groups, "rrf")
                for name in selected_names
            ], axis=0)
            methods[f"rrf__{label}"] = evaluate(dataset, rrf)

    for turn1_model, later_model in itertools.permutations(names, 2):
        gated = turn_gate_predictions(
            dataset,
            predictions,
            turn1_model,
            later_model,
        )
        methods[
            f"gate__turn1_{turn1_model}__turn2plus_{later_model}"
        ] = evaluate(dataset, gated)

    ranked = sorted(
        (
            {"method": name, **compact(metrics)}
            for name, metrics in methods.items()
        ),
        key=lambda item: item["blind_weighted_ndcg@20"],
        reverse=True,
    )
    report = {
        "models": models,
        "dev_feature_cache_dir": args.dev_feature_cache_dir,
        "rows": int(len(dataset.labels)),
        "groups": len(dataset.groups),
        "methods": methods,
        "ranking": ranked,
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({
        "ranking": ranked,
        "output_path": args.output_path,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model specifications in name=path form.",
    )
    parser.add_argument(
        "--dev_feature_cache_dir",
        default="cache/ltr/dev_all_top100_v1",
    )
    parser.add_argument("--variant", choices=list(VARIANTS), default="no_metadata_cf_popularity")
    parser.add_argument(
        "--output_path",
        default="exp/ltr/lean_30k_top100_ensemble/report.json",
    )
    main(parser.parse_args())
