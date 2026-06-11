"""Analyze candidate-recall and ranking losses for a saved LTR model by request segment."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import lightgbm as lgb
from datasets import load_dataset

from constraint_gating import strict_constraint_categories
from evaluate_ltr_feature_ablation import active_candidate_mask, ranks_from_masked_predictions
from run_inference_ltr_blindset import load_removed_channels
from train_ltr_ranker import load_feature_dataset, normalize_text, summarize_ranks


def request_intents(request: str) -> set[str]:
    query = f" {normalize_text(request)} "
    intents = set()
    if any(value in query for value in (" album art ", " cover art ", " album cover ", " artwork ", " visual ")):
        intents.add("visual")
    if any(value in query for value in (" lyric ", " lyrics ", " theme ", " themes ", " story ", " narrative ")):
        intents.add("lyrics_or_theme")
    if any(value in query for value in (
        " what album ", " which song ", " which track ", " tell me more ",
        " can you tell me ", " what specific ", " what makes ",
    )):
        intents.add("factual_or_explanatory")
    if any(value in query for value in (" play ", " put on ", " next ")):
        intents.add("specific_or_play")
    if any(value in query for value in (" another ", " more ", " keep them coming ", " what else ", " similar ")):
        intents.add("continuation")
    if any(value in query for value in (" mood ", " feel ", " vibe ", " relax ", " focus ", " work ", " run ", " party ", " dance ")):
        intents.add("mood_or_activity")
    if any(value in query for value in (" new artist ", " other artist ", " different artist ", " branch out ", " discover ")):
        intents.add("discovery")
    intents.update(f"constraint_{value}" for value in strict_constraint_categories(request))
    return intents or {"other"}


def task_attributes(dataset) -> list[dict[str, str | int | set[str]]]:
    attributes = []
    for item in dataset:
        target_turns = sorted({
            turn["turn_number"]
            for turn in item["conversations"]
            if turn["role"] == "music"
        })
        goal = item.get("conversation_goal", {})
        for target_turn in target_turns:
            request = next(
                turn["content"]
                for turn in item["conversations"]
                if turn["role"] == "user" and turn["turn_number"] == target_turn
            )
            attributes.append({
                "turn": int(target_turn),
                "goal_category": str(goal.get("category") or "unknown"),
                "specificity": str(goal.get("specificity") or "unknown"),
                "intents": request_intents(str(request)),
            })
    return attributes


def rank_score(rank: int | None) -> float:
    return 1.0 / math.log2(rank + 1) if rank is not None and rank <= 20 else 0.0


def summarize_segment(
    indices: list[int],
    model_ranks: list[int | None],
    baseline_ranks: dict[str, list[int | None]],
    total_tasks: int,
) -> dict:
    ranks = [model_ranks[index] for index in indices]
    count = len(indices)
    candidate_recall = sum(rank is not None for rank in ranks) / count
    recall20 = sum(rank is not None and rank <= 20 for rank in ranks) / count
    ndcg20 = sum(rank_score(rank) for rank in ranks) / count
    channel_recall20 = {
        name: sum(
            baseline_ranks[name][index] is not None and baseline_ranks[name][index] <= 20
            for index in indices
        ) / count
        for name in baseline_ranks
    }
    return {
        "tasks": count,
        "task_fraction": count / total_tasks,
        "candidate_recall": candidate_recall,
        "recall@20": recall20,
        "ndcg@20": ndcg20,
        "candidate_miss_rate": 1.0 - candidate_recall,
        "ranking_loss_after_recall": candidate_recall - ndcg20,
        "weighted_candidate_miss": count / total_tasks * (1.0 - candidate_recall),
        "weighted_ranking_loss": count / total_tasks * (candidate_recall - ndcg20),
        "best_single_channel_recall@20": max(channel_recall20.values()),
        "single_channel_recall@20": dict(sorted(
            channel_recall20.items(),
            key=lambda item: item[1],
            reverse=True,
        )),
    }


def main(args: argparse.Namespace) -> None:
    cache = load_feature_dataset(args.dev_feature_cache_dir, mmap_mode="r")
    raw = load_dataset(args.dataset_name, split="test", cache_dir=args.cache_dir)
    attributes = task_attributes(raw)
    if len(attributes) != cache.task_count:
        raise RuntimeError(f"Attribute/task mismatch: {len(attributes)} != {cache.task_count}")

    model = lgb.Booster(model_file=args.model_path)
    names = model.feature_name()
    missing = [name for name in names if name not in cache.feature_names]
    if missing:
        raise RuntimeError(f"Model is missing cached features: {missing}")
    indices = [cache.feature_names.index(name) for name in names]
    iteration = model.best_iteration if model.best_iteration > 0 else model.current_iteration()
    predictions = model.predict(cache.features[:, indices], num_iteration=iteration)
    active = active_candidate_mask(
        cache.features,
        cache.feature_names,
        load_removed_channels(args.model_path),
    )
    ranks = ranks_from_masked_predictions(cache, predictions, active)

    dimensions: dict[str, dict[str, list[int]]] = {
        "turn": defaultdict(list),
        "goal_category": defaultdict(list),
        "specificity": defaultdict(list),
        "intent": defaultdict(list),
    }
    for index, item in enumerate(attributes):
        dimensions["turn"][str(item["turn"])].append(index)
        dimensions["goal_category"][str(item["goal_category"])].append(index)
        dimensions["specificity"][str(item["specificity"])].append(index)
        for intent in item["intents"]:
            dimensions["intent"][str(intent)].append(index)

    segments = {
        dimension: {
            value: summarize_segment(task_indices, ranks, cache.baseline_ranks, cache.task_count)
            for value, task_indices in values.items()
            if len(task_indices) >= args.min_segment_tasks
        }
        for dimension, values in dimensions.items()
    }
    opportunities = []
    for dimension, values in segments.items():
        for value, metrics in values.items():
            opportunities.append({
                "dimension": dimension,
                "segment": value,
                **{key: metrics[key] for key in (
                    "tasks",
                    "candidate_recall",
                    "recall@20",
                    "ndcg@20",
                    "weighted_candidate_miss",
                    "weighted_ranking_loss",
                    "best_single_channel_recall@20",
                )},
            })
    candidate_opportunities = sorted(
        opportunities, key=lambda value: value["weighted_candidate_miss"], reverse=True
    )
    ranking_opportunities = sorted(
        opportunities, key=lambda value: value["weighted_ranking_loss"], reverse=True
    )
    report = {
        "model_path": args.model_path,
        "model_iteration": iteration,
        "dev_feature_cache_dir": args.dev_feature_cache_dir,
        "overall": summarize_ranks(ranks, cache.task_turns),
        "segments": segments,
        "largest_candidate_recall_opportunities": candidate_opportunities[:args.top_segments],
        "largest_ranking_opportunities": ranking_opportunities[:args.top_segments],
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({
        "overall": report["overall"],
        "largest_candidate_recall_opportunities": report["largest_candidate_recall_opportunities"],
        "largest_ranking_opportunities": report["largest_ranking_opportunities"],
        "output_path": args.output_path,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument("--dev_feature_cache_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--min_segment_tasks", type=int, default=30)
    parser.add_argument("--top_segments", type=int, default=20)
    parser.add_argument(
        "--output_path",
        default="exp/ltr/error_segments/report.json",
    )
    main(parser.parse_args())
