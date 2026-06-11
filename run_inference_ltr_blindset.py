"""Run Blind A inference with multi-channel retrieval and LambdaRank fusion."""

import argparse
import json
import os

import lightgbm as lgb
import numpy as np
from datasets import load_dataset
from omegaconf import OmegaConf

from evaluate_final_turn_recall import (
    build_metadata_indexes,
    feedback_rich_query,
)
from mcrs.retrieval_modules.bm25 import BM25_MODEL
from mcrs.retrieval_modules.precomputed_embeddings import PrecomputedTrackEmbeddingIndex
from train_ltr_ranker import (
    RankingTask,
    build_inference_feature_dataset,
    load_user_cf_embeddings,
)
from mcrs.retrieval_modules.qwen_dense import DEFAULT_TASK_INSTRUCTION
from train_reranker import build_query_text, track_to_text


def load_removed_channels(model_path: str) -> list[str]:
    report_path = os.path.join(os.path.dirname(model_path), "report.json")
    if not os.path.exists(report_path):
        return []
    with open(report_path, encoding="utf-8") as file:
        report = json.load(file)
    return list(report.get("removed_channels", []))


def prepare_model_inputs(
    inference,
    model_feature_names: list[str],
    removed_channels: list[str],
) -> tuple[np.ndarray, list[list[str]], list[int]]:
    missing = [
        name
        for name in model_feature_names
        if name not in inference.feature_names
    ]
    if missing:
        raise RuntimeError(f"Inference is missing model features: {missing}")
    feature_indices = [
        inference.feature_names.index(name)
        for name in model_feature_names
    ]
    model_channels = [
        name.removesuffix("__present")
        for name in model_feature_names
        if name.endswith("__present")
    ]
    remaining_channels = [channel for channel in model_channels if channel not in removed_channels]
    present_indices = [
        inference.feature_names.index(f"{channel}__present")
        for channel in remaining_channels
    ]

    feature_groups = []
    filtered_candidates = []
    filtered_groups = []
    offset = 0
    for group_size, candidates in zip(inference.groups, inference.candidates):
        group_features = inference.features[offset:offset + group_size]
        if removed_channels:
            active = np.any(group_features[:, present_indices] > 0, axis=1)
        else:
            active = np.ones(group_size, dtype=bool)
        selected_candidates = [
            track_id
            for track_id, is_active in zip(candidates, active)
            if is_active
        ]
        if len(selected_candidates) < 20:
            raise RuntimeError(
                f"Only {len(selected_candidates)} active candidates remain after channel filtering."
            )
        feature_groups.append(group_features[active][:, feature_indices])
        filtered_candidates.append(selected_candidates)
        filtered_groups.append(len(selected_candidates))
        offset += group_size
    return np.concatenate(feature_groups), filtered_candidates, filtered_groups


def build_blind_tasks(
    dataset,
    track_texts: dict[str, str],
    history_turns: int,
) -> list[RankingTask]:
    tasks = []
    for item in dataset:
        user_turn = item["conversations"][-1]
        if user_turn["role"] != "user":
            raise ValueError(f"Session {item['session_id']} does not end with a user turn.")
        target_turn = user_turn["turn_number"]
        history = [
            turn["content"]
            for turn in item["conversations"]
            if turn["role"] == "music" and turn["turn_number"] < target_turn
        ]
        goal = item.get("conversation_goal", {})
        tasks.append(RankingTask(
            session_id=item["session_id"],
            user_id=item["user_id"],
            turn_number=target_turn,
            target="",
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
            goal_category=str(goal.get("category") or ""),
            specificity=str(goal.get("specificity") or ""),
        ))
    return tasks


def main(args) -> None:
    config = OmegaConf.load(f"config/{args.tid}.yaml")
    args.cache_dir = config.cache_dir
    model = lgb.Booster(model_file=args.model_path)
    best_iteration = (
        model.best_iteration
        if model.best_iteration > 0
        else model.current_iteration()
    )
    turn1_model = None
    turn1_iteration = None
    if args.turn1_model_path:
        turn1_model = lgb.Booster(model_file=args.turn1_model_path)
        turn1_iteration = (
            turn1_model.best_iteration
            if turn1_model.best_iteration > 0
            else turn1_model.current_iteration()
        )
        if turn1_model.feature_name() != model.feature_name():
            raise RuntimeError("Turn 1 and later-turn models use different feature sets.")
        if load_removed_channels(args.turn1_model_path) != load_removed_channels(args.model_path):
            raise RuntimeError("Turn 1 and later-turn models remove different retrieval channels.")

    bm25 = BM25_MODEL(
        config.item_db_name,
        config.track_split_types,
        config.corpus_types,
        config.cache_dir,
    )
    track_texts = {
        track_id: track_to_text(item)
        for track_id, item in bm25.metadata_dict.items()
    }
    artist_tracks, album_tracks = build_metadata_indexes(bm25.metadata_dict)
    user_cf = load_user_cf_embeddings(args.user_embedding_name, config.cache_dir)
    track_cf = PrecomputedTrackEmbeddingIndex(
        embedding_field="cf-bpr",
        cache_dir=config.cache_dir,
        device="cpu",
    )

    blind = load_dataset(
        config.test_dataset_name,
        split="test",
        cache_dir=config.cache_dir,
    )
    tasks = build_blind_tasks(blind, track_texts, args.history_turns)
    inference = build_inference_feature_dataset(
        tasks,
        bm25,
        artist_tracks,
        album_tracks,
        user_cf,
        track_cf,
        args,
    )
    if model.feature_name() != inference.feature_names:
        print(
            f"Selecting {model.num_feature()} model features from "
            f"{len(inference.feature_names)} available inference features."
        )
    removed_channels = load_removed_channels(args.model_path)
    model_features, candidates_by_task, groups = prepare_model_inputs(
        inference,
        model.feature_name(),
        removed_channels,
    )

    predictions = model.predict(
        model_features,
        num_iteration=best_iteration,
    )
    turn1_predictions = (
        turn1_model.predict(model_features, num_iteration=turn1_iteration)
        if turn1_model is not None
        else None
    )
    results = []
    offset = 0
    for task, candidates, group_size in zip(tasks, candidates_by_task, groups):
        source_predictions = (
            turn1_predictions
            if task.turn_number == 1 and turn1_predictions is not None
            else predictions
        )
        group_scores = source_predictions[offset:offset + group_size]
        order = np.argsort(-group_scores, kind="stable")
        ranked = [candidates[index] for index in order[:args.topk]]
        if len(ranked) != args.topk:
            raise RuntimeError(
                f"Session {task.session_id} produced {len(ranked)} tracks, expected {args.topk}."
            )
        results.append({
            "session_id": task.session_id,
            "user_id": task.user_id,
            "turn_number": task.turn_number,
            "predicted_track_ids": ranked,
            "predicted_response": "",
        })
        offset += group_size

    output_dir = os.path.join("exp", "inference", args.eval_dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.output_name}.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False)
    print(json.dumps({
        "sessions": len(results),
        "features": model_features.shape[1],
        "candidate_rows": int(model_features.shape[0]),
        "removed_channels": removed_channels,
        "model_iteration": best_iteration,
        "turn1_model_path": args.turn1_model_path,
        "turn1_model_iteration": turn1_iteration,
        "output_path": output_path,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Blind A inference with multi-channel LambdaRank fusion."
    )
    parser.add_argument("--tid", default="bm25_tags_doubao_blindset_A")
    parser.add_argument(
        "--model_path",
        default="exp/ltr/multichannel_v1_10k_top100/model.txt",
    )
    parser.add_argument("--turn1_model_path", default=None)
    parser.add_argument("--output_name", default="multichannel_ltr_top100_empty")
    parser.add_argument("--eval_dataset", default="blindset_A")
    parser.add_argument(
        "--user_embedding_name",
        default="talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
    )
    parser.add_argument("--channel_topk", type=int, default=100)
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
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
