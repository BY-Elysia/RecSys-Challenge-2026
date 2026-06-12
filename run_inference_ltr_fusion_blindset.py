"""Fuse LTR models with different feature sets and lock a champion prefix."""

from __future__ import annotations

import argparse
import json
import os

import lightgbm as lgb
import numpy as np
from datasets import load_dataset
from omegaconf import OmegaConf

from evaluate_cross_model_fusion import masked_groupwise_scores, model_active_rows
from evaluate_final_turn_recall import build_metadata_indexes
from mcrs.retrieval_modules.bm25 import BM25_MODEL
from mcrs.retrieval_modules.precomputed_embeddings import PrecomputedTrackEmbeddingIndex
from mcrs.retrieval_modules.qwen_dense import DEFAULT_TASK_INSTRUCTION
from run_inference_ltr_blindset import build_blind_tasks, load_removed_channels
from train_ltr_ranker import build_inference_feature_dataset, load_user_cf_embeddings
from train_reranker import track_to_text


def model_predictions(inference, model_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    model = lgb.Booster(model_file=model_path)
    names = model.feature_name()
    missing = [name for name in names if name not in inference.feature_names]
    if missing:
        raise RuntimeError(f"Model {model_path} is missing inference features: {missing}")
    indices = [inference.feature_names.index(name) for name in names]
    iteration = model.best_iteration if model.best_iteration > 0 else model.current_iteration()
    predictions = model.predict(inference.features[:, indices], num_iteration=iteration)
    removed = load_removed_channels(model_path)
    active = model_active_rows(inference, names, removed)
    return predictions, active, {
        "path": model_path,
        "iteration": iteration,
        "feature_count": len(names),
        "removed_channels": removed,
    }


def main(args: argparse.Namespace) -> None:
    config = OmegaConf.load(f"config/{args.tid}.yaml")
    args.cache_dir = config.cache_dir
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
    blind = load_dataset(config.test_dataset_name, split="test", cache_dir=config.cache_dir)
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

    anchor_raw, anchor_active, anchor_meta = model_predictions(inference, args.anchor_model_path)
    candidate_raw, candidate_active, candidate_meta = model_predictions(
        inference,
        args.candidate_model_path,
    )
    anchor_scores = masked_groupwise_scores(
        anchor_raw, anchor_active, inference.groups, args.score_mode, args.rrf_k
    )
    candidate_scores = masked_groupwise_scores(
        candidate_raw, candidate_active, inference.groups, args.score_mode, args.rrf_k
    )
    fused = args.anchor_weight * anchor_scores + args.candidate_weight * candidate_scores
    union_active = anchor_active | candidate_active

    base = json.load(open(args.base_prediction_path, encoding="utf-8"))
    base_by_session = {item["session_id"]: item for item in base}
    results = []
    offset = 0
    for task, candidates, group_size in zip(tasks, inference.candidates, inference.groups):
        base_item = base_by_session[task.session_id]
        locked = list(base_item["predicted_track_ids"][:args.lock_prefix])
        group_active = union_active[offset:offset + group_size]
        group_scores = fused[offset:offset + group_size]
        active_indices = np.flatnonzero(group_active)
        order = active_indices[np.argsort(-group_scores[active_indices], kind="stable")]
        ranked = locked.copy()
        for index in order:
            track_id = candidates[int(index)]
            if track_id not in ranked:
                ranked.append(track_id)
            if len(ranked) == args.topk:
                break
        for track_id in base_item["predicted_track_ids"]:
            if track_id not in ranked:
                ranked.append(track_id)
            if len(ranked) == args.topk:
                break
        results.append({
            "session_id": task.session_id,
            "user_id": task.user_id,
            "turn_number": task.turn_number,
            "predicted_track_ids": ranked,
            "predicted_response": base_item["predicted_response"],
        })
        offset += group_size

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False)
    print(json.dumps({
        "output_path": args.output_path,
        "sessions": len(results),
        "anchor": anchor_meta,
        "candidate": candidate_meta,
        "score_mode": args.score_mode,
        "anchor_weight": args.anchor_weight,
        "candidate_weight": args.candidate_weight,
        "lock_prefix": args.lock_prefix,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tid", default="bm25_tags_doubao_blindset_A")
    parser.add_argument("--anchor_model_path", required=True)
    parser.add_argument("--candidate_model_path", required=True)
    parser.add_argument(
        "--base_prediction_path",
        default="exp/inference/blindset_A/multichannel_ltr_turn1_s13_later_v2_top1lock_prediction.json",
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--score_mode", choices=["rrf", "minmax"], default="rrf")
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument("--anchor_weight", type=float, default=1.0)
    parser.add_argument("--candidate_weight", type=float, required=True)
    parser.add_argument("--lock_prefix", type=int, default=1)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--user_embedding_name",
        default="talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
    )
    parser.add_argument("--channel_topk", type=int, default=100)
    parser.add_argument("--embedding_batch_size", type=int, default=64)
    parser.add_argument("--text_retrieval_batch_size", type=int, default=5000)
    parser.add_argument("--history_turns", type=int, default=0)
    parser.add_argument("--enable_query_dense", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable_cf_retrieval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--enable_supervised_dense",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--query_dense_embedding_field", default="metadata-qwen3_embedding_0.6b")
    parser.add_argument("--query_dense_model_name", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--query_dense_max_length", type=int, default=512)
    parser.add_argument("--query_dense_batch_size", type=int, default=16)
    parser.add_argument("--query_dense_instruction", default=DEFAULT_TASK_INSTRUCTION)
    parser.add_argument(
        "--supervised_dense_checkpoint",
        default="exp/dense/supervised_qwen_adapter_10k_feedback",
    )
    parser.add_argument("--supervised_dense_query_batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
