"""Evaluate BM25 plus dense candidate union with the current cross-encoder reranker."""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from typing import Any

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluate_devset import score_candidates
from mcrs.retrieval_modules.bm25 import BM25_MODEL
from mcrs.retrieval_modules.qwen_dense import DEFAULT_TASK_INSTRUCTION, QwenDenseRetriever
from train_reranker import build_query_text, get_turn_target, track_to_text


def average(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def macro_by_session(records: list[dict[str, Any]], metric: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record["session_id"]].append(float(record[metric]))
    return average([average(values) for values in grouped.values()])


def rank_of(candidates: list[str], target: str) -> int | None:
    return candidates.index(target) + 1 if target in candidates else None


def ndcg(rank: int | None, cutoff: int = 20) -> float:
    if rank is None or rank > cutoff:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def union_candidates(primary: list[str], supplement: list[str]) -> list[str]:
    seen = set(primary)
    return primary + [track_id for track_id in supplement if track_id not in seen]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sessions": len({record["session_id"] for record in records}),
        "turns": len(records),
        "bm25_candidate_recall": macro_by_session(records, "bm25_hit"),
        "dense_candidate_recall": macro_by_session(records, "dense_hit"),
        "union_candidate_recall": macro_by_session(records, "union_hit"),
        "dense_incremental_recall": macro_by_session(records, "dense_incremental_hit"),
        "final_recall@20": macro_by_session(records, "final_recall@20"),
        "final_ndcg@20": macro_by_session(records, "final_ndcg@20"),
        "average_union_size": average([record["union_size"] for record in records]),
        "dense_incremental_targets": int(sum(record["dense_incremental_hit"] for record in records)),
        "dense_incremental_targets_reaching_final@20": int(
            sum(record["dense_incremental_final_hit"] for record in records)
        ),
    }


def main(args) -> None:
    config = OmegaConf.load(f"config/{args.tid}.yaml")
    corpus_types = args.corpus_types or list(config.corpus_types)
    bm25 = BM25_MODEL(config.item_db_name, config.track_split_types, corpus_types, config.cache_dir)
    track_texts = {track_id: track_to_text(metadata) for track_id, metadata in bm25.metadata_dict.items()}

    dataset = load_dataset(args.dev_dataset_name, split=args.dev_split, cache_dir=config.cache_dir)
    if args.max_sessions:
        indices = list(range(len(dataset)))
        random.Random(args.seed).shuffle(indices)
        dataset = dataset.select(indices[:min(args.max_sessions, len(dataset))])

    tasks: list[dict[str, Any]] = []
    for item in tqdm(dataset, desc="Building hybrid Dev queries"):
        target_turns = sorted({
            turn["turn_number"]
            for turn in item["conversations"]
            if turn["role"] == "music"
        })
        for target_turn in target_turns:
            target = get_turn_target(item["conversations"], target_turn)
            if not target:
                continue
            tasks.append({
                "session_id": item["session_id"],
                "turn_number": target_turn,
                "target": target,
                "bm25_query": build_query_text(
                    item, target_turn, track_texts, args.bm25_query_mode, args.history_turns
                ),
                "dense_query": build_query_text(
                    item, target_turn, track_texts, args.dense_query_mode, args.history_turns
                ),
                "reranker_query": build_query_text(
                    item, target_turn, track_texts, args.reranker_query_mode, args.history_turns
                ),
            })

    bm25_lists = bm25.batch_text_to_item_retrieval(
        [task["bm25_query"] for task in tasks],
        topk=args.bm25_topk,
    )
    dense = QwenDenseRetriever(
        dataset_name=args.embedding_dataset_name,
        split=args.embedding_split,
        embedding_field=args.embedding_field,
        model_name=args.embedding_model_name,
        cache_dir=config.cache_dir,
        device=args.device,
        max_length=args.dense_max_length,
        query_batch_size=args.dense_batch_size,
        task_instruction=args.task_instruction,
    )
    dense_lists = dense.batch_text_to_item_retrieval(
        [task["dense_query"] for task in tasks],
        topk=args.dense_topk,
    )
    dense.unload()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.reranker_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.reranker_dir)
    model.to(device)

    records_by_penalty: dict[float, list[dict[str, Any]]] = {
        penalty: [] for penalty in args.dense_penalties
    }
    for task, bm25_candidates, dense_candidates in tqdm(
        zip(tasks, bm25_lists, dense_lists),
        total=len(tasks),
        desc="Reranking hybrid Dev candidates",
    ):
        target = task["target"]
        combined = union_candidates(bm25_candidates, dense_candidates)
        scores = score_candidates(
            model,
            tokenizer,
            device,
            task["reranker_query"],
            [track_texts[track_id] for track_id in combined],
            args.rerank_batch_size,
            args.max_length,
        )
        bm25_hit = float(target in bm25_candidates)
        dense_hit = float(target in dense_candidates)
        bm25_set = set(bm25_candidates)
        for penalty in args.dense_penalties:
            adjusted_scores = [
                score - penalty if track_id not in bm25_set else score
                for track_id, score in zip(combined, scores)
            ]
            ranked = [
                track_id
                for track_id, _ in sorted(
                    zip(combined, adjusted_scores),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
            ]
            final_rank = rank_of(ranked, target)
            records_by_penalty[penalty].append({
                "session_id": task["session_id"],
                "turn_number": task["turn_number"],
                "bm25_hit": bm25_hit,
                "dense_hit": dense_hit,
                "union_hit": float(target in combined),
                "dense_incremental_hit": float(not bm25_hit and dense_hit),
                "dense_incremental_final_hit": float(
                    not bm25_hit and dense_hit and final_rank is not None and final_rank <= 20
                ),
                "final_rank": final_rank,
                "final_recall@20": float(final_rank is not None and final_rank <= 20),
                "final_ndcg@20": ndcg(final_rank),
                "union_size": len(combined),
            })

    report = {
        "settings": {
            "sampled_sessions": args.max_sessions,
            "seed": args.seed,
            "bm25_topk": args.bm25_topk,
            "dense_topk": args.dense_topk,
            "bm25_query_mode": args.bm25_query_mode,
            "dense_query_mode": args.dense_query_mode,
            "reranker_query_mode": args.reranker_query_mode,
            "history_turns": args.history_turns,
            "reranker_dir": args.reranker_dir,
            "embedding_field": args.embedding_field,
            "dense_penalties": args.dense_penalties,
        },
        "metrics_by_dense_penalty": {
            str(penalty): summarize(records)
            for penalty, records in records_by_penalty.items()
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"Saved report to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hybrid retrieval with a reranker on Dev.")
    parser.add_argument("--tid", default="bm25_tags_doubao_blindset_A")
    parser.add_argument("--dev_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--dev_split", default="test")
    parser.add_argument("--reranker_dir", default="./exp/reranker/minilm_bm25_tags_top400_e1")
    parser.add_argument("--output_path", default="exp/evaluation/dev_hybrid_reranked.json")
    parser.add_argument("--corpus_types", nargs="+", default=None)
    parser.add_argument("--bm25_topk", type=int, default=400)
    parser.add_argument("--dense_topk", type=int, default=200)
    parser.add_argument("--bm25_query_mode", choices=["focused", "legacy"], default="legacy")
    parser.add_argument("--dense_query_mode", choices=["focused", "legacy"], default="focused")
    parser.add_argument("--reranker_query_mode", choices=["focused", "legacy"], default="focused")
    parser.add_argument("--history_turns", type=int, default=3)
    parser.add_argument("--embedding_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    parser.add_argument("--embedding_split", default="all_tracks")
    parser.add_argument("--embedding_field", default="metadata-qwen3_embedding_0.6b")
    parser.add_argument("--embedding_model_name", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--task_instruction", default=DEFAULT_TASK_INSTRUCTION)
    parser.add_argument("--dense_max_length", type=int, default=512)
    parser.add_argument("--dense_batch_size", type=int, default=16)
    parser.add_argument("--rerank_batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--dense_penalties", nargs="+", type=float, default=[0.0])
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None)
    main(parser.parse_args())
