"""Measure whether dense retrieval recovers Dev targets missed by BM25."""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Any

from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

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


def first_rank(candidates: list[str], target: str) -> int | None:
    return candidates.index(target) + 1 if target in candidates else None


def union_candidates(primary: list[str], supplement: list[str]) -> list[str]:
    seen = set(primary)
    return primary + [track_id for track_id in supplement if track_id not in seen]


def summarize(records: list[dict[str, Any]], bm25_topk: int, dense_topk: int) -> dict[str, Any]:
    metrics = {
        "sessions": len({record["session_id"] for record in records}),
        "turns": len(records),
        f"bm25_recall@{bm25_topk}": macro_by_session(records, "bm25_hit"),
        f"dense_recall@{dense_topk}": macro_by_session(records, "dense_hit"),
        f"union_recall@{bm25_topk}+{dense_topk}": macro_by_session(records, "union_hit"),
        "dense_incremental_recall": macro_by_session(records, "dense_incremental_hit"),
        "bm25_unique_recall": macro_by_session(records, "bm25_unique_hit"),
        "both_hit_recall": macro_by_session(records, "both_hit"),
        "average_dense_overlap_with_bm25": average([record["dense_overlap"] for record in records]),
        "average_union_size": average([record["union_size"] for record in records]),
        "dense_incremental_targets": int(sum(record["dense_incremental_hit"] for record in records)),
        "bm25_targets": int(sum(record["bm25_hit"] for record in records)),
        "dense_targets": int(sum(record["dense_hit"] for record in records)),
        "union_targets": int(sum(record["union_hit"] for record in records)),
    }
    per_turn = {}
    for turn_number in sorted({record["turn_number"] for record in records}):
        turn_records = [record for record in records if record["turn_number"] == turn_number]
        per_turn[str(turn_number)] = {
            "turns": len(turn_records),
            "bm25_recall": average([record["bm25_hit"] for record in turn_records]),
            "dense_recall": average([record["dense_hit"] for record in turn_records]),
            "union_recall": average([record["union_hit"] for record in turn_records]),
            "dense_incremental_recall": average(
                [record["dense_incremental_hit"] for record in turn_records]
            ),
        }
    metrics["per_turn"] = per_turn
    return metrics


def main(args) -> None:
    config = OmegaConf.load(f"config/{args.tid}.yaml")
    corpus_types = args.corpus_types or list(config.corpus_types)
    bm25 = BM25_MODEL(config.item_db_name, config.track_split_types, corpus_types, config.cache_dir)
    track_texts = {track_id: track_to_text(metadata) for track_id, metadata in bm25.metadata_dict.items()}

    dataset = load_dataset(
        args.dev_dataset_name,
        split=args.dev_split,
        cache_dir=config.cache_dir,
    )
    if args.max_sessions:
        indices = list(range(len(dataset)))
        random.Random(args.seed).shuffle(indices)
        dataset = dataset.select(indices[:min(args.max_sessions, len(dataset))])

    tasks: list[dict[str, Any]] = []
    for item in tqdm(dataset, desc="Building hybrid retrieval queries"):
        target_turns = sorted({
            turn["turn_number"]
            for turn in item["conversations"]
            if turn["role"] == "music"
        })
        for target_turn in target_turns:
            target_track_id = get_turn_target(item["conversations"], target_turn)
            if not target_track_id:
                continue
            tasks.append({
                "session_id": item["session_id"],
                "turn_number": target_turn,
                "target_track_id": target_track_id,
                "bm25_query": build_query_text(
                    item,
                    target_turn,
                    track_texts,
                    query_mode=args.bm25_query_mode,
                    history_turns=args.history_turns,
                ),
                "dense_query": build_query_text(
                    item,
                    target_turn,
                    track_texts,
                    query_mode=args.dense_query_mode,
                    history_turns=args.history_turns,
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

    records = []
    for task, bm25_candidates, dense_candidates in zip(tasks, bm25_lists, dense_lists):
        target = task["target_track_id"]
        combined = union_candidates(bm25_candidates, dense_candidates)
        bm25_rank = first_rank(bm25_candidates, target)
        dense_rank = first_rank(dense_candidates, target)
        union_rank = first_rank(combined, target)
        bm25_hit = float(bm25_rank is not None)
        dense_hit = float(dense_rank is not None)
        records.append({
            "session_id": task["session_id"],
            "turn_number": task["turn_number"],
            "bm25_rank": bm25_rank,
            "dense_rank": dense_rank,
            "union_rank": union_rank,
            "bm25_hit": bm25_hit,
            "dense_hit": dense_hit,
            "union_hit": float(union_rank is not None),
            "dense_incremental_hit": float(not bm25_hit and dense_hit),
            "bm25_unique_hit": float(bm25_hit and not dense_hit),
            "both_hit": float(bm25_hit and dense_hit),
            "dense_overlap": len(set(bm25_candidates) & set(dense_candidates)) / max(1, len(dense_candidates)),
            "union_size": len(combined),
        })

    report = {
        "settings": {
            "dev_dataset_name": args.dev_dataset_name,
            "dev_split": args.dev_split,
            "sampled_sessions": args.max_sessions,
            "seed": args.seed,
            "corpus_types": corpus_types,
            "bm25_topk": args.bm25_topk,
            "dense_topk": args.dense_topk,
            "bm25_query_mode": args.bm25_query_mode,
            "dense_query_mode": args.dense_query_mode,
            "history_turns": args.history_turns,
            "embedding_dataset_name": args.embedding_dataset_name,
            "embedding_split": args.embedding_split,
            "embedding_field": args.embedding_field,
            "embedding_model_name": args.embedding_model_name,
            "task_instruction": args.task_instruction,
        },
        "metrics": summarize(records, args.bm25_topk, args.dense_topk),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"Saved report to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BM25 plus dense candidate recall on Dev.")
    parser.add_argument("--tid", default="bm25_tags_doubao_blindset_A")
    parser.add_argument("--dev_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--dev_split", default="test")
    parser.add_argument("--output_path", default="exp/evaluation/dev_hybrid_recall.json")
    parser.add_argument("--corpus_types", nargs="+", default=None)
    parser.add_argument("--bm25_topk", type=int, default=400)
    parser.add_argument("--dense_topk", type=int, default=50)
    parser.add_argument("--bm25_query_mode", choices=["focused", "legacy"], default="legacy")
    parser.add_argument("--dense_query_mode", choices=["focused", "legacy"], default="focused")
    parser.add_argument("--history_turns", type=int, default=3)
    parser.add_argument("--embedding_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    parser.add_argument("--embedding_split", default="all_tracks")
    parser.add_argument("--embedding_field", default="metadata-qwen3_embedding_0.6b")
    parser.add_argument("--embedding_model_name", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--task_instruction", default=DEFAULT_TASK_INSTRUCTION)
    parser.add_argument("--dense_max_length", type=int, default=512)
    parser.add_argument("--dense_batch_size", type=int, default=8)
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None)
    main(parser.parse_args())
