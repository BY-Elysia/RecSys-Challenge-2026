"""Evaluate the complete BM25 + reranker pipeline on the official Dev split."""

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

from mcrs.retrieval_modules.bm25 import BM25_MODEL
from train_reranker import build_query_text, get_turn_target, track_to_text


def score_candidates(
    model,
    tokenizer,
    device,
    query: str,
    candidate_texts: list[str],
    batch_size: int,
    max_length: int,
) -> list[float]:
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(candidate_texts), batch_size):
            tracks = candidate_texts[start:start + batch_size]
            encoded = tokenizer(
                [query] * len(tracks),
                tracks,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            scores.extend(model(**encoded).logits.squeeze(-1).detach().cpu().tolist())
    return scores


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank is not None else 0.0


def ndcg(rank: int | None, cutoff: int) -> float:
    if rank is None or rank > cutoff:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def average(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def macro_by_session(records: list[dict[str, Any]], metric: str) -> float:
    session_values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        session_values[record["session_id"]].append(float(record[metric]))
    return average([average(values) for values in session_values.values()])


def summarize(records: list[dict[str, Any]], candidate_cutoffs: list[int], final_cutoffs: list[int]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "sessions": len({record["session_id"] for record in records}),
        "turns": len(records),
    }
    for cutoff in candidate_cutoffs:
        metric = f"candidate_recall@{cutoff}"
        report[metric] = macro_by_session(records, metric)
    report["candidate_mrr"] = macro_by_session(records, "candidate_mrr")

    for cutoff in final_cutoffs:
        for prefix in ["final_recall", "final_ndcg"]:
            metric = f"{prefix}@{cutoff}"
            report[metric] = macro_by_session(records, metric)
    report["final_mrr"] = macro_by_session(records, "final_mrr")

    retrieved = [record for record in records if record["candidate_rank"] is not None]
    report["retrieved_turns"] = len(retrieved)
    report["conditional_final_ndcg@20"] = average([record["final_ndcg@20"] for record in retrieved])

    per_turn = {}
    for turn_number in sorted({record["turn_number"] for record in records}):
        turn_records = [record for record in records if record["turn_number"] == turn_number]
        per_turn[str(turn_number)] = {
            "turns": len(turn_records),
            f"candidate_recall@{max(candidate_cutoffs)}": average(
                [record[f"candidate_recall@{max(candidate_cutoffs)}"] for record in turn_records]
            ),
            "final_ndcg@20": average([record["final_ndcg@20"] for record in turn_records]),
        }
    report["per_turn"] = per_turn

    user_segments = {}
    for segment in ["seen_train_user", "unseen_train_user"]:
        segment_records = [record for record in records if record["user_segment"] == segment]
        if segment_records:
            user_segments[segment] = {
                "sessions": len({record["session_id"] for record in segment_records}),
                f"candidate_recall@{max(candidate_cutoffs)}": macro_by_session(
                    segment_records, f"candidate_recall@{max(candidate_cutoffs)}"
                ),
                "final_ndcg@20": macro_by_session(segment_records, "final_ndcg@20"),
            }
    report["user_segments"] = user_segments
    return report


def main(args) -> None:
    config = OmegaConf.load(f"config/{args.tid}.yaml")
    corpus_types = args.corpus_types or list(config.corpus_types)
    retriever = BM25_MODEL(config.item_db_name, config.track_split_types, corpus_types, config.cache_dir)
    track_texts = {track_id: track_to_text(metadata) for track_id, metadata in retriever.metadata_dict.items()}

    model = None
    tokenizer = None
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.reranker_dir:
        tokenizer = AutoTokenizer.from_pretrained(args.reranker_dir, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.reranker_dir)
        model.to(device)

    dataset = load_dataset(args.dev_dataset_name, split=args.dev_split, cache_dir=config.cache_dir)
    if args.max_sessions:
        indices = list(range(len(dataset)))
        random.Random(args.seed).shuffle(indices)
        dataset = dataset.select(indices[:min(args.max_sessions, len(dataset))])

    train_user_ids: set[str] = set()
    if not args.skip_user_segments:
        train_dataset = load_dataset(args.dev_dataset_name, split="train", cache_dir=config.cache_dir)
        train_user_ids = set(train_dataset["user_id"])

    candidate_cutoffs = sorted(set(args.candidate_cutoffs + [args.candidate_topk]))
    final_cutoffs = sorted(set(args.final_cutoffs + [20]))
    tasks: list[dict[str, Any]] = []
    for item in tqdm(dataset, desc="Building Dev queries"):
        target_turns = sorted({
            turn["turn_number"]
            for turn in item["conversations"]
            if turn["role"] == "music"
        })
        for target_turn in target_turns:
            target_track_id = get_turn_target(item["conversations"], target_turn)
            if not target_track_id:
                continue

            retrieval_query = build_query_text(
                item,
                target_turn,
                track_texts,
                query_mode=args.retrieval_query_mode,
                history_turns=args.history_turns,
            )
            reranker_query = build_query_text(
                item,
                target_turn,
                track_texts,
                query_mode=args.reranker_query_mode,
                history_turns=args.history_turns,
            )
            tasks.append({
                "session_id": item["session_id"],
                "user_id": item["user_id"],
                "turn_number": target_turn,
                "target_track_id": target_track_id,
                "retrieval_query": retrieval_query,
                "reranker_query": reranker_query,
            })

    candidate_lists = retriever.batch_text_to_item_retrieval(
        [task["retrieval_query"] for task in tasks],
        topk=args.candidate_topk,
    )
    records: list[dict[str, Any]] = []
    for task, candidates in tqdm(zip(tasks, candidate_lists), total=len(tasks), desc="Evaluating Dev turns"):
        target_track_id = task["target_track_id"]
        candidate_rank = candidates.index(target_track_id) + 1 if target_track_id in candidates else None

        if model is None:
            final_rank = candidate_rank
        else:
            candidate_texts = [track_texts[track_id] for track_id in candidates]
            scores = score_candidates(
                model,
                tokenizer,
                device,
                task["reranker_query"],
                candidate_texts,
                args.rerank_batch_size,
                args.max_length,
            )
            ranked = [
                track_id
                for track_id, _ in sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
            ]
            final_rank = ranked.index(target_track_id) + 1 if target_track_id in ranked else None

        record: dict[str, Any] = {
            "session_id": task["session_id"],
            "turn_number": task["turn_number"],
            "candidate_rank": candidate_rank,
            "final_rank": final_rank,
            "candidate_mrr": reciprocal_rank(candidate_rank),
            "final_mrr": reciprocal_rank(final_rank),
            "user_segment": (
                "not_evaluated"
                if args.skip_user_segments
                else ("seen_train_user" if task["user_id"] in train_user_ids else "unseen_train_user")
            ),
        }
        for cutoff in candidate_cutoffs:
            record[f"candidate_recall@{cutoff}"] = float(candidate_rank is not None and candidate_rank <= cutoff)
        for cutoff in final_cutoffs:
            record[f"final_recall@{cutoff}"] = float(final_rank is not None and final_rank <= cutoff)
            record[f"final_ndcg@{cutoff}"] = ndcg(final_rank, cutoff)
        records.append(record)

    report = {
        "settings": {
            "dev_dataset_name": args.dev_dataset_name,
            "dev_split": args.dev_split,
            "corpus_types": corpus_types,
            "candidate_topk": args.candidate_topk,
            "reranker_dir": args.reranker_dir,
            "retrieval_query_mode": args.retrieval_query_mode,
            "reranker_query_mode": args.reranker_query_mode,
            "history_turns": args.history_turns,
        },
        "metrics": summarize(records, candidate_cutoffs, final_cutoffs),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(f"Saved report to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate full Music CRS retrieval and reranking on official Dev.")
    parser.add_argument("--tid", default="bm25_tags_doubao_blindset_A")
    parser.add_argument("--dev_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--dev_split", default="test")
    parser.add_argument("--reranker_dir", default=None)
    parser.add_argument("--output_path", default="exp/evaluation/dev_tags_top400.json")
    parser.add_argument("--corpus_types", nargs="+", default=None)
    parser.add_argument("--candidate_topk", type=int, default=400)
    parser.add_argument("--candidate_cutoffs", nargs="+", type=int, default=[20, 100, 200, 400])
    parser.add_argument("--final_cutoffs", nargs="+", type=int, default=[1, 10, 20])
    parser.add_argument("--rerank_batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--retrieval_query_mode", choices=["focused", "legacy"], default="legacy")
    parser.add_argument("--reranker_query_mode", choices=["focused", "legacy"], default="focused")
    parser.add_argument("--history_turns", type=int, default=3)
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--skip_user_segments", action="store_true")
    parser.add_argument("--device", default=None)
    main(parser.parse_args())
