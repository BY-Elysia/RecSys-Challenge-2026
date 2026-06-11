"""Mine model-scored hard negatives from BM25 candidates and cache them as JSONL."""

import argparse
import json
import os
import random
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mcrs.retrieval_modules.bm25 import BM25_MODEL
from train_reranker import build_query_text, get_turn_target, is_clean_negative, track_to_text


def example_key(session_id: str, target_turn: int) -> str:
    return f"{session_id}:{target_turn}"


def load_completed_keys(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
            keys.add(example_key(record["session_id"], record["target_turn"]))
    return keys


def score_tracks(
    model,
    tokenizer,
    device,
    query: str,
    track_ids: list[str],
    track_texts: dict[str, str],
    batch_size: int,
    max_length: int,
) -> list[float]:
    scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(track_ids), batch_size):
            batch_ids = track_ids[start:start + batch_size]
            encoded = tokenizer(
                [query] * len(batch_ids),
                [track_texts[track_id] for track_id in batch_ids],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            scores.extend(model(**encoded).logits.squeeze(-1).detach().cpu().tolist())
    return scores


def write_metadata(args, stats: dict[str, Any]) -> None:
    metadata_path = f"{args.output_path}.meta.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump({"args": vars(args), "stats": stats}, file, indent=2, ensure_ascii=False)


def finish_mining(output, args, stats: dict[str, Any]) -> None:
    output.flush()
    stats["total_cached_records"] = stats["cache_records_at_start"] + stats["records_written"]
    write_metadata(args, stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Saved hard negatives to {args.output_path}")


def main(args) -> None:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    retriever = BM25_MODEL(args.item_db_name, ["all_tracks"], args.corpus_types, args.cache_dir)
    track_texts = {track_id: track_to_text(metadata) for track_id, metadata in retriever.metadata_dict.items()}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.to(device)

    train_db = load_dataset(args.train_dataset_name, split="train", cache_dir=args.cache_dir)
    sessions = list(train_db)
    rng.shuffle(sessions)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    completed_keys = load_completed_keys(args.output_path) if args.resume else set()
    mode = "a" if args.resume else "w"
    stats: dict[str, Any] = {
        "eligible_turns": 0,
        "positive_not_retrieved": 0,
        "cache_records_at_start": len(completed_keys),
        "already_cached": 0,
        "records_written": 0,
        "clean_candidates_scored": 0,
        "hard_negatives_above_positive": 0,
    }

    with open(args.output_path, mode, encoding="utf-8") as output:
        for item in tqdm(sessions, desc="Mining hard negatives"):
            target_turns = sorted({turn["turn_number"] for turn in item["conversations"]})
            for target_turn in target_turns:
                positive_track_id = get_turn_target(item["conversations"], target_turn)
                if not positive_track_id or positive_track_id not in track_texts:
                    continue

                stats["eligible_turns"] += 1
                key = example_key(item["session_id"], target_turn)
                if key in completed_keys:
                    stats["already_cached"] += 1
                    if args.max_turns and stats["eligible_turns"] >= args.max_turns:
                        finish_mining(output, args, stats)
                        return
                    continue

                retrieval_query = build_query_text(
                    item,
                    target_turn,
                    track_texts,
                    query_mode=args.retrieval_query_mode,
                    history_turns=args.history_turns,
                )
                candidates = retriever.text_to_item_retrieval(retrieval_query, topk=args.retrieval_topk)
                if positive_track_id not in candidates:
                    stats["positive_not_retrieved"] += 1
                    if args.max_turns and stats["eligible_turns"] >= args.max_turns:
                        finish_mining(output, args, stats)
                        return
                    continue

                positive_metadata = retriever.metadata_dict[positive_track_id]
                clean_candidate_ids = []
                for track_id in candidates:
                    if track_id == positive_track_id:
                        continue
                    if is_clean_negative(
                        positive_metadata,
                        retriever.metadata_dict[track_id],
                        args.exclude_same_artist,
                        args.exclude_same_album,
                        args.max_negative_tag_jaccard,
                    ):
                        clean_candidate_ids.append(track_id)
                    if len(clean_candidate_ids) >= args.candidate_pool_size:
                        break

                reranker_query = build_query_text(
                    item,
                    target_turn,
                    track_texts,
                    query_mode=args.reranker_query_mode,
                    history_turns=args.history_turns,
                )
                scored_ids = [positive_track_id] + clean_candidate_ids
                scores = score_tracks(
                    model,
                    tokenizer,
                    device,
                    reranker_query,
                    scored_ids,
                    track_texts,
                    args.rerank_batch_size,
                    args.max_length,
                )
                positive_score = scores[0]
                ranked_negatives = sorted(
                    zip(clean_candidate_ids, scores[1:]),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
                hard_negative_ids = [
                    track_id for track_id, _ in ranked_negatives[:args.hard_negatives_per_positive]
                ]
                stats["clean_candidates_scored"] += len(clean_candidate_ids)
                stats["hard_negatives_above_positive"] += sum(
                    score > positive_score for _, score in ranked_negatives[:args.hard_negatives_per_positive]
                )

                record = {
                    "session_id": item["session_id"],
                    "target_turn": target_turn,
                    "positive_track_id": positive_track_id,
                    "retrieval_rank": candidates.index(positive_track_id) + 1,
                    "positive_score": positive_score,
                    "hard_negative_ids": hard_negative_ids,
                    "hard_negative_scores": [
                        score for _, score in ranked_negatives[:args.hard_negatives_per_positive]
                    ],
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["records_written"] += 1
                if stats["records_written"] % args.flush_every == 0:
                    output.flush()
                    write_metadata(args, stats)

                if args.max_turns and stats["eligible_turns"] >= args.max_turns:
                    finish_mining(output, args, stats)
                    return

        finish_mining(output, args, stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine model-scored hard negatives for Music CRS reranker training.")
    parser.add_argument("--train_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--item_db_name", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument("--model_name", default="./exp/reranker/minilm_bm25_tags_top400_e1")
    parser.add_argument("--output_path", default="./exp/hard_negatives/tags_top400_e1_pool100.jsonl")
    parser.add_argument(
        "--corpus_types",
        nargs="+",
        default=["track_name", "artist_name", "album_name", "release_date", "tag_list"],
    )
    parser.add_argument("--retrieval_topk", type=int, default=400)
    parser.add_argument("--candidate_pool_size", type=int, default=100)
    parser.add_argument("--hard_negatives_per_positive", type=int, default=12)
    parser.add_argument("--max_turns", type=int, default=50000)
    parser.add_argument("--rerank_batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--retrieval_query_mode", choices=["focused", "legacy"], default="legacy")
    parser.add_argument("--reranker_query_mode", choices=["focused", "legacy"], default="focused")
    parser.add_argument("--history_turns", type=int, default=3)
    parser.add_argument("--exclude_same_artist", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude_same_album", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_negative_tag_jaccard", type=float, default=0.8)
    parser.add_argument("--flush_every", type=int, default=25)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None)
    main(parser.parse_args())
