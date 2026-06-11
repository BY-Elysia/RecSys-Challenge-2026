"""Cache dense retrieval candidates for reranker training."""

import argparse
import json
import os
import random
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

from mcrs.retrieval_modules.bm25 import BM25_MODEL
from mcrs.retrieval_modules.qwen_dense import DEFAULT_TASK_INSTRUCTION, QwenDenseRetriever
from train_reranker import build_query_text, get_turn_target, track_to_text


def example_key(session_id: str, target_turn: int) -> str:
    return f"{session_id}:{target_turn}"


def load_completed_cache_stats(path: str) -> tuple[set[str], int]:
    if not os.path.exists(path):
        return set(), 0
    keys = set()
    positive_hits = 0
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid dense-candidate JSONL at {path}:{line_number}") from exc
            keys.add(example_key(record["session_id"], record["target_turn"]))
            positive_hits += int(record.get("dense_rank") is not None)
    return keys, positive_hits


def write_metadata(args, stats: dict[str, Any]) -> None:
    with open(f"{args.output_path}.meta.json", "w", encoding="utf-8") as file:
        json.dump({"args": vars(args), "stats": stats}, file, indent=2, ensure_ascii=False)


def flush_tasks(output, dense, tasks: list[dict[str, Any]], args, stats: dict[str, Any]) -> None:
    if not tasks:
        return
    candidate_lists = dense.batch_text_to_item_retrieval(
        [task["query"] for task in tasks],
        topk=args.dense_topk,
    )
    for task, candidates in zip(tasks, candidate_lists):
        positive = task["positive_track_id"]
        dense_rank = candidates.index(positive) + 1 if positive in candidates else None
        if dense_rank is not None:
            stats["dense_positive_hits"] += 1
        record = {
            "session_id": task["session_id"],
            "target_turn": task["target_turn"],
            "positive_track_id": positive,
            "dense_rank": dense_rank,
            "dense_candidate_ids": candidates,
        }
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats["records_written"] += 1
    output.flush()
    write_metadata(args, stats)
    tasks.clear()


def main(args) -> None:
    rng = random.Random(args.seed)
    bm25 = BM25_MODEL(args.item_db_name, ["all_tracks"], args.corpus_types, args.cache_dir)
    track_texts = {track_id: track_to_text(metadata) for track_id, metadata in bm25.metadata_dict.items()}
    dense = QwenDenseRetriever(
        dataset_name=args.embedding_dataset_name,
        split=args.embedding_split,
        embedding_field=args.embedding_field,
        model_name=args.embedding_model_name,
        cache_dir=args.cache_dir,
        device=args.device,
        max_length=args.dense_max_length,
        query_batch_size=args.dense_batch_size,
        task_instruction=args.task_instruction,
    )

    train_db = load_dataset(args.train_dataset_name, split="train", cache_dir=args.cache_dir)
    sessions = list(train_db)
    rng.shuffle(sessions)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    if args.resume:
        completed_keys, cached_positive_hits = load_completed_cache_stats(args.output_path)
    else:
        completed_keys, cached_positive_hits = set(), 0
    stats: dict[str, Any] = {
        "eligible_turns": 0,
        "cache_records_at_start": len(completed_keys),
        "cache_positive_hits_at_start": cached_positive_hits,
        "already_cached": 0,
        "records_written": 0,
        "dense_positive_hits": 0,
    }
    tasks: list[dict[str, Any]] = []
    mode = "a" if args.resume else "w"

    with open(args.output_path, mode, encoding="utf-8") as output:
        stop = False
        for item in tqdm(sessions, desc="Preparing dense training candidates"):
            target_turns = sorted({turn["turn_number"] for turn in item["conversations"]})
            for target_turn in target_turns:
                positive = get_turn_target(item["conversations"], target_turn)
                if not positive or positive not in track_texts:
                    continue

                stats["eligible_turns"] += 1
                key = example_key(item["session_id"], target_turn)
                if key in completed_keys:
                    stats["already_cached"] += 1
                else:
                    tasks.append({
                        "session_id": item["session_id"],
                        "target_turn": target_turn,
                        "positive_track_id": positive,
                        "query": build_query_text(
                            item,
                            target_turn,
                            track_texts,
                            query_mode=args.dense_query_mode,
                            history_turns=args.history_turns,
                        ),
                    })
                    if len(tasks) >= args.write_batch_size:
                        flush_tasks(output, dense, tasks, args, stats)

                if args.max_turns and stats["eligible_turns"] >= args.max_turns:
                    stop = True
                    break
            if stop:
                break

        flush_tasks(output, dense, tasks, args, stats)

    stats["total_cached_records"] = stats["cache_records_at_start"] + stats["records_written"]
    stats["total_dense_positive_hits"] = stats["cache_positive_hits_at_start"] + stats["dense_positive_hits"]
    stats["new_dense_recall"] = stats["dense_positive_hits"] / max(1, stats["records_written"])
    stats["dense_recall"] = stats["total_dense_positive_hits"] / max(1, stats["total_cached_records"])
    write_metadata(args, stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Saved dense candidates to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache Qwen dense candidates for mixed reranker training.")
    parser.add_argument("--train_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--item_db_name", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument("--output_path", default="./exp/dense_candidates/train_dense_top100.jsonl")
    parser.add_argument(
        "--corpus_types",
        nargs="+",
        default=["track_name", "artist_name", "album_name", "release_date", "tag_list"],
    )
    parser.add_argument("--dense_topk", type=int, default=100)
    parser.add_argument("--dense_query_mode", choices=["focused", "legacy"], default="focused")
    parser.add_argument("--history_turns", type=int, default=3)
    parser.add_argument("--embedding_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    parser.add_argument("--embedding_split", default="all_tracks")
    parser.add_argument("--embedding_field", default="metadata-qwen3_embedding_0.6b")
    parser.add_argument("--embedding_model_name", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--task_instruction", default=DEFAULT_TASK_INSTRUCTION)
    parser.add_argument("--dense_max_length", type=int, default=512)
    parser.add_argument("--dense_batch_size", type=int, default=16)
    parser.add_argument("--write_batch_size", type=int, default=256)
    parser.add_argument("--max_turns", type=int, default=50000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None)
    main(parser.parse_args())
