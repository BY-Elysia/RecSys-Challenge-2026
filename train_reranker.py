"""Train a listwise cross-encoder reranker for Music CRS retrieval."""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from mcrs.retrieval_modules.bm25 import BM25_MODEL


def _join_value(value: Any, limit: int | None = None) -> str:
    if isinstance(value, list):
        if limit is not None:
            value = value[:limit]
        return ", ".join(str(item) for item in value)
    return str(value)


def track_to_text(metadata: dict[str, Any]) -> str:
    fields = [
        ("track", _join_value(metadata.get("track_name", []))),
        ("artist", _join_value(metadata.get("artist_name", []))),
        ("album", _join_value(metadata.get("album_name", []))),
        ("tags", _join_value(metadata.get("tag_list", []), limit=24)),
        ("release_date", metadata.get("release_date", "")),
        ("popularity", metadata.get("popularity", "")),
    ]
    return "\n".join(f"{name}: {value}" for name, value in fields if value not in ("", [], None))


def _compact_track_text(track_text: str) -> str:
    fields = []
    for line in track_text.splitlines():
        if line.startswith(("track:", "artist:", "album:")):
            fields.append(line)
        elif line.startswith("tags:"):
            tags = [tag.strip() for tag in line.removeprefix("tags:").split(",")[:8]]
            fields.append(f"tags: {', '.join(tags)}")
    return "; ".join(fields)


def _current_user_request(item: dict[str, Any], target_turn: int) -> str:
    for turn in item["conversations"]:
        if turn["turn_number"] == target_turn and turn["role"] == "user":
            return str(turn["content"])
    return ""


def build_query_text(
    item: dict[str, Any],
    target_turn: int,
    track_texts: dict[str, str],
    query_mode: str = "focused",
    history_turns: int = 3,
) -> str:
    """Build a current-request-first query while removing unrelated dialogue noise."""
    if query_mode == "legacy":
        profile = item.get("user_profile", {})
        goal = item.get("conversation_goal", {})
        parts = [
            "User profile:",
            "; ".join(f"{key}: {value}" for key, value in profile.items()),
            "Conversation goal:",
            "; ".join(f"{key}: {value}" for key, value in goal.items()),
            "Conversation history:",
        ]
        current_query = ""
        for turn in item["conversations"]:
            role = turn["role"]
            content = turn["content"]
            turn_number = turn["turn_number"]
            if turn_number == target_turn and role == "user":
                current_query = content
            if turn_number >= target_turn:
                continue
            if role == "music":
                content = track_texts.get(content, f"track_id: {content}")
                role = "assistant_recommended_music"
            parts.append(f"turn {turn_number} {role}: {content}")
        parts.extend(["Current user request:", current_query])
        return "\n".join(parts)

    profile = item.get("user_profile", {})
    goal = item.get("conversation_goal", {})
    parts = [f"Current request: {_current_user_request(item, target_turn)}"]

    listener_goal = goal.get("listener_goal")
    if listener_goal:
        parts.append(f"Overall listener goal: {listener_goal}")

    musical_culture = profile.get("preferred_musical_culture")
    if musical_culture:
        parts.append(f"Music preference: {musical_culture}")

    start_turn = max(1, target_turn - history_turns)
    recent_context = []
    for turn in item["conversations"]:
        turn_number = turn["turn_number"]
        role = turn["role"]
        if turn_number < start_turn or turn_number >= target_turn:
            continue
        if role == "user":
            recent_context.append(f"Recent user feedback: {turn['content']}")
        elif role == "music":
            track_text = track_texts.get(turn["content"], str(turn["content"]))
            recent_context.append(f"Previously recommended: {_compact_track_text(track_text)}")

    if recent_context:
        parts.extend(recent_context)
    return "\n".join(parts)


def get_turn_target(conversations: list[dict[str, Any]], target_turn: int) -> str | None:
    for turn in conversations:
        if turn["turn_number"] == target_turn and turn["role"] == "music":
            return turn["content"]
    return None


def _normalized_set(metadata: dict[str, Any], field: str) -> set[str]:
    value = metadata.get(field, [])
    if not isinstance(value, list):
        value = [value]
    return {str(item).strip().lower() for item in value if str(item).strip()}


def _tag_jaccard(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_tags = _normalized_set(left, "tag_list")
    right_tags = _normalized_set(right, "tag_list")
    union = left_tags | right_tags
    return len(left_tags & right_tags) / len(union) if union else 0.0


def is_clean_negative(
    positive: dict[str, Any],
    negative: dict[str, Any],
    exclude_same_artist: bool,
    exclude_same_album: bool,
    max_tag_jaccard: float,
) -> bool:
    """Filter candidates that are likely valid alternatives rather than true negatives."""
    if exclude_same_artist and _normalized_set(positive, "artist_id") & _normalized_set(negative, "artist_id"):
        return False
    if exclude_same_album and _normalized_set(positive, "album_id") & _normalized_set(negative, "album_id"):
        return False
    if max_tag_jaccard >= 0 and _tag_jaccard(positive, negative) > max_tag_jaccard:
        return False
    return True


@dataclass
class RankingExample:
    query: str
    tracks: list[str]
    positive_index: int
    retrieval_rank: int | None


class RankingDataset(Dataset):
    def __init__(self, examples: list[RankingExample], tokenizer, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        encoded = self.tokenizer(
            [example.query] * len(example.tracks),
            example.tracks,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["target"] = torch.tensor(example.positive_index, dtype=torch.long)
        return encoded


def unique_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _example_key(session_id: str, target_turn: int) -> str:
    return f"{session_id}:{target_turn}"


def load_hard_negative_cache(path: str | None) -> dict[str, list[str]]:
    if not path:
        return {}
    cache: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid hard-negative JSONL at {path}:{line_number}") from exc
            cache[_example_key(record["session_id"], record["target_turn"])] = record["hard_negative_ids"]
    return cache


def load_dense_candidate_cache(path: str | None) -> dict[str, list[str]]:
    if not path:
        return {}
    cache: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid dense-candidate JSONL at {path}:{line_number}") from exc
            cache[_example_key(record["session_id"], record["target_turn"])] = record["dense_candidate_ids"]
    return cache


def build_examples(
    args,
    bm25: BM25_MODEL,
    track_texts: dict[str, str],
    rng: random.Random,
) -> tuple[list[RankingExample], dict[str, int]]:
    train_db = load_dataset(args.train_dataset_name, split="train")
    all_track_ids = list(track_texts.keys())
    hard_negative_cache = load_hard_negative_cache(args.hard_negative_cache)
    dense_candidate_cache = load_dense_candidate_cache(args.dense_candidate_cache)
    examples: list[RankingExample] = []
    stats = {
        "eligible_turns": 0,
        "positive_not_retrieved": 0,
        "bm25_positive_not_retrieved": 0,
        "retrieval_hit@20": 0,
        "retrieval_hit@100": 0,
        "retrieval_hit@200": 0,
        "retrieval_hit@400": 0,
        "union_hit@20": 0,
        "union_hit@100": 0,
        "union_hit@200": 0,
        "union_hit@400": 0,
        "filtered_negatives": 0,
        "hard_negative_cache_hits": 0,
        "hard_negative_cache_misses": 0,
        "hard_negatives_used": 0,
        "stale_hard_negatives": 0,
        "skipped_missing_hard_negative_cache": 0,
        "dense_candidate_cache_hits": 0,
        "dense_candidate_cache_misses": 0,
        "skipped_missing_dense_candidate_cache": 0,
        "dense_positive_hits": 0,
        "dense_only_positive_hits": 0,
        "union_positive_hits": 0,
        "dense_negatives_used": 0,
        "filtered_dense_negatives": 0,
    }

    iterable = list(train_db)
    rng.shuffle(iterable)

    for item in tqdm(iterable, desc="Building listwise examples"):
        target_turns = sorted({turn["turn_number"] for turn in item["conversations"]})
        for target_turn in target_turns:
            positive_track_id = get_turn_target(item["conversations"], target_turn)
            if not positive_track_id or positive_track_id not in track_texts:
                continue

            stats["eligible_turns"] += 1
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
            candidates = bm25.text_to_item_retrieval(retrieval_query, topk=args.retrieval_topk)
            cache_key = _example_key(item["session_id"], target_turn)
            dense_candidates = dense_candidate_cache.get(cache_key, [])
            if args.dense_candidate_cache:
                if cache_key in dense_candidate_cache:
                    stats["dense_candidate_cache_hits"] += 1
                else:
                    stats["dense_candidate_cache_misses"] += 1
                    if args.require_dense_candidate_cache_hit:
                        stats["skipped_missing_dense_candidate_cache"] += 1
                        if args.max_turns and stats["eligible_turns"] >= args.max_turns:
                            return examples, stats
                        continue

            bm25_candidate_set = set(candidates)
            dense_candidate_set = set(dense_candidates)
            union_candidates = unique_preserve_order(candidates + dense_candidates)
            if positive_track_id in dense_candidate_set:
                stats["dense_positive_hits"] += 1
                if positive_track_id not in bm25_candidate_set:
                    stats["dense_only_positive_hits"] += 1
            if positive_track_id in union_candidates:
                stats["union_positive_hits"] += 1
            try:
                bm25_rank = candidates.index(positive_track_id) + 1
            except ValueError:
                bm25_rank = None
                stats["bm25_positive_not_retrieved"] += 1
            try:
                retrieval_rank = union_candidates.index(positive_track_id) + 1
            except ValueError:
                retrieval_rank = None
                stats["positive_not_retrieved"] += 1
                if not args.allow_positive_outside_retrieval:
                    if args.max_turns and stats["eligible_turns"] >= args.max_turns:
                        return examples, stats
                    continue
            for cutoff in [20, 100, 200, 400]:
                if bm25_rank is not None and bm25_rank <= cutoff:
                    stats[f"retrieval_hit@{cutoff}"] += 1
                if retrieval_rank is not None and retrieval_rank <= cutoff:
                    stats[f"union_hit@{cutoff}"] += 1

            positive_metadata = bm25.metadata_dict[positive_track_id]
            negatives = []
            cached_hard_negatives = hard_negative_cache.get(cache_key)
            if args.hard_negative_cache:
                if cached_hard_negatives is None:
                    stats["hard_negative_cache_misses"] += 1
                    if args.require_hard_negative_cache_hit:
                        stats["skipped_missing_hard_negative_cache"] += 1
                        if args.max_turns and stats["eligible_turns"] >= args.max_turns:
                            return examples, stats
                        continue
                else:
                    stats["hard_negative_cache_hits"] += 1
                    candidate_set = set(union_candidates)
                    for track_id in cached_hard_negatives[:args.hard_negatives_per_positive]:
                        if track_id not in candidate_set or track_id not in track_texts:
                            stats["stale_hard_negatives"] += 1
                            continue
                        if not is_clean_negative(
                            positive_metadata,
                            bm25.metadata_dict[track_id],
                            args.exclude_same_artist,
                            args.exclude_same_album,
                            args.max_negative_tag_jaccard,
                        ):
                            stats["stale_hard_negatives"] += 1
                            continue
                        negatives.append(track_id)
                        stats["hard_negatives_used"] += 1

            dense_negatives_added = 0
            for track_id in dense_candidates[:args.dense_candidate_pool_size]:
                if dense_negatives_added >= args.dense_negatives_per_positive:
                    break
                if track_id == positive_track_id or track_id not in track_texts:
                    continue
                if track_id in bm25_candidate_set or track_id in negatives:
                    continue
                if not is_clean_negative(
                    positive_metadata,
                    bm25.metadata_dict[track_id],
                    args.exclude_same_artist,
                    args.exclude_same_album,
                    args.max_negative_tag_jaccard,
                ):
                    stats["filtered_dense_negatives"] += 1
                    continue
                negatives.append(track_id)
                dense_negatives_added += 1
                stats["dense_negatives_used"] += 1

            for track_id in candidates:
                if track_id == positive_track_id or track_id not in track_texts:
                    continue
                if track_id in negatives:
                    continue
                if not is_clean_negative(
                    positive_metadata,
                    bm25.metadata_dict[track_id],
                    args.exclude_same_artist,
                    args.exclude_same_album,
                    args.max_negative_tag_jaccard,
                ):
                    stats["filtered_negatives"] += 1
                    continue
                negatives.append(track_id)
                if len(negatives) >= args.negatives_per_positive:
                    break

            attempts = 0
            while len(negatives) < args.negatives_per_positive and attempts < 10000:
                attempts += 1
                sampled = rng.choice(all_track_ids)
                if sampled == positive_track_id or sampled in negatives:
                    continue
                if is_clean_negative(
                    positive_metadata,
                    bm25.metadata_dict[sampled],
                    args.exclude_same_artist,
                    args.exclude_same_album,
                    args.max_negative_tag_jaccard,
                ):
                    negatives.append(sampled)

            negatives = unique_preserve_order(negatives)[:args.negatives_per_positive]
            if len(negatives) < args.negatives_per_positive:
                continue

            group = [(positive_track_id, True)] + [(track_id, False) for track_id in negatives]
            rng.shuffle(group)
            positive_index = next(index for index, (_, is_positive) in enumerate(group) if is_positive)
            examples.append(
                RankingExample(
                    query=reranker_query,
                    tracks=[track_texts[track_id] for track_id, _ in group],
                    positive_index=positive_index,
                    retrieval_rank=retrieval_rank,
                )
            )

            if args.max_turns and stats["eligible_turns"] >= args.max_turns:
                return examples, stats

    return examples, stats


def retrieval_metrics(examples: list[RankingExample], cutoffs: list[int]) -> dict[str, float]:
    metrics = {}
    for cutoff in cutoffs:
        hits = sum(example.retrieval_rank is not None and example.retrieval_rank <= cutoff for example in examples)
        metrics[f"retrieval_recall@{cutoff}"] = hits / max(1, len(examples))
    ranks = [example.retrieval_rank for example in examples if example.retrieval_rank is not None]
    metrics["retrieval_mrr"] = sum(1.0 / rank for rank in ranks) / max(1, len(examples))
    return metrics


def _listwise_forward(model, batch: dict[str, torch.Tensor], device, temperature: float):
    targets = batch.pop("target").to(device)
    batch_size, group_size, sequence_length = batch["input_ids"].shape
    flat_batch = {
        key: value.reshape(batch_size * group_size, sequence_length).to(device)
        for key, value in batch.items()
    }
    logits = model(**flat_batch).logits.reshape(batch_size, group_size)
    loss = F.cross_entropy(logits / temperature, targets)
    return loss, logits, targets


def evaluate(model, loader, device, temperature: float) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    ranks = []
    with torch.no_grad():
        for batch in loader:
            loss, logits, targets = _listwise_forward(model, batch, device, temperature)
            total_loss += loss.item()
            order = torch.argsort(logits, dim=1, descending=True)
            batch_ranks = (order == targets.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1
            ranks.extend(batch_ranks.cpu().tolist())

    metrics = {
        "valid_loss": total_loss / max(1, len(loader)),
        "mrr": sum(1.0 / rank for rank in ranks) / max(1, len(ranks)),
    }
    for cutoff in [1, 5, 10, 20]:
        metrics[f"recall@{cutoff}"] = sum(rank <= cutoff for rank in ranks) / max(1, len(ranks))
        metrics[f"ndcg@{cutoff}"] = sum(1.0 / math.log2(rank + 1) if rank <= cutoff else 0.0 for rank in ranks) / max(1, len(ranks))
    return metrics


def main(args) -> None:
    if args.dense_negatives_per_positive > args.negatives_per_positive:
        raise ValueError("dense_negatives_per_positive cannot exceed negatives_per_positive.")
    if args.dense_negatives_per_positive and not args.dense_candidate_cache:
        raise ValueError("dense_candidate_cache is required when dense_negatives_per_positive is non-zero.")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    bm25 = BM25_MODEL(args.item_db_name, ["all_tracks"], args.corpus_types, args.cache_dir)
    track_texts = {track_id: track_to_text(metadata) for track_id, metadata in bm25.metadata_dict.items()}

    examples, build_stats = build_examples(args, bm25, track_texts, rng)
    if args.require_hard_negative_cache_hit and build_stats["hard_negative_cache_misses"]:
        raise RuntimeError(
            "Hard-negative cache is incomplete for this training run: "
            f"{build_stats['hard_negative_cache_misses']} retrieved examples are missing."
        )
    if args.require_dense_candidate_cache_hit and build_stats["dense_candidate_cache_misses"]:
        raise RuntimeError(
            "Dense-candidate cache is incomplete for this training run: "
            f"{build_stats['dense_candidate_cache_misses']} examples are missing."
        )
    if not examples:
        raise RuntimeError("No trainable examples were built. Increase retrieval_topk or allow positives outside retrieval.")
    rng.shuffle(examples)
    valid_size = max(1, int(len(examples) * args.valid_ratio))
    valid_examples = examples[:valid_size]
    train_examples = examples[valid_size:]

    build_report = {"build_stats": build_stats}
    for cutoff in [20, 100, 200, 400]:
        build_report[f"bm25_recall@{cutoff}"] = build_stats[f"retrieval_hit@{cutoff}"] / max(1, build_stats["eligible_turns"])
        build_report[f"union_recall@{cutoff}"] = build_stats[f"union_hit@{cutoff}"] / max(1, build_stats["eligible_turns"])
    build_report["dense_recall"] = build_stats["dense_positive_hits"] / max(1, build_stats["eligible_turns"])
    build_report["union_recall"] = build_stats["union_positive_hits"] / max(1, build_stats["eligible_turns"])
    build_report.update({f"trainable_{key}": value for key, value in retrieval_metrics(valid_examples, [20, 100, 200, 400]).items()})
    print(json.dumps(build_report, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1, ignore_mismatched_sizes=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    train_loader = DataLoader(RankingDataset(train_examples, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(RankingDataset(valid_examples, tokenizer, args.max_length), batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = max(1, optimizer_steps_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_ndcg = -1.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{args.epochs}"), start=1):
            loss, _, _ = _listwise_forward(model, batch, device, args.listwise_temperature)
            (loss / args.gradient_accumulation_steps).backward()
            total_loss += loss.item()

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        metrics = evaluate(model, valid_loader, device, args.listwise_temperature)
        metrics.update({
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, len(train_loader)),
            "train_groups": len(train_examples),
            "valid_groups": len(valid_examples),
        })
        print(json.dumps(metrics, ensure_ascii=False))
        if metrics["ndcg@20"] > best_ndcg:
            best_ndcg = metrics["ndcg@20"]
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a listwise Music CRS cross-encoder reranker.")
    parser.add_argument("--train_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--item_db_name", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument("--model_name", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--output_dir", default="./exp/reranker/minilm_bm25_listwise")
    parser.add_argument("--corpus_types", nargs="+", default=["track_name", "artist_name", "album_name", "release_date"])
    parser.add_argument("--retrieval_topk", type=int, default=200)
    parser.add_argument("--negatives_per_positive", type=int, default=19)
    parser.add_argument("--hard_negative_cache", default=None)
    parser.add_argument("--hard_negatives_per_positive", type=int, default=12)
    parser.add_argument("--require_hard_negative_cache_hit", action="store_true")
    parser.add_argument("--dense_candidate_cache", default=None)
    parser.add_argument("--dense_candidate_pool_size", type=int, default=100)
    parser.add_argument("--dense_negatives_per_positive", type=int, default=0)
    parser.add_argument("--require_dense_candidate_cache_hit", action="store_true")
    parser.add_argument("--max_turns", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4, help="Number of candidate groups per GPU batch.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--valid_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--listwise_temperature", type=float, default=1.0)
    parser.add_argument("--retrieval_query_mode", choices=["focused", "legacy"], default="legacy")
    parser.add_argument("--reranker_query_mode", choices=["focused", "legacy"], default="focused")
    parser.add_argument("--history_turns", type=int, default=3)
    parser.add_argument("--allow_positive_outside_retrieval", action="store_true")
    parser.add_argument("--exclude_same_artist", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude_same_album", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_negative_tag_jaccard", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None)
    main(parser.parse_args())
