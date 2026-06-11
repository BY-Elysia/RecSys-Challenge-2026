"""Evaluate multi-channel retrieval on Dev conversation prefixes."""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any

from datasets import load_dataset

from mcrs.retrieval_modules.bm25 import BM25_MODEL
from mcrs.retrieval_modules.precomputed_embeddings import PrecomputedTrackEmbeddingIndex
from train_reranker import build_query_text, get_turn_target, track_to_text


DEFAULT_BLIND_TURN_WEIGHTS = {1: 20, 2: 15, 3: 10, 4: 5, 5: 8, 6: 9, 7: 8, 8: 5}


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def filter_seen(candidates: list[str], seen: list[str], topk: int) -> list[str]:
    seen_set = set(seen)
    return [track_id for track_id in candidates if track_id not in seen_set][:topk]


def compact_track(track_text: str) -> str:
    return "; ".join(
        line
        for line in track_text.splitlines()
        if line.startswith(("track:", "artist:", "album:", "tags:"))
    )


def current_user_turn(item: dict[str, Any], target_turn: int) -> dict[str, Any]:
    for turn in item["conversations"]:
        if turn["turn_number"] == target_turn and turn["role"] == "user":
            return turn
    raise KeyError(f"No user turn {target_turn} in session {item['session_id']}.")


def feedback_rich_query(
    item: dict[str, Any],
    target_turn: int,
    track_texts: dict[str, str],
    history_turns: int,
) -> str:
    user_turn = current_user_turn(item, target_turn)
    profile = item.get("user_profile", {})
    goal = item.get("conversation_goal", {})
    parts = [
        f"Current request: {user_turn['content']}",
        f"User reasoning: {user_turn.get('thought') or ''}",
        f"Listener goal: {goal.get('listener_goal') or ''}",
        f"Goal category: {goal.get('category') or ''}",
        f"Goal specificity: {goal.get('specificity') or ''}",
        f"Music preference: {profile.get('preferred_musical_culture') or ''}",
    ]
    start_turn = 1 if history_turns <= 0 else max(1, target_turn - history_turns)
    for turn in item["conversations"]:
        turn_number = turn["turn_number"]
        if turn_number < start_turn or turn_number >= target_turn:
            continue
        if turn["role"] == "user":
            parts.append(f"Previous user feedback: {turn['content']}")
            if turn.get("thought"):
                parts.append(f"Previous user reasoning: {turn['thought']}")
        elif turn["role"] == "music":
            parts.append(
                f"Previously recommended: {compact_track(track_texts.get(turn['content'], str(turn['content'])))}"
            )
        elif turn["role"] == "assistant" and turn.get("thought"):
            parts.append(f"Previous recommender reasoning: {turn['thought']}")
    return "\n".join(part for part in parts if not part.endswith(": "))


def build_metadata_indexes(metadata: dict[str, dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    artist_tracks: dict[str, list[str]] = defaultdict(list)
    album_tracks: dict[str, list[str]] = defaultdict(list)
    for track_id, item in metadata.items():
        for artist_id in item.get("artist_id", []):
            artist_tracks[str(artist_id)].append(track_id)
        for album_id in item.get("album_id", []):
            album_tracks[str(album_id)].append(track_id)

    def sort_index(index: dict[str, list[str]]) -> None:
        for key, track_ids in index.items():
            index[key] = sorted(
                track_ids,
                key=lambda track_id: float(metadata[track_id].get("popularity") or 0),
                reverse=True,
            )

    sort_index(artist_tracks)
    sort_index(album_tracks)
    return artist_tracks, album_tracks


def structural_candidates(
    history: list[str],
    metadata: dict[str, dict[str, Any]],
    artist_tracks: dict[str, list[str]],
    album_tracks: dict[str, list[str]],
    topk: int,
) -> list[str]:
    if not history:
        return []
    history_metadata = [metadata[track_id] for track_id in history if track_id in metadata]
    if not history_metadata:
        return []
    last = history_metadata[-1]
    all_artists = unique([
        str(artist_id)
        for item in reversed(history_metadata)
        for artist_id in item.get("artist_id", [])
    ])
    all_albums = unique([
        str(album_id)
        for item in reversed(history_metadata)
        for album_id in item.get("album_id", [])
    ])
    groups = [
        [track_id for album_id in last.get("album_id", []) for track_id in album_tracks.get(str(album_id), [])],
        [track_id for album_id in all_albums for track_id in album_tracks.get(album_id, [])],
        [track_id for artist_id in last.get("artist_id", []) for track_id in artist_tracks.get(str(artist_id), [])],
        [track_id for artist_id in all_artists for track_id in artist_tracks.get(artist_id, [])],
    ]
    return filter_seen(unique([track_id for group in groups for track_id in group]), history, topk)


def rank_of(candidates: list[str], target: str) -> int | None:
    try:
        return candidates.index(target) + 1
    except ValueError:
        return None


def ndcg(rank: int | None, cutoff: int = 20) -> float:
    return 1.0 / math.log2(rank + 1) if rank is not None and rank <= cutoff else 0.0


def reciprocal_rank_fusion(channels: list[list[str]], topk: int, rrf_k: int) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    for candidates in channels:
        for rank, track_id in enumerate(candidates, start=1):
            scores[track_id] += 1.0 / (rrf_k + rank)
            best_rank[track_id] = min(rank, best_rank.get(track_id, rank))
    return [
        track_id
        for track_id, _ in sorted(
            scores.items(),
            key=lambda pair: (-pair[1], best_rank[pair[0]], pair[0]),
        )[:topk]
    ]


def metric_summary(records: list[dict[str, Any]], channel: str) -> dict[str, float | int]:
    ranks = [record["ranks"].get(channel) for record in records]
    return {
        "turns": len(records),
        "recall@1": sum(rank == 1 for rank in ranks) / max(1, len(ranks)),
        "recall@5": sum(rank is not None and rank <= 5 for rank in ranks) / max(1, len(ranks)),
        "recall@20": sum(rank is not None and rank <= 20 for rank in ranks) / max(1, len(ranks)),
        "recall@100": sum(rank is not None and rank <= 100 for rank in ranks) / max(1, len(ranks)),
        "recall@400": sum(rank is not None and rank <= 400 for rank in ranks) / max(1, len(ranks)),
        "ndcg@20": sum(ndcg(rank) for rank in ranks) / max(1, len(ranks)),
    }


def weighted_by_turn(
    records: list[dict[str, Any]],
    channel: str,
    weights: dict[int, int],
) -> dict[str, float]:
    per_turn = {
        turn: metric_summary([record for record in records if record["turn_number"] == turn], channel)
        for turn in sorted(weights)
    }
    denominator = max(1, sum(weights.values()))
    return {
        metric: sum(weights[turn] * float(per_turn[turn][metric]) for turn in weights) / denominator
        for metric in ["recall@1", "recall@5", "recall@20", "recall@100", "recall@400", "ndcg@20"]
    }


def union_summary(
    records: list[dict[str, Any]],
    channel_names: list[str],
) -> dict[str, float | int]:
    result: dict[str, float | int] = {"turns": len(records)}
    for cutoff in [1, 5, 20, 100, 400]:
        result[f"recall@{cutoff}"] = sum(
            any(
                record["ranks"].get(channel) is not None
                and record["ranks"][channel] <= cutoff
                for channel in channel_names
            )
            for record in records
        ) / max(1, len(records))
    return result


def load_blind_turn_weights(args) -> dict[int, int]:
    if args.blind_dataset_name:
        blind = load_dataset(
            args.blind_dataset_name,
            split=args.blind_split,
            cache_dir=args.cache_dir,
        )
        return dict(Counter(max(turn["turn_number"] for turn in item["conversations"]) for item in blind))
    return DEFAULT_BLIND_TURN_WEIGHTS


def main(args) -> None:
    dev = load_dataset(
        args.dev_dataset_name,
        split=args.dev_split,
        cache_dir=args.cache_dir,
    )
    if args.max_sessions:
        dev = dev.select(range(min(args.max_sessions, len(dev))))

    bm25 = BM25_MODEL(
        args.track_metadata_name,
        ["all_tracks"],
        args.corpus_types,
        args.cache_dir,
    )
    track_texts = {track_id: track_to_text(item) for track_id, item in bm25.metadata_dict.items()}
    artist_tracks, album_tracks = build_metadata_indexes(bm25.metadata_dict)

    tasks: list[dict[str, Any]] = []
    for item in dev:
        target_turns = sorted({
            turn["turn_number"]
            for turn in item["conversations"]
            if turn["role"] == "music"
        })
        if args.turn_mode == "final":
            target_turns = target_turns[-1:]
        for target_turn in target_turns:
            history = [
                turn["content"]
                for turn in item["conversations"]
                if turn["role"] == "music" and turn["turn_number"] < target_turn
            ]
            tasks.append({
                "session_id": item["session_id"],
                "turn_number": target_turn,
                "target": get_turn_target(item["conversations"], target_turn),
                "history": history,
                "legacy_query": build_query_text(
                    item,
                    target_turn,
                    track_texts,
                    query_mode="legacy",
                    history_turns=args.history_turns,
                ),
                "feedback_query": feedback_rich_query(
                    item,
                    target_turn,
                    track_texts,
                    args.history_turns,
                ),
            })

    legacy_candidates = bm25.batch_text_to_item_retrieval(
        [task["legacy_query"] for task in tasks],
        topk=args.bm25_topk + 8,
    )
    feedback_candidates = bm25.batch_text_to_item_retrieval(
        [task["feedback_query"] for task in tasks],
        topk=args.bm25_topk + 8,
    )
    channels: dict[str, list[list[str]]] = {
        "bm25_legacy": [
            filter_seen(candidates, task["history"], args.bm25_topk)
            for task, candidates in zip(tasks, legacy_candidates)
        ],
        "bm25_feedback": [
            filter_seen(candidates, task["history"], args.bm25_topk)
            for task, candidates in zip(tasks, feedback_candidates)
        ],
        "structure": [
            structural_candidates(
                task["history"],
                bm25.metadata_dict,
                artist_tracks,
                album_tracks,
                args.embedding_topk,
            )
            for task in tasks
        ],
    }

    for embedding_field in args.embedding_fields:
        index = PrecomputedTrackEmbeddingIndex(
            embedding_field=embedding_field,
            cache_dir=args.cache_dir,
            device=args.device,
        )
        for aggregation in args.embedding_aggregations:
            candidate_lists, _ = index.batch_history_retrieval(
                [task["history"] for task in tasks],
                topk=args.embedding_topk,
                aggregation=aggregation,
                batch_size=args.embedding_batch_size,
                exclude_seen=True,
            )
            channels[f"{embedding_field}:{aggregation}"] = candidate_lists
        index.unload()

    records = []
    channel_names = list(channels)
    for index, task in enumerate(tasks):
        task_channels = {name: channels[name][index] for name in channel_names}
        text_rrf = reciprocal_rank_fusion(
            [task_channels["bm25_legacy"], task_channels["bm25_feedback"]],
            args.final_topk,
            args.rrf_k,
        )
        history_names = [
            name
            for name in channel_names
            if name == "structure" or ":" in name
        ]
        history_rrf = reciprocal_rank_fusion(
            [task_channels[name] for name in history_names],
            args.final_topk,
            args.rrf_k,
        )
        all_rrf = reciprocal_rank_fusion(
            [task_channels[name] for name in channel_names],
            args.final_topk,
            args.rrf_k,
        )
        task_channels["rrf_text"] = text_rrf
        task_channels["rrf_history"] = history_rrf
        task_channels["rrf_all"] = all_rrf
        records.append({
            "session_id": task["session_id"],
            "turn_number": task["turn_number"],
            "target": task["target"],
            "history_size": len(task["history"]),
            "ranks": {
                name: rank_of(candidates, task["target"])
                for name, candidates in task_channels.items()
            },
        })

    evaluated_channels = channel_names + ["rrf_text", "rrf_history", "rrf_all"]
    blind_turn_weights = load_blind_turn_weights(args)
    final_turn_by_session: dict[str, int] = {}
    for record in records:
        final_turn_by_session[record["session_id"]] = max(
            record["turn_number"],
            final_turn_by_session.get(record["session_id"], 0),
        )
    final_turn_records = [
        record
        for record in records
        if record["turn_number"] == final_turn_by_session[record["session_id"]]
    ]
    report = {
        "settings": {
            "dev_dataset_name": args.dev_dataset_name,
            "dev_split": args.dev_split,
            "sessions": len(dev),
            "turn_mode": args.turn_mode,
            "tasks": len(tasks),
            "bm25_topk": args.bm25_topk,
            "embedding_topk": args.embedding_topk,
            "embedding_fields": args.embedding_fields,
            "embedding_aggregations": args.embedding_aggregations,
            "exclude_seen": True,
            "blind_turn_weights": blind_turn_weights,
        },
        "overall": {
            channel: metric_summary(records, channel)
            for channel in evaluated_channels
        },
        "turn1": {
            channel: metric_summary(
                [record for record in records if record["turn_number"] == 1],
                channel,
            )
            for channel in evaluated_channels
        },
        "turn2plus": {
            channel: metric_summary(
                [record for record in records if record["turn_number"] >= 2],
                channel,
            )
            for channel in evaluated_channels
        },
        "final_turn": {
            channel: metric_summary(final_turn_records, channel)
            for channel in evaluated_channels
        },
        "per_turn": {
            str(turn): {
                channel: metric_summary(
                    [record for record in records if record["turn_number"] == turn],
                    channel,
                )
                for channel in evaluated_channels
            }
            for turn in sorted({record["turn_number"] for record in records})
        },
        "candidate_union": {
            "overall": union_summary(records, channel_names),
            "turn1": union_summary(
                [record for record in records if record["turn_number"] == 1],
                channel_names,
            ),
            "turn2plus": union_summary(
                [record for record in records if record["turn_number"] >= 2],
                channel_names,
            ),
            "final_turn": union_summary(final_turn_records, channel_names),
            "per_turn": {
                str(turn): union_summary(
                    [record for record in records if record["turn_number"] == turn],
                    channel_names,
                )
                for turn in sorted({record["turn_number"] for record in records})
            },
        },
    }
    if args.turn_mode == "all":
        report["blind_turn_weighted"] = {
            channel: weighted_by_turn(records, channel, blind_turn_weights)
            for channel in evaluated_channels
        }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    console_summary = {
        "settings": report["settings"],
        "overall": report["overall"],
        "final_turn": report["final_turn"],
        "candidate_union": report["candidate_union"],
    }
    if "blind_turn_weighted" in report:
        console_summary["blind_turn_weighted"] = report["blind_turn_weighted"]
    print(json.dumps(console_summary, indent=2, ensure_ascii=False))
    print(f"Saved report to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate final-turn and prefix retrieval channels.")
    parser.add_argument("--dev_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--dev_split", default="test")
    parser.add_argument("--blind_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
    parser.add_argument("--blind_split", default="test")
    parser.add_argument("--track_metadata_name", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument("--output_path", default="exp/evaluation/dev_multichannel_all_turns.json")
    parser.add_argument(
        "--corpus_types",
        nargs="+",
        default=["track_name", "artist_name", "album_name", "release_date", "tag_list"],
    )
    parser.add_argument(
        "--embedding_fields",
        nargs="+",
        default=["image-siglip2", "metadata-qwen3_embedding_0.6b"],
    )
    parser.add_argument("--embedding_aggregations", nargs="+", default=["last", "mean"])
    parser.add_argument("--bm25_topk", type=int, default=400)
    parser.add_argument("--embedding_topk", type=int, default=400)
    parser.add_argument("--final_topk", type=int, default=400)
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--history_turns", type=int, default=0)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--turn_mode", choices=["all", "final"], default="all")
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--device", default=None)
    main(parser.parse_args())
