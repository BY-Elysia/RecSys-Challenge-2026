"""Analyze legal catalog coverage and the risk of global tail diversification."""

import argparse
import json
import math
from collections import Counter, defaultdict


def main(args) -> None:
    with open(args.prediction_path, encoding="utf-8") as file:
        predictions = json.load(file)

    lengths = [len(item["predicted_track_ids"]) for item in predictions]
    if any(length > args.max_tracks_per_prediction for length in lengths):
        raise ValueError(
            "Submission violates the official maximum of "
            f"{args.max_tracks_per_prediction} tracks per prediction."
        )

    tracks = [
        track_id
        for item in predictions
        for track_id in item["predicted_track_ids"]
    ]
    counts = Counter(tracks)
    repeated_by_rank: dict[int, int] = defaultdict(int)
    seen = set()
    for item in predictions:
        for rank, track_id in enumerate(item["predicted_track_ids"], start=1):
            if track_id in seen:
                repeated_by_rank[rank] += 1
            else:
                seen.add(track_id)

    legal_slots = len(predictions) * args.max_tracks_per_prediction
    unique_tracks = len(counts)
    legal_max_diversity = min(legal_slots, args.catalog_size) / args.catalog_size
    current_diversity = unique_tracks / args.catalog_size
    max_composite_gain = (
        args.catalog_weight * (legal_max_diversity - current_diversity)
    )
    one_rank20_hit_loss = (
        args.ndcg_weight
        * (1.0 / math.log2(args.max_tracks_per_prediction + 1))
        / len(predictions)
    )

    report = {
        "prediction_path": args.prediction_path,
        "predictions": len(predictions),
        "catalog_size": args.catalog_size,
        "recommendation_slots": len(tracks),
        "legal_recommendation_slots": legal_slots,
        "unique_tracks": unique_tracks,
        "repeated_slots": len(tracks) - unique_tracks,
        "current_catalog_diversity": current_diversity,
        "legal_max_catalog_diversity": legal_max_diversity,
        "max_catalog_diversity_gain": legal_max_diversity - current_diversity,
        "max_composite_gain_from_catalog_only": max_composite_gain,
        "composite_loss_if_one_rank20_hit_is_removed": one_rank20_hit_loss,
        "risk_to_max_gain_ratio": (
            one_rank20_hit_loss / max_composite_gain
            if max_composite_gain > 0
            else None
        ),
        "duplicate_occurrences_by_rank": dict(sorted(repeated_by_rank.items())),
        "tracks_recommended_more_than_once": sum(value > 1 for value in counts.values()),
        "maximum_track_frequency": max(counts.values(), default=0),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_path")
    parser.add_argument("--catalog_size", type=int, default=47071)
    parser.add_argument("--max_tracks_per_prediction", type=int, default=20)
    parser.add_argument("--catalog_weight", type=float, default=0.10)
    parser.add_argument("--ndcg_weight", type=float, default=0.50)
    main(parser.parse_args())
