"""Merge a candidate ranking with champion top tracks and responses locked."""

from __future__ import annotations

import argparse
import json
import os


def locked_prefix_merge(
    base_item: dict,
    candidate_item: dict,
    topk: int,
    lock_prefix: int,
) -> dict:
    merged = list(base_item["predicted_track_ids"][:lock_prefix])
    for track_id in candidate_item["predicted_track_ids"]:
        if track_id not in merged:
            merged.append(track_id)
        if len(merged) == topk:
            break
    for track_id in base_item["predicted_track_ids"]:
        if track_id not in merged:
            merged.append(track_id)
        if len(merged) == topk:
            break
    if len(merged) != topk:
        raise RuntimeError(f"Only produced {len(merged)} tracks for {base_item['session_id']}.")

    output = dict(candidate_item)
    output["predicted_track_ids"] = merged
    output["predicted_response"] = base_item.get("predicted_response", "")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        default="exp/inference/blindset_A/multichannel_ltr_lean_v2_prediction.json",
    )
    parser.add_argument(
        "--candidate_path",
        default="exp/inference/blindset_A/multichannel_ltr_turn1_s13_later_v2_empty.json",
    )
    parser.add_argument(
        "--output_path",
        default="exp/inference/blindset_A/multichannel_ltr_turn1_s13_later_v2_top1lock_prediction.json",
    )
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--lock_prefix", type=int, default=1)
    parser.add_argument(
        "--candidate_turn_min",
        type=int,
        default=1,
        help="Keep base rows unchanged before this turn number.",
    )
    args = parser.parse_args()

    base = json.load(open(args.base_path, encoding="utf-8"))
    candidate = json.load(open(args.candidate_path, encoding="utf-8"))
    base_by_session = {item["session_id"]: item for item in base}

    output = []
    changed_lists = 0
    changed_top1 = 0
    changed_prefix = 0
    changed_by_turn: dict[int, int] = {}
    for candidate_item in candidate:
        base_item = base_by_session[candidate_item["session_id"]]
        turn_number = int(base_item["turn_number"])
        if turn_number < args.candidate_turn_min:
            merged = dict(base_item)
        else:
            merged = locked_prefix_merge(
                base_item,
                candidate_item,
                args.topk,
                args.lock_prefix,
            )
        changed_lists += merged["predicted_track_ids"] != base_item["predicted_track_ids"]
        changed_top1 += merged["predicted_track_ids"][0] != base_item["predicted_track_ids"][0]
        changed_prefix += (
            merged["predicted_track_ids"][:args.lock_prefix]
            != base_item["predicted_track_ids"][:args.lock_prefix]
        )
        if merged["predicted_track_ids"] != base_item["predicted_track_ids"]:
            changed_by_turn[turn_number] = changed_by_turn.get(turn_number, 0) + 1
        output.append(merged)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False)
    print(json.dumps({
        "output_path": args.output_path,
        "rows": len(output),
        "changed_lists": changed_lists,
        "changed_top1": changed_top1,
        "changed_prefix": changed_prefix,
        "changed_by_turn": changed_by_turn,
        "lock_prefix": args.lock_prefix,
        "candidate_turn_min": args.candidate_turn_min,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
