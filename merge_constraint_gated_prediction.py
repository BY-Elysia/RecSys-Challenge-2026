"""Merge constraint-model rankings only for sessions with approved strict constraints."""

from __future__ import annotations

import argparse
import json
import os

from datasets import load_dataset
from omegaconf import OmegaConf

from constraint_gating import CONSTRAINT_CATEGORIES, locked_prefix_merge, strict_constraint_categories


def main(args: argparse.Namespace) -> None:
    config = OmegaConf.load(f"config/{args.tid}.yaml")
    blind = load_dataset(config.test_dataset_name, split="test", cache_dir=config.cache_dir)
    requests = {
        item["session_id"]: str(item["conversations"][-1]["content"])
        for item in blind
    }
    approved = set(args.approved_categories)
    base = json.load(open(args.base_path, encoding="utf-8"))
    candidate = json.load(open(args.candidate_path, encoding="utf-8"))
    candidate_by_session = {item["session_id"]: item for item in candidate}

    output = []
    selected = []
    changed = []
    for base_item in base:
        session_id = base_item["session_id"]
        categories = strict_constraint_categories(requests[session_id])
        active = sorted(categories & approved)
        result = dict(base_item)
        if active:
            selected.append({
                "session_id": session_id,
                "categories": active,
                "request": requests[session_id],
            })
            result["predicted_track_ids"] = locked_prefix_merge(
                base_item["predicted_track_ids"],
                candidate_by_session[session_id]["predicted_track_ids"],
                args.lock_prefix,
                args.topk,
            )
        if result["predicted_track_ids"] != base_item["predicted_track_ids"]:
            changed.append(session_id)
        result["predicted_response"] = base_item["predicted_response"]
        output.append(result)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False)
    print(json.dumps({
        "output_path": args.output_path,
        "rows": len(output),
        "approved_categories": sorted(approved),
        "selected_sessions": len(selected),
        "changed_lists": len(changed),
        "changed_session_ids": changed,
        "selection": selected,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tid", default="bm25_tags_doubao_blindset_A")
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--candidate_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--approved_categories",
        nargs="+",
        choices=CONSTRAINT_CATEGORIES,
        required=True,
    )
    parser.add_argument("--lock_prefix", type=int, default=1)
    parser.add_argument("--topk", type=int, default=20)
    main(parser.parse_args())
