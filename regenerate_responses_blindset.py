"""Regenerate predicted_response for an existing Blind A prediction file."""

import argparse
import json
import os

from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from mcrs.db_item import MusicCatalogDB
from mcrs.db_user import UserProfileDB
from mcrs.lm_modules import load_lm_module


def clean_response(response: str) -> str:
    return " ".join(response.replace("*", "").replace("`", "").split())


def build_system_prompt(user_db: UserProfileDB, user_id: str) -> str:
    prompts_dir = os.path.join(os.path.dirname(__file__), "mcrs", "system_prompts")
    role_play = open(os.path.join(prompts_dir, "roleplay.txt"), encoding="utf-8").read()
    response_generation = open(os.path.join(prompts_dir, "response_generation.txt"), encoding="utf-8").read()
    personalization = open(os.path.join(prompts_dir, "personalization.txt"), encoding="utf-8").read()
    return role_play + response_generation + personalization + "\n" + user_db.id_to_profile_str(user_id)


def main(args) -> None:
    config = OmegaConf.load(f"config/{args.tid}.yaml")
    predictions = json.load(open(args.input_path, encoding="utf-8"))
    reusable = {}
    if args.reuse_responses_from:
        previous = json.load(open(args.reuse_responses_from, encoding="utf-8"))
        reusable = {item["session_id"]: item for item in previous}
    blind_db = load_dataset(config.test_dataset_name, split="test")
    blind_by_session = {item["session_id"]: item for item in blind_db}

    item_db = MusicCatalogDB(config.item_db_name, config.track_split_types, config.corpus_types)
    user_db = UserProfileDB(config.user_db_name, config.user_split_types)
    lm = load_lm_module(config.lm_type, config.device, config.attn_implementation, None)

    response_jobs = []
    for pred in tqdm(predictions, desc="Preparing response jobs"):
        previous = reusable.get(pred["session_id"])
        if (
            previous
            and previous.get("predicted_track_ids")
            and previous["predicted_track_ids"][0] == pred["predicted_track_ids"][0]
            and str(previous.get("predicted_response") or "").strip()
        ):
            pred["predicted_response"] = clean_response(previous["predicted_response"])
            continue
        item = blind_by_session[pred["session_id"]]
        conversations = item["conversations"]
        session_memory = conversations[:-1]
        user_query = conversations[-1]["content"]
        session_for_lm = [dict(turn) for turn in session_memory]
        session_for_lm.append({"role": "user", "content": user_query})

        recommend_item = item_db.id_to_metadata(pred["predicted_track_ids"][0])
        response_jobs.append((
            pred,
            build_system_prompt(user_db, pred["user_id"]),
            session_for_lm,
            recommend_item,
        ))

    if hasattr(lm, "batch_response_generation"):
        responses = lm.batch_response_generation(
            [job[1] for job in response_jobs],
            [job[2] for job in response_jobs],
            [job[3] for job in response_jobs],
        )
    else:
        responses = [
            lm.response_generation(sys_prompt, session_for_lm, recommend_item)
            for _, sys_prompt, session_for_lm, recommend_item in tqdm(response_jobs, desc="Regenerating responses")
        ]

    for (pred, _, _, _), response in zip(response_jobs, responses):
        pred["predicted_response"] = clean_response(response)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)
    print(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate Blind A response text without changing track rankings.")
    parser.add_argument("--tid", default="bm25_doubao_blindset_A")
    parser.add_argument("--input_path", default="exp/inference/blindset_A/bm25_rerank_doubao_blindset_A.json")
    parser.add_argument("--output_path", default="exp/inference/blindset_A/bm25_rerank_doubao_response_v2.json")
    parser.add_argument("--reuse_responses_from", default=None)
    main(parser.parse_args())
