"""Run Blind A inference with BM25 candidate recall and a trained reranker."""

import argparse
import json
import os
from typing import Any

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mcrs import load_crs_baseline
from mcrs.retrieval_modules.bm25 import BM25_MODEL
from train_reranker import build_query_text, track_to_text


def score_candidates(model, tokenizer, device, query: str, candidate_texts: list[str], batch_size: int, max_length: int) -> list[float]:
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
            logits = model(**encoded).logits.squeeze(-1)
            scores.extend(logits.detach().cpu().tolist())
    return scores


def main(args) -> None:
    config = OmegaConf.load(f"config/{args.tid}.yaml")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    retriever = BM25_MODEL(config.item_db_name, config.track_split_types, config.corpus_types, config.cache_dir)
    track_texts = {track_id: track_to_text(metadata) for track_id, metadata in retriever.metadata_dict.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.reranker_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.reranker_dir)
    model.to(device)

    music_crs = None
    if args.response_mode == "config":
        music_crs = load_crs_baseline(
            lm_type=config.lm_type,
            retrieval_type="bm25",
            item_db_name=config.item_db_name,
            user_db_name=config.user_db_name,
            track_split_types=config.track_split_types,
            user_split_types=config.user_split_types,
            corpus_types=config.corpus_types,
            cache_dir=config.cache_dir,
            device=config.device,
            attn_implementation=config.attn_implementation,
            dtype=torch.bfloat16,
        )

    db = load_dataset(config.test_dataset_name, split="test")
    results = []
    for item in tqdm(db, desc="Rerank Blind A"):
        retrieval_query = build_query_text(
            item,
            item["conversations"][-1]["turn_number"],
            track_texts,
            query_mode=args.retrieval_query_mode,
            history_turns=args.history_turns,
        )
        reranker_query = build_query_text(
            item,
            item["conversations"][-1]["turn_number"],
            track_texts,
            query_mode=args.reranker_query_mode,
            history_turns=args.history_turns,
        )
        candidates = retriever.text_to_item_retrieval(retrieval_query, topk=args.candidate_topk)
        candidate_texts = [track_texts[track_id] for track_id in candidates]
        scores = score_candidates(model, tokenizer, device, reranker_query, candidate_texts, args.rerank_batch_size, args.max_length)
        ranked = [track_id for track_id, _ in sorted(zip(candidates, scores), key=lambda item_score: item_score[1], reverse=True)]
        predicted_track_ids = ranked[:20]

        session_memory = item["conversations"][:-1]
        user_query = item["conversations"][-1]["content"]
        turn_number = item["conversations"][-1]["turn_number"]
        if args.response_mode == "empty":
            response = ""
        else:
            recommend_item = music_crs.item_db.id_to_metadata(predicted_track_ids[0])
            session_for_lm = [dict(turn) for turn in session_memory]
            session_for_lm.append({"role": "user", "content": user_query})
            response = music_crs.lm.response_generation(music_crs._get_system_prompt(item["user_id"]), session_for_lm, recommend_item)

        results.append({
            "session_id": item["session_id"],
            "user_id": item["user_id"],
            "turn_number": turn_number,
            "predicted_track_ids": predicted_track_ids,
            "predicted_response": response,
        })

    os.makedirs(f"exp/inference/{args.eval_dataset}", exist_ok=True)
    output_path = f"exp/inference/{args.eval_dataset}/{args.output_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Blind A inference with a trained reranker.")
    parser.add_argument("--tid", default="bm25_doubao_blindset_A")
    parser.add_argument("--eval_dataset", default="blindset_A")
    parser.add_argument("--reranker_dir", default="./exp/reranker/minilm_bm25")
    parser.add_argument("--output_name", default="bm25_rerank_doubao_blindset_A")
    parser.add_argument("--candidate_topk", type=int, default=200)
    parser.add_argument("--rerank_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--device", default=None)
    parser.add_argument("--retrieval_query_mode", choices=["focused", "legacy"], default="legacy")
    parser.add_argument("--reranker_query_mode", choices=["focused", "legacy"], default="focused")
    parser.add_argument("--history_turns", type=int, default=3)
    parser.add_argument("--response_mode", choices=["config", "empty"], default="empty")
    main(parser.parse_args())
