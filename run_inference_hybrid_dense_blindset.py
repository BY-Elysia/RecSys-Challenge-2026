"""Run Blind A inference with BM25 plus Qwen dense candidate recall and a reranker."""

import argparse
import json
import os

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mcrs.retrieval_modules.bm25 import BM25_MODEL
from mcrs.retrieval_modules.qwen_dense import DEFAULT_TASK_INSTRUCTION, QwenDenseRetriever
from run_inference_rerank_blindset import score_candidates
from train_reranker import build_query_text, track_to_text


def union_candidates(primary: list[str], supplement: list[str]) -> list[str]:
    seen = set(primary)
    return primary + [track_id for track_id in supplement if track_id not in seen]


def main(args) -> None:
    config = OmegaConf.load(f"config/{args.tid}.yaml")
    bm25 = BM25_MODEL(config.item_db_name, config.track_split_types, config.corpus_types, config.cache_dir)
    track_texts = {track_id: track_to_text(metadata) for track_id, metadata in bm25.metadata_dict.items()}
    db = load_dataset(config.test_dataset_name, split="test", cache_dir=config.cache_dir)

    tasks = []
    for item in db:
        target_turn = item["conversations"][-1]["turn_number"]
        tasks.append({
            "item": item,
            "bm25_query": build_query_text(
                item, target_turn, track_texts, args.bm25_query_mode, args.history_turns
            ),
            "dense_query": build_query_text(
                item, target_turn, track_texts, args.dense_query_mode, args.history_turns
            ),
            "reranker_query": build_query_text(
                item, target_turn, track_texts, args.reranker_query_mode, args.history_turns
            ),
        })

    bm25_lists = bm25.batch_text_to_item_retrieval(
        [task["bm25_query"] for task in tasks],
        topk=args.bm25_topk,
    )
    dense = QwenDenseRetriever(
        dataset_name=args.embedding_dataset_name,
        split=args.embedding_split,
        embedding_field=args.embedding_field,
        model_name=args.embedding_model_name,
        cache_dir=config.cache_dir,
        device=args.device,
        max_length=args.dense_max_length,
        query_batch_size=args.dense_batch_size,
        task_instruction=args.task_instruction,
    )
    dense_lists = dense.batch_text_to_item_retrieval(
        [task["dense_query"] for task in tasks],
        topk=args.dense_topk,
    )
    dense.unload()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.reranker_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.reranker_dir)
    model.to(device)

    results = []
    for task, bm25_candidates, dense_candidates in tqdm(
        zip(tasks, bm25_lists, dense_lists),
        total=len(tasks),
        desc="Reranking hybrid Blind A candidates",
    ):
        candidates = union_candidates(bm25_candidates, dense_candidates)
        scores = score_candidates(
            model,
            tokenizer,
            device,
            task["reranker_query"],
            [track_texts[track_id] for track_id in candidates],
            args.rerank_batch_size,
            args.max_length,
        )
        bm25_set = set(bm25_candidates)
        adjusted_scores = [
            score - args.dense_penalty if track_id not in bm25_set else score
            for track_id, score in zip(candidates, scores)
        ]
        ranked = [
            track_id
            for track_id, _ in sorted(
                zip(candidates, adjusted_scores),
                key=lambda pair: pair[1],
                reverse=True,
            )
        ]
        item = task["item"]
        results.append({
            "session_id": item["session_id"],
            "user_id": item["user_id"],
            "turn_number": item["conversations"][-1]["turn_number"],
            "predicted_track_ids": ranked[:20],
            "predicted_response": "",
        })

    os.makedirs(f"exp/inference/{args.eval_dataset}", exist_ok=True)
    output_path = f"exp/inference/{args.eval_dataset}/{args.output_name}.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False)
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hybrid dense Blind A inference.")
    parser.add_argument("--tid", default="bm25_tags_doubao_blindset_A")
    parser.add_argument("--eval_dataset", default="blindset_A")
    parser.add_argument("--reranker_dir", default="./exp/reranker/minilm_bm25_tags_top400_e1")
    parser.add_argument("--output_name", default="tags_top400_dense200_empty_blindset_A")
    parser.add_argument("--bm25_topk", type=int, default=400)
    parser.add_argument("--dense_topk", type=int, default=200)
    parser.add_argument("--bm25_query_mode", choices=["focused", "legacy"], default="legacy")
    parser.add_argument("--dense_query_mode", choices=["focused", "legacy"], default="focused")
    parser.add_argument("--reranker_query_mode", choices=["focused", "legacy"], default="focused")
    parser.add_argument("--history_turns", type=int, default=3)
    parser.add_argument("--embedding_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    parser.add_argument("--embedding_split", default="all_tracks")
    parser.add_argument("--embedding_field", default="metadata-qwen3_embedding_0.6b")
    parser.add_argument("--embedding_model_name", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--task_instruction", default=DEFAULT_TASK_INSTRUCTION)
    parser.add_argument("--dense_max_length", type=int, default=512)
    parser.add_argument("--dense_batch_size", type=int, default=16)
    parser.add_argument("--rerank_batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--dense_penalty", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    main(parser.parse_args())
