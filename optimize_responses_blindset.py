"""Optimize Blind A response text while keeping the track ranking fixed."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import requests
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from mcrs.db_item import MusicCatalogDB


FORBIDDEN_HINTS = (
    "metadata",
    "ranking",
    "ranked",
    "system",
    "tool",
    "mistake",
    "mismatch",
    "sorry",
    "apolog",
)


@dataclass
class ResponseContext:
    prediction: dict[str, Any]
    blind_item: dict[str, Any]
    track: dict[str, Any]
    current_response: str


class DoubaoClient:
    def __init__(self, args: argparse.Namespace) -> None:
        self.api_key = os.getenv("DOUBAO_API_KEY") or os.getenv("ARK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DOUBAO_API_KEY or ARK_API_KEY is not set.")
        self.base_url = os.getenv(
            "DOUBAO_BASE_URL",
            "https://ark.cn-beijing.volces.com/api/v3",
        ).rstrip("/")
        self.model = os.getenv("DOUBAO_MODEL", args.model)
        self.timeout = float(os.getenv("DOUBAO_TIMEOUT", str(args.timeout)))
        self.max_retries = int(os.getenv("DOUBAO_MAX_RETRIES", str(args.max_retries)))
        self.reasoning_effort = os.getenv("DOUBAO_REASONING_EFFORT", "minimal")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": self.reasoning_effort,
        }
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Doubao request failed after {self.max_retries} attempts: {last_error}")


def first_value(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def clean_response(response: str) -> str:
    response = response.replace("*", "").replace("`", "")
    response = re.sub(r"\s+", " ", response)
    return response.strip().strip('"')


def normalize_tags(tags: Any, limit: int = 12) -> list[str]:
    if not isinstance(tags, list):
        return []
    seen = set()
    cleaned = []
    for raw in tags:
        tag = str(raw).strip()
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        if len(tag) > 32 or any(char.isdigit() for char in tag):
            continue
        seen.add(key)
        cleaned.append(tag)
        if len(cleaned) >= limit:
            break
    return cleaned


def track_brief(track: dict[str, Any]) -> dict[str, Any]:
    release_date = first_value(track.get("release_date"))
    return {
        "track_id": first_value(track.get("track_id")),
        "track_name": first_value(track.get("track_name")),
        "artist_name": first_value(track.get("artist_name")),
        "album_name": first_value(track.get("album_name")),
        "release_date": release_date,
        "year": release_date[:4] if len(release_date) >= 4 else "",
        "tags": normalize_tags(track.get("tag_list")),
    }


def track_text(track: dict[str, Any]) -> str:
    brief = track_brief(track)
    parts = [
        f"title: {brief['track_name']}",
        f"artist: {brief['artist_name']}",
    ]
    if brief["album_name"]:
        parts.append(f"album: {brief['album_name']}")
    if brief["release_date"]:
        parts.append(f"release_date: {brief['release_date']}")
    if brief["tags"]:
        parts.append(f"tags: {', '.join(brief['tags'])}")
    return "; ".join(parts)


def render_conversation(item: dict[str, Any], item_db: MusicCatalogDB) -> str:
    lines = []
    for turn in item["conversations"]:
        role = turn.get("role", "user")
        content = str(turn.get("content", ""))
        if role == "music":
            track_id = content.strip()
            metadata = item_db.metadata_dict.get(track_id)
            if metadata:
                brief = track_brief(metadata)
                content = f"previous recommended track: {brief['track_name']} by {brief['artist_name']}"
            else:
                content = "previous recommended track"
            role = "assistant"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def user_goal_text(item: dict[str, Any]) -> str:
    goal = item.get("conversation_goal") or {}
    fields = [
        str(goal.get("listener_goal") or ""),
        str(goal.get("category") or ""),
        str(goal.get("specificity") or ""),
    ]
    return " | ".join(field for field in fields if field)


def has_basic_issues(response: str, track: dict[str, Any]) -> list[str]:
    lowered = response.lower()
    title = first_value(track.get("track_name")).lower()
    artist = first_value(track.get("artist_name")).lower()
    issues = []
    if title and title not in lowered:
        issues.append("missing_title")
    if artist and artist not in lowered:
        issues.append("missing_artist")
    words = response.split()
    if len(words) < 40:
        issues.append("too_short")
    if len(words) > 95:
        issues.append("too_long")
    if not response.endswith("?"):
        issues.append("no_followup_question")
    for hint in FORBIDDEN_HINTS:
        if hint in lowered:
            issues.append(f"forbidden_{hint}")
            break
    if "\n" in response:
        issues.append("multi_line")
    return issues


def generation_prompt(ctx: ResponseContext, item_db: MusicCatalogDB, variant: int) -> list[dict[str, str]]:
    item = ctx.blind_item
    track = track_brief(ctx.track)
    goal = user_goal_text(item)
    transcript = render_conversation(item, item_db)
    style_notes = [
        "Use a direct, grounded answer first if the latest user asks a factual question.",
        "Prefer concrete musical fit words from the metadata tags and user wording.",
        "Do not invent album years, lyrical meanings, critic praise, chart facts, or soundtrack facts.",
        "Mention the recommended title and artist exactly once.",
        "If a user asks about a different song, briefly answer only what the conversation already establishes, then bridge to the recommended track.",
        "Write one natural English paragraph, 55-85 words, with no markdown or list formatting.",
        "End with a short follow-up question.",
    ]
    variant_notes = [
        "Keep the tone warm and confident, with clear continuity from the user's latest wording.",
        "Make the first sentence especially precise and avoid hype words unless the user asked for high energy.",
        "Sound like a human music assistant: specific, concise, and not salesy.",
    ]
    content = {
        "conversation_goal": goal,
        "conversation": transcript,
        "recommended_top1_track": track,
        "current_response_to_improve_from": ctx.current_response,
        "rules": style_notes,
        "variant_instruction": variant_notes[variant % len(variant_notes)],
    }
    return [
        {
            "role": "system",
            "content": (
                "You write final responses for a music conversational recommender. "
                "You must stay grounded in the supplied conversation and track metadata."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(content, ensure_ascii=False, indent=2),
        },
    ]


def selector_prompt(
    ctx: ResponseContext,
    item_db: MusicCatalogDB,
    candidates: list[dict[str, str]],
) -> list[dict[str, str]]:
    item = ctx.blind_item
    content = {
        "task": (
            "Choose the best final assistant response for the music recommendation. "
            "Reward relevance to the latest user request, grounding in the recommended track metadata, "
            "specific musical explanation, naturalness, and a good follow-up question. "
            "Penalize unsupported factual claims, wrong album/year/lyrics, generic praise, apology/tool/system wording, "
            "and recommending a different track."
        ),
        "conversation_goal": user_goal_text(item),
        "conversation": render_conversation(item, item_db),
        "recommended_top1_track": track_brief(ctx.track),
        "candidates": candidates,
        "output_format": {
            "best_id": "candidate id string",
            "scores": {"candidate id": "number from 1 to 5"},
            "unsupported_claims": {"candidate id": ["short issue strings"]},
            "reason": "short reason",
        },
    }
    return [
        {
            "role": "system",
            "content": "You are a strict evaluator. Return only valid JSON.",
        },
        {
            "role": "user",
            "content": json.dumps(content, ensure_ascii=False, indent=2),
        },
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object found in selector output: {text[:200]}")
    return json.loads(match.group(0))


def candidate_score(result: dict[str, Any], candidate_id: str) -> float:
    scores = result.get("scores") or {}
    value = scores.get(candidate_id, 0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def optimize_one(
    ctx: ResponseContext,
    item_db: MusicCatalogDB,
    client: DoubaoClient,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current = clean_response(ctx.current_response)
    candidates = [{"id": "current", "response": current}]
    generated = []
    for index in range(args.num_candidates):
        messages = generation_prompt(ctx, item_db, index)
        response = client.chat(
            messages,
            temperature=args.generation_temperature,
            max_tokens=args.generation_max_tokens,
        )
        response = clean_response(response)
        if response and response not in {item["response"] for item in candidates}:
            generated.append(response)
            candidates.append({"id": f"gen_{index + 1}", "response": response})

    selector_raw = client.chat(
        selector_prompt(ctx, item_db, candidates),
        temperature=0.0,
        max_tokens=args.selector_max_tokens,
    )
    selector = extract_json_object(selector_raw)
    best_id = str(selector.get("best_id", "current"))
    if best_id not in {item["id"] for item in candidates}:
        best_id = "current"

    current_score = candidate_score(selector, "current")
    best_score = candidate_score(selector, best_id)
    issues = has_basic_issues(current, ctx.track)
    replace = best_id != "current" and (
        best_score >= current_score + args.min_delta or bool(issues)
    )
    chosen_id = best_id if replace else "current"
    chosen = next(item["response"] for item in candidates if item["id"] == chosen_id)

    updated = dict(ctx.prediction)
    updated["predicted_response"] = clean_response(chosen)
    report = {
        "session_id": ctx.prediction["session_id"],
        "top1": ctx.prediction["predicted_track_ids"][0],
        "track": track_brief(ctx.track),
        "current_issues": issues,
        "candidates": candidates,
        "selector": selector,
        "chosen_id": chosen_id,
        "changed": chosen_id != "current",
    }
    return updated, report


def load_contexts(args: argparse.Namespace) -> tuple[list[ResponseContext], MusicCatalogDB]:
    config = OmegaConf.load(f"config/{args.tid}.yaml")
    predictions = json.load(open(args.input_path, encoding="utf-8"))
    blind = load_dataset(config.test_dataset_name, split="test", cache_dir=config.cache_dir)
    blind_by_session = {item["session_id"]: item for item in blind}
    item_db = MusicCatalogDB(config.item_db_name, config.track_split_types, config.corpus_types)
    contexts = []
    for pred in predictions:
        track_id = pred["predicted_track_ids"][0]
        contexts.append(ResponseContext(
            prediction=pred,
            blind_item=blind_by_session[pred["session_id"]],
            track=item_db.metadata_dict[track_id],
            current_response=str(pred.get("predicted_response") or ""),
        ))
    return contexts, item_db


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)


def main(args: argparse.Namespace) -> None:
    all_contexts, item_db = load_contexts(args)
    contexts = all_contexts
    if args.max_items:
        contexts = contexts[:args.max_items]
    client = DoubaoClient(args)

    by_session = {ctx.prediction["session_id"]: dict(ctx.prediction) for ctx in all_contexts}
    reports = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_ctx = {
            executor.submit(optimize_one, ctx, item_db, client, args): ctx
            for ctx in contexts
        }
        for future in tqdm(as_completed(future_to_ctx), total=len(future_to_ctx), desc="Optimizing responses"):
            ctx = future_to_ctx[future]
            updated, report = future.result()
            by_session[ctx.prediction["session_id"]] = updated
            reports.append(report)

    output = [by_session[ctx.prediction["session_id"]] for ctx in all_contexts]
    write_json(args.output_path, output)
    if args.report_path:
        write_json(args.report_path, sorted(reports, key=lambda item: item["session_id"]))

    changed = sum(1 for item in reports if item["changed"])
    print(json.dumps({
        "output_path": args.output_path,
        "report_path": args.report_path,
        "sessions": len(output),
        "changed_responses": changed,
        "unchanged_responses": len(output) - changed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Blind A responses with fixed rankings.")
    parser.add_argument("--tid", default="bm25_tags_doubao_blindset_A")
    parser.add_argument(
        "--input_path",
        default="exp/inference/blindset_A/multichannel_ltr_lean_v2_prediction.json",
    )
    parser.add_argument(
        "--output_path",
        default="exp/inference/blindset_A/multichannel_ltr_lean_v2_response_opt_prediction.json",
    )
    parser.add_argument(
        "--report_path",
        default="exp/inference/blindset_A/multichannel_ltr_lean_v2_response_opt_report.json",
    )
    parser.add_argument("--model", default="doubao-seed-2-0-pro-260215")
    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--generation_temperature", type=float, default=0.55)
    parser.add_argument("--generation_max_tokens", type=int, default=180)
    parser.add_argument("--selector_max_tokens", type=int, default=600)
    parser.add_argument("--min_delta", type=float, default=0.15)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_items", type=int, default=0)
    main(parser.parse_args())
