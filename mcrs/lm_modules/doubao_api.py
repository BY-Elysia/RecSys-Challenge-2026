import os
import time
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import requests


class DOUBAO_API_MODEL:
    """OpenAI-compatible Doubao response generator."""

    def __init__(self) -> None:
        self.api_key = os.getenv("DOUBAO_API_KEY") or os.getenv("ARK_API_KEY")
        self.base_url = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
        self.model = os.getenv("DOUBAO_MODEL", "doubao-seed-2-0-pro-260215")
        self.timeout = float(os.getenv("DOUBAO_TIMEOUT", "120"))
        self.max_retries = int(os.getenv("DOUBAO_MAX_RETRIES", "3"))
        self.temperature = float(os.getenv("DOUBAO_TEMPERATURE", "0.3"))
        self.max_tokens = int(os.getenv("DOUBAO_MAX_TOKENS", "160"))
        self.reasoning_effort = os.getenv("DOUBAO_REASONING_EFFORT", "minimal")
        self.concurrency = int(os.getenv("DOUBAO_CONCURRENCY", "4"))

    def response_generation(self, sys_prompt: str, chat_history: list, recommend_item: str, max_new_tokens=512, response_format=None):
        return self._chat(sys_prompt, chat_history, recommend_item)

    def batch_response_generation(self, sys_prompts: list[str], chat_histories: list[list], recommend_items: list[str], max_new_tokens=64):
        jobs = list(zip(sys_prompts, chat_histories, recommend_items))
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            return list(executor.map(lambda job: self._chat(*job), jobs))

    def _chat(self, sys_prompt: str, chat_history: list[dict[str, Any]], recommend_item: str) -> str:
        if not self.api_key:
            raise RuntimeError("DOUBAO_API_KEY or ARK_API_KEY is not set.")

        messages = [{"role": "system", "content": sys_prompt}]
        messages.extend(self._clean_history(chat_history))
        messages.append({
            "role": "assistant",
            "content": f"Recommended track metadata: {recommend_item}",
        })
        messages.append({
            "role": "user",
            "content": (
                "Write the final assistant reply for this music recommendation result.\n"
                "Use the user's latest request, the conversation context, and the recommended track metadata.\n"
                "Write one natural English paragraph, 45-90 words.\n"
                "If the user asks a factual music question, answer it directly before moving into the recommendation.\n"
                "Mention the recommended track title and artist exactly once when available.\n"
                "Explain why this specific track fits the user's taste, mood, genre, era, activity, or stated constraint.\n"
                "Sound confident and listener-focused; do not say the pick is outside the request, only adjacent, or not an exact match.\n"
                "Do not use bullets, lists, markdown, quotes, apology language, or phrases about systems, tools, rankings, metadata, or mistakes.\n"
                "Vary the wording across examples; avoid stock openings like 'A strong pick' or 'I recommend'.\n"
                "End with a short, natural follow-up question."
            ),
        })

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(2 ** attempt)

        raise RuntimeError(f"Doubao API request failed after {self.max_retries} attempts: {last_error}")

    def _clean_history(self, chat_history: list[dict[str, Any]]) -> list[dict[str, str]]:
        cleaned = []
        for message in chat_history:
            role = message.get("role", "user")
            if role not in {"user", "assistant", "system"}:
                role = "assistant"
            cleaned.append({
                "role": role,
                "content": str(message.get("content", "")),
            })
        return cleaned
