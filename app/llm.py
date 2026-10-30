import json
from typing import AsyncIterator

import httpx

from .config import settings
from .retrieval import Hit

SYSTEM = (
    "You answer questions using only the numbered context passages. "
    "Cite passages like [1] or [2]. If the context doesn't contain the answer, say so."
)


class LLMError(Exception):
    """The chat endpoint answered with an error status; the message carries what it said."""


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    ctx = "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, start=1))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"},
    ]


async def stream_completion(messages: list[dict]) -> AsyncIterator[str]:
    """Yields text deltas from any OpenAI-compatible /chat/completions endpoint."""
    if not settings.llm_api_key:
        yield "(LLM_API_KEY is not set, returning retrieval results only)"
        return

    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    body = {"model": settings.llm_model, "messages": messages, "stream": True, "temperature": 0.1}
    if settings.llm_reasoning_effort:
        body["reasoning_effort"] = settings.llm_reasoning_effort

    timeout = httpx.Timeout(settings.llm_timeout_seconds, connect=10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", f"{settings.llm_base_url.rstrip('/')}/chat/completions", headers=headers, json=body
        ) as resp:
            if resp.status_code >= 400:
                detail = (await resp.aread()).decode(errors="replace")[:300]
                raise LLMError(f"LLM request failed ({resp.status_code}): {detail}")
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                # some compatible servers send keep-alive or usage chunks with no choices
                choices = json.loads(data).get("choices") or []
                delta = (choices[0].get("delta") or {}).get("content") if choices else None
                if delta:
                    yield delta
