"""External llama-server HTTP backend for exam APIs.

Calls a llama-server instance running natively on the Mac host (Metal-accelerated)
via HTTP. Used ONLY for exam question generation and writing evaluation — dialog/word explainer
use the in-cluster llamacpp backend.

Set via environment variable:
  EXTERNAL_LLM_URL → base URL of the llama-server (e.g. http://192.168.2.1:9090)
"""
from __future__ import annotations

import json
import logging
from typing import Iterable

import httpx

import os

try:
    from ..config import EXTERNAL_LLM_URL
except (ImportError, AttributeError):
    EXTERNAL_LLM_URL = os.getenv("EXTERNAL_LLM_URL", "http://192.168.2.1:9090")

logger = logging.getLogger("lang_learn.backends.external_llama")

# Persistent HTTP client (connection pooling)
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            base_url=EXTERNAL_LLM_URL,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=10.0),
        )
    return _client


def health_check() -> bool:
    """Check if the external llama-server is reachable."""
    try:
        resp = _get_client().get("/health")
        data = resp.json()
        return data.get("status") == "ok"
    except Exception as e:
        logger.warning("External LLM health check failed: %s", e)
        return False


def generate_external(
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """Generate text using the external llama-server via OpenAI-compatible chat completion.

    Applies the model's native chat template and returns the generated content.
    """
    client = _get_client()

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "stream": False,
        "stop": ["</s>", "<|endoftext|>", "<|im_end|>", "<end_of_turn>"],
    }

    logger.info("Calling external LLM at %s/v1/chat/completions (max_tokens=%d)", EXTERNAL_LLM_URL, max_tokens)
    response = client.post("/v1/chat/completions", json=payload)
    response.raise_for_status()

    data = response.json()
    msg = data.get("choices", [{}])[0].get("message", {})
    content = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    tokens_used = data.get("usage", {}).get("completion_tokens", len(content.split()))
    logger.info("External LLM response: %d chars, ~%d tokens", len(content), tokens_used)
    return content


def generate_external_stream(
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> Iterable[str]:
    """Stream tokens from the external llama-server via SSE."""
    client = _get_client()

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "stream": True,
        "stop": ["</s>", "<|endoftext|>", "<|im_end|>", "<end_of_turn>"],
    }

    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if not data_str or data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content") or delta.get("reasoning_content") or ""
                if token:
                    yield token
            except (ValueError, KeyError, IndexError):
                continue
