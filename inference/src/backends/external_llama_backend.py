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
import os
from typing import Iterable

import httpx

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
    temperature: float = 0.2,
) -> str:
    """Generate text using the external llama-server.

    Uses Gemma's turn template with prompt prefixing to ensure instant, valid JSON output.
    """
    client = _get_client()

    # If the prompt requests a JSON array or object, prefix the model turn to prevent rambling/thinking
    if "JSON array" in prompt or "[\n  {" in prompt:
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n```json\n["
        prefix = "["
    elif "JSON object" in prompt or "{\n" in prompt or "Return ONLY a valid JSON" in prompt:
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n```json\n"
        prefix = "```json\n"
    else:
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        prefix = ""

    payload = {
        "prompt": formatted_prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "stream": False,
        "stop": ["<end_of_turn>", "</s>"],
    }

    logger.info("Calling external LLM at %s/completion (max_tokens=%d)", EXTERNAL_LLM_URL, max_tokens)
    response = client.post("/completion", json=payload)
    response.raise_for_status()

    data = response.json()
    content = prefix + (data.get("content") or "").strip()
    tokens_used = data.get("tokens_predicted", len(content.split()))
    logger.info("External LLM response: %d chars, ~%d tokens", len(content), tokens_used)
    return content


def generate_external_stream(
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> Iterable[str]:
    """Stream tokens from the external llama-server via SSE."""
    client = _get_client()

    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    payload = {
        "prompt": formatted_prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "stream": True,
        "stop": ["<end_of_turn>", "</s>"],
    }

    with client.stream("POST", "/completion", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if not data_str or data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                token = chunk.get("content", "")
                if token:
                    yield token
                if chunk.get("stop", False):
                    break
            except (ValueError, KeyError, IndexError):
                continue
