"""External llama-server HTTP backend for exam, dialogue, and word explainer APIs.

Calls a llama-server instance running natively on the Mac host (Metal-accelerated)
via HTTP. Uses standard library urllib for zero external dependencies and maximum portability.

Set via environment variable:
  EXTERNAL_LLM_URL → base URL of the llama-server (e.g. http://192.168.2.1:9090)
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Iterable

try:
    from ..config import EXTERNAL_LLM_URL
except (ImportError, AttributeError):
    EXTERNAL_LLM_URL = os.getenv("EXTERNAL_LLM_URL", "http://192.168.2.1:9090")

logger = logging.getLogger("lang_learn.backends.external_llama")


def _get_base_url() -> str:
    return os.getenv("EXTERNAL_LLM_URL", EXTERNAL_LLM_URL).rstrip("/")


def health_check() -> bool:
    """Check if the external llama-server is reachable."""
    try:
        url = f"{_get_base_url()}/health"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
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

    Uses Gemma's turn template with prompt prefixing to ensure instant, valid output.
    """
    # If the prompt requests a JSON array or object, prefix the model turn to prevent rambling/thinking
    if "JSON array" in prompt or "[\n  {" in prompt:
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n```json\n["
        prefix = "["
    elif "JSON object" in prompt or "{\n" in prompt or "Return ONLY a valid JSON" in prompt:
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n```json\n"
        prefix = "```json\n"
    elif prompt.strip().endswith("Person A:"):
        base_prompt = prompt.rstrip()[:-len("Person A:")].rstrip()
        formatted_prompt = f"<start_of_turn>user\n{base_prompt}<end_of_turn>\n<start_of_turn>model\nPerson A:\n"
        prefix = "Person A:\n"
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

    url = f"{_get_base_url()}/completion"
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"}
    )

    logger.info("Calling external LLM at %s (max_tokens=%d)", url, max_tokens)
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))

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
    if prompt.strip().endswith("Person A:"):
        base_prompt = prompt.rstrip()[:-len("Person A:")].rstrip()
        formatted_prompt = f"<start_of_turn>user\n{base_prompt}<end_of_turn>\n<start_of_turn>model\nPerson A:\n"
        initial_token = "Person A:\n"
    else:
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        initial_token = ""

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

    if initial_token:
        yield initial_token

    url = f"{_get_base_url()}/completion"
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=120.0) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
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
