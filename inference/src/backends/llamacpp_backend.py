"""llama-cpp-python inference backend (GGUF Gemma).

Used on Linux/KServe where MLX (Apple Metal) is unavailable.
Model path should point to a GGUF file, e.g.:
    /app/models/gemma-3-4b-it-q4_k_m.gguf

Set via the LANG_LEARN_MODEL_PATH environment variable.
"""
from __future__ import annotations

from typing import Iterable

from ..config import MAX_TOKENS, MODEL_PATH, TEMPERATURE

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        # pyrefly: ignore [missing-import]
        from llama_cpp import Llama

        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_threads=4,
            verbose=False,
        )
    return _llm


def warmup() -> None:
    """Pre-load the GGUF model into memory."""
    _get_llm()


def generate(
    prompt: str,
    images: Iterable[str] | None = None,  # noqa: ARG001 — not supported by llama.cpp
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """Generate a completion using llama-cpp-python (CPU/GPU via llama.cpp).

    Uses chat completion format so instruction-tuned models (Gemma-IT)
    apply their chat template and produce proper responses.
    """
    llm = _get_llm()
    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return output["choices"][0]["message"]["content"]
