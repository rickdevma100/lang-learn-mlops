"""Backend interface for the lang-learn inference service.

Each backend module must expose:
    warmup() -> None
    generate(prompt, images, max_tokens, temperature) -> str
"""
from __future__ import annotations

from typing import Iterable


def warmup() -> None:  # pragma: no cover
    raise NotImplementedError


def generate(
    prompt: str,
    images: Iterable[str] | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:  # pragma: no cover
    raise NotImplementedError
