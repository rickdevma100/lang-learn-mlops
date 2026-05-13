"""Prompt template loader.

Templates live in `inference/prompts/` and are resolved relative to this
package so the service works regardless of the process working directory.
"""
from __future__ import annotations

from functools import lru_cache

from .config import PROMPTS_DIR


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")
