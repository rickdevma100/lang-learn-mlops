"""Runtime configuration for the inference service.

Override via environment variables when running locally or in containers.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH: str = os.getenv(
    "LANG_LEARN_MODEL_PATH",
    str(REPO_ROOT / "models" / "gemma-mlx"),
)

MAX_TOKENS: int = int(os.getenv("LANG_LEARN_MAX_TOKENS", "512"))
TEMPERATURE: float = float(os.getenv("LANG_LEARN_TEMPERATURE", "0.7"))

PROMPTS_DIR: Path = Path(__file__).resolve().parent.parent / "prompts"
