"""Runtime configuration for the inference service.

Override via environment variables when running locally or in containers.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH: str = os.getenv(
    "LANG_LEARN_MODEL_PATH",
    str(REPO_ROOT / "models" / "gemma-mlx"),
)

MAX_TOKENS: int = int(os.getenv("LANG_LEARN_MAX_TOKENS", "512"))
TEMPERATURE: float = float(os.getenv("LANG_LEARN_TEMPERATURE", "0.7"))

PROMPTS_DIR: Path = Path(__file__).resolve().parent.parent / "prompts"

# Backend selection:
#   "mlx"      — Apple Silicon (Mac), uses mlx + mlx-vlm
#   "llamacpp" — Linux / KServe containers, uses llama-cpp-python (GGUF)
_default_backend = "mlx" if sys.platform == "darwin" else "llamacpp"
BACKEND: str = os.getenv("BACKEND", _default_backend)
