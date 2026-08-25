"""Runtime configuration for the inference service.

Override via environment variables when running locally or in containers.
Reads defaults from params.yaml so switching models only needs one edit.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Load params.yaml once
# ---------------------------------------------------------------------------
_PARAMS: dict = {}
_params_file = REPO_ROOT / "params.yaml"
if _yaml is not None and _params_file.is_file():
    with open(_params_file, encoding="utf-8") as _f:
        _PARAMS = _yaml.safe_load(_f) or {}


def _default_model_path() -> str:
    """Read model.path from params.yaml, falling back to models/gemma-q4.

    For GGUF models (llamacpp backend), auto-resolve the .gguf file inside
    the directory so llama-cpp-python gets a file path, not a directory.
    """
    model_dir = str(
        REPO_ROOT / _PARAMS.get("model", {}).get("path", "models/gemma-q4")
    )
    backend = _resolve_backend()

    # llamacpp needs a direct .gguf file path, not a directory
    if backend == "llamacpp" and Path(model_dir).is_dir():
        gguf_files = glob.glob(os.path.join(model_dir, "*.gguf"))
        if gguf_files:
            return gguf_files[0]

    return model_dir


def _resolve_backend() -> str:
    """Determine backend: env var > params.yaml > platform default."""
    env_val = os.getenv("BACKEND")
    if env_val:
        return env_val

    params_backend = _PARAMS.get("model", {}).get("backend", "default")
    if params_backend and params_backend != "default":
        return params_backend

    return "mlx" if sys.platform == "darwin" else "llamacpp"


MODEL_PATH: str = os.getenv("LANG_LEARN_MODEL_PATH", _default_model_path())

MAX_TOKENS: int = int(os.getenv("LANG_LEARN_MAX_TOKENS", "700"))
TEMPERATURE: float = float(os.getenv("LANG_LEARN_TEMPERATURE", "0.5"))

PROMPTS_DIR: Path = Path(__file__).resolve().parent.parent / "prompts"

# Backend selection:
#   "mlx"      — Apple Silicon (Mac), uses mlx + mlx-vlm
#   "llamacpp" — Linux / KServe containers, uses llama-cpp-python (GGUF)
BACKEND: str = _resolve_backend()

# Redis Configuration
REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))

# External LLM server (Mac-native llama-server with Metal acceleration)
# Used for exam generation/evaluation, word explainer, and dialogue generation
EXTERNAL_LLM_URL: str = os.getenv("EXTERNAL_LLM_URL", "http://192.168.2.1:9090")

# Backend selections: "external" (Mac llama-server) or "cluster" (in-cluster llamacpp)
EXPLAIN_WORD_BACKEND: str = os.getenv("EXPLAIN_WORD_BACKEND", "external")
DIALOGUE_BACKEND: str = os.getenv("DIALOGUE_BACKEND", "external")

def _default_embedding_model_path() -> str:
    local_path = REPO_ROOT / "models" / "multilingual-e5-small"
    container_path = Path("/app/models/multilingual-e5-small")
    if container_path.exists():
        return str(container_path)
    return str(local_path)

EMBEDDING_MODEL_PATH: str = os.getenv(
    "LANG_LEARN_EMBEDDING_MODEL_PATH",
    _default_embedding_model_path()
)
