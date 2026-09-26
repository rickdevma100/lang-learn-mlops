"""Backend dispatcher.

Reads the BACKEND env var and re-exports warmup/generate from the
appropriate backend module:

    BACKEND=mlx       -> backends.mlx_backend   (Mac, Apple Metal)
    BACKEND=llamacpp  -> backends.llamacpp_backend  (Linux / KServe)
    BACKEND=external  -> backends.external_llama_backend  (Mac-native llama-server via HTTP)
"""
from __future__ import annotations

import logging

from .config import BACKEND

logger = logging.getLogger("lang_learn.runner")

if BACKEND == "mlx":
    from .backends.mlx_backend import generate, warmup
    generate_stream = None  # MLX does not support streaming
elif BACKEND == "llamacpp":
    from .backends.llamacpp_backend import generate, generate_stream, warmup
elif BACKEND == "external":
    from .backends.external_llama_backend import (
        generate_external as generate,
        generate_external_stream as generate_stream,
        health_check as _health_check,
    )

    def warmup() -> None:
        """Verify connectivity to the external llama-server (no local model loaded)."""
        logger.info("External backend: checking llama-server health...")
        if _health_check():
            logger.info("External llama-server is reachable and healthy")
        else:
            logger.warning(
                "External llama-server is NOT reachable — generation calls "
                "will fail until the server is available"
            )
else:
    raise ValueError(
        f"Unknown BACKEND={BACKEND!r}. Must be 'mlx', 'llamacpp', or 'external'."
    )

# Backward compat alias — exam generation uses the same model now
generate_exam = generate

__all__ = ["warmup", "generate", "generate_exam", "generate_stream"]
