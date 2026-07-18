"""Backend dispatcher.

Reads the BACKEND env var and re-exports warmup/generate from the
appropriate backend module:

    BACKEND=mlx       -> backends.mlx_backend   (Mac, Apple Metal)
    BACKEND=llamacpp  -> backends.llamacpp_backend  (Linux / KServe)
"""
from __future__ import annotations

from .config import BACKEND

if BACKEND == "mlx":
    from .backends.mlx_backend import generate, warmup
    generate_stream = None  # MLX does not support streaming
elif BACKEND == "llamacpp":
    from .backends.llamacpp_backend import generate, generate_stream, warmup
else:
    raise ValueError(
        f"Unknown BACKEND={BACKEND!r}. Must be 'mlx' or 'llamacpp'."
    )

__all__ = ["warmup", "generate", "generate_stream"]
