"""MLX-VLM inference runner.

Loads the Gemma 4 multimodal model from disk exactly once (lazy singleton) and
exposes a single `generate()` function used by the BentoML service.

Runtime: Apple Silicon (Metal) via the `mlx` and `mlx-vlm` packages.

Threading strategy
------------------
MLX ≥ 0.31 makes GPU streams thread-local.  ``mlx_vlm`` creates a module-level
``ThreadLocalStream`` and calls ``mx.async_eval`` outside its ``mx.stream()``
context, which crashes when the caller is a BentoML worker thread (the stream
object was initialised on the import-thread, not the worker).

We solve this by funnelling **all** inference through a single long-lived
daemon thread.  BentoML workers submit jobs via a queue and block on a
``concurrent.futures.Future`` until the result is ready.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from typing import Iterable

from .config import MAX_TOKENS, MODEL_PATH, TEMPERATURE

_engine: tuple | None = None
_lock = threading.Lock()

# Single-slot job queue for the inference thread.
_job_queue: queue.Queue | None = None
_worker_thread: threading.Thread | None = None


def _get_engine() -> tuple:
    """Lazy-load model, processor, and config on first call (inference thread)."""
    global _engine
    if _engine is None:
        from mlx_vlm import load as _mlx_load
        from mlx_vlm.utils import load_config

        model, processor = _mlx_load(MODEL_PATH)
        config = load_config(MODEL_PATH)
        _engine = (model, processor, config)
    return _engine


# ── Inference thread ────────────────────────────────────────────────
def _inference_loop(q: queue.Queue) -> None:
    """Runs forever on a daemon thread; owns all MLX GPU streams."""
    # Import mlx_vlm here so the ThreadLocalStream is created on THIS thread.
    from mlx_vlm import generate as _mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    _get_engine()  # warm up on this thread

    while True:
        fut, prompt, images, max_tokens, temperature = q.get()
        try:
            model, processor, config = _get_engine()

            image_list = list(images) if images else []
            formatted = apply_chat_template(
                processor, config, prompt, num_images=len(image_list)
            )

            result = _mlx_generate(
                model,
                processor,
                formatted,
                image=image_list or None,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            )
            text = result.text if hasattr(result, "text") else str(result)
            fut.set_result(text)
        except Exception as exc:
            fut.set_exception(exc)


def _ensure_worker() -> queue.Queue:
    """Start the inference thread (once) and return the job queue."""
    global _job_queue, _worker_thread
    if _job_queue is None:
        with _lock:
            if _job_queue is None:
                _job_queue = queue.Queue(maxsize=1)
                _worker_thread = threading.Thread(
                    target=_inference_loop, args=(_job_queue,), daemon=True
                )
                _worker_thread.start()
    return _job_queue


def warmup() -> None:
    """Force model load. Call at service startup to avoid first-request latency."""
    q = _ensure_worker()
    # Send a no-op generate to warm up the model on the inference thread.
    fut: Future[str] = Future()
    q.put((fut, "hello", None, 1, 0.0))
    fut.result(timeout=120)  # wait for model load + 1-token gen


def generate(
    prompt: str,
    images: Iterable[str] | None = None,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """Generate a completion for `prompt`.

    `images` is an optional iterable of file paths, URLs, or PIL images. For
    text-only generation pass `None` (or an empty list).

    This call blocks until the dedicated inference thread finishes.
    """
    q = _ensure_worker()
    fut: Future[str] = Future()
    q.put((fut, prompt, images, max_tokens, temperature))
    return fut.result()  # blocks until inference completes



