"""BentoML service for lang-learn — instrumented with Prometheus metrics.

Run locally (from the inference/ directory):
    cd inference
    bentoml serve src.service:LangLearnService --reload --port 3000

Test:
    curl -X POST http://localhost:3000/scenario_dialogue \\
        -H "Content-Type: application/json" \\
        -d '{"scenario":"bargaining on cup price"}'

Metrics endpoint (auto-exposed by BentoML + prometheus_client):
    curl http://localhost:3000/metrics
"""
from __future__ import annotations

import logging
import sys
import time
import traceback
from pathlib import Path

import bentoml
from prometheus_client import make_asgi_app

from .config import MAX_TOKENS, REPO_ROOT, TEMPERATURE
from .metrics import MODEL_LOADED, USER_FEEDBACK, record_inference
from .prompts import load_prompt
from .runner import generate, warmup

logger = logging.getLogger("lang_learn.service")
ERROR_LOG = Path(REPO_ROOT) / "inference" / "last_error.log"


def _write_error(prefix: str, exc: BaseException) -> str:
    tb = traceback.format_exc()
    payload = f"=== {prefix} ===\n{type(exc).__name__}: {exc}\n{tb}\n"
    print(payload, file=sys.stderr, flush=True)
    try:
        ERROR_LOG.write_text(payload, encoding="utf-8")
    except Exception:
        pass
    return tb


@bentoml.service(
    name="lang-learn",
    traffic={"timeout": 300},
    resources={"cpu": "2", "memory": "16Gi"},
)
class LangLearnService:
    """Single-replica service exposing language-learning generation APIs."""

    def __init__(self) -> None:
        try:
            warmup()
            MODEL_LOADED.set(1)
        except Exception as e:
            MODEL_LOADED.set(0)
            _write_error("warmup", e)
            raise

    @bentoml.api
    def scenario_dialogue(
        self,
        scenario: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        language: str = "German",
        level: str = "A2",
    ) -> dict:
        """Generate an A1/A2 German dialogue for the given scenario.

        Extra parameters `language` and `level` are used for metric labels
        and CEFR scoring. They default to German/A2.
        """
        t0 = time.time()
        try:
            template = load_prompt("scenario_dialogue.txt")
            prompt = template.format(scenario=scenario)
            text = generate(prompt, max_tokens=max_tokens, temperature=temperature)

            latency = time.time() - t0
            token_count = len(text.split())  # rough estimate; replace with real count if available

            cefr_score = record_inference(
                endpoint="scenario_dialogue",
                language=language,
                level=level,
                status="success",
                latency_s=latency,
                response_text=text,
                token_count=token_count,
            )

            return {"response": text, "cefr_score": cefr_score, "latency_s": round(latency, 2)}

        except Exception as e:
            latency = time.time() - t0
            record_inference(
                endpoint="scenario_dialogue",
                language=language,
                level=level,
                status="error",
                latency_s=latency,
                response_text="",
                token_count=0,
            )
            tb = _write_error("scenario_dialogue", e)
            return {
                "error": str(e),
                "type": type(e).__name__,
                "traceback": tb,
            }

    @bentoml.api
    def feedback(
        self,
        endpoint: str,
        language: str,
        level: str,
        rating: str,
    ) -> dict:
        """Record user feedback (thumbs up/down).

        Args:
            endpoint:  The API endpoint the response came from.
            language:  Language of the response (e.g. 'German').
            level:     CEFR level (e.g. 'A2').
            rating:    'up' or 'down'.
        """
        if rating not in ("up", "down"):
            return {"error": "rating must be 'up' or 'down'"}

        USER_FEEDBACK.labels(
            endpoint=endpoint, language=language, level=level, rating=rating
        ).inc()
        return {"status": "recorded", "rating": rating}
