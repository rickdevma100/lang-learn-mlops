"""BentoML service for lang-learn.

Run locally (from the inference/ directory):
    cd inference
    bentoml serve src.service:LangLearnService --reload --port 3000

Test:
    curl -X POST http://localhost:3000/scenario_dialogue \\
        -H "Content-Type: application/json" \\
        -d '{"scenario":"bargaining on cup price"}'
"""
from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

import bentoml

from .config import MAX_TOKENS, REPO_ROOT, TEMPERATURE
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
        except Exception as e:
            _write_error("warmup", e)
            raise

    @bentoml.api
    def scenario_dialogue(
        self,
        scenario: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> dict:
        """Generate an A1/A2 German dialogue for the given scenario."""
        try:
            template = load_prompt("scenario_dialogue.txt")
            prompt = template.format(scenario=scenario)
            text = generate(prompt, max_tokens=max_tokens, temperature=temperature)
            return {"response": text}
        except Exception as e:
            tb = _write_error("scenario_dialogue", e)
            return {
                "error": str(e),
                "type": type(e).__name__,
                "traceback": tb,
            }
