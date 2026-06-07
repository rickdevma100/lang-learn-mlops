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
import re
import sys
import time
import traceback
from pathlib import Path

import bentoml
# pyrefly: ignore [missing-import]
from prometheus_client import make_asgi_app

from .config import MAX_TOKENS, REPO_ROOT, TEMPERATURE
from .metrics import MODEL_LOADED, USER_FEEDBACK, record_inference, record_cache_lookup
from .prompts import load_prompt
from .runner import generate, warmup
from .cache import SemanticCache

logger = logging.getLogger("lang_learn.service")
ERROR_LOG = Path(REPO_ROOT) / "inference" / "last_error.log"


def parse_dialogue(text: str) -> list[dict]:
    normalized = text.strip()
    if not re.match(r"(?i)^Person\s+[AB]", normalized):
        normalized = "Person A:\n" + normalized
        
    parts = re.split(r"(?i)\n*(Person\s+[AB])\s*:?\s*\n*", normalized)
    
    dialogue = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            speaker = parts[i].strip()
            if speaker.lower().endswith("a"):
                speaker = "Person A"
            elif speaker.lower().endswith("b"):
                speaker = "Person B"
            
            content = parts[i+1].strip()
            if content:
                # Extract German part and English translation
                subparts = re.split(r"(?i)\n*(?:Translation|English|Englisch)\s*:?\s*\n*", content, maxsplit=1)
                german_part = subparts[0].strip()
                english_part = subparts[1].strip() if len(subparts) > 1 else ""
                
                dialogue.append({
                    "speaker": speaker,
                    "german": german_part,
                    "english": english_part
                })
    return dialogue


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

        # Initialize semantic cache
        self.cache = SemanticCache()

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
            # 1. Semantic Cache Lookup
            is_hit, cached_response, similarity = self.cache.lookup(scenario, language, level)
            if is_hit and cached_response:
                latency = time.time() - t0
                # Record cache hit metric
                record_cache_lookup("scenario_dialogue", language, level, "hit", similarity)

                dialogue_turns = parse_dialogue(cached_response)
                cefr_score = record_inference(
                    endpoint="scenario_dialogue",
                    language=language,
                    level=level,
                    status="success",
                    latency_s=latency,
                    response_text=cached_response,
                    token_count=0,  # Cache hit results in 0 tokens generated
                )
                return {
                    "title": "Conversation",
                    "level": level,
                    "cefr_score": cefr_score,
                    "latency_s": round(latency, 2),
                    "dialogue": dialogue_turns,
                    "response": cached_response,
                    "result": cached_response,
                    "cached": True,
                    "cache_similarity": round(similarity, 3),
                }

            # Record cache miss metric (with closest similarity score if present)
            record_cache_lookup("scenario_dialogue", language, level, "miss", similarity)

            # 2. Cache Miss - Generate Dialogue
            template = load_prompt("scenario_dialogue.txt")
            prompt = template.format(scenario=scenario)
            text = generate(prompt, max_tokens=max_tokens, temperature=temperature)

            latency = time.time() - t0
            token_count = len(text.split())

            cefr_score = record_inference(
                endpoint="scenario_dialogue",
                language=language,
                level=level,
                status="success",
                latency_s=latency,
                response_text=text,
                token_count=token_count,
            )

            # 3. Store in Semantic Cache
            self.cache.store(text, scenario, language, level, cefr_score)

            dialogue_turns = parse_dialogue(text)

            return {
                "title": "Conversation",
                "level": level,
                "cefr_score": cefr_score,
                "latency_s": round(latency, 2),
                "dialogue": dialogue_turns,
                "response": text,
                "result": text,
                "cached": False,
            }

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

    @bentoml.api
    def explain_word(
        self,
        word: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> dict:
        """Explain a German word, providing part of speech, meaning, examples, and synonyms."""
        t0 = time.time()
        try:
            template = load_prompt("explain_word.txt")
            prompt = template.format(word=word)
            text = generate(prompt, max_tokens=max_tokens, temperature=temperature)

            latency = time.time() - t0
            token_count = len(text.split())

            record_inference(
                endpoint="explain_word",
                language="German",
                level="A2",
                status="success",
                latency_s=latency,
                response_text=text,
                token_count=token_count,
            )

            cleaned_text = text.strip()
            # Strip markdown code fences (```json ... ```, ``` ... ```, etc.)
            cleaned_text = re.sub(
                r"^```(?:json|JSON)?\s*\n?", "", cleaned_text
            )
            cleaned_text = re.sub(r"\n?\s*```\s*$", "", cleaned_text)
            cleaned_text = cleaned_text.strip()

            import json
            parsed_data = None

            # Attempt 1: Direct JSON parse
            try:
                parsed_data = json.loads(cleaned_text)
            except json.JSONDecodeError:
                pass

            # Attempt 2: Extract first JSON object via regex
            if parsed_data is None:
                json_match = re.search(
                    r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned_text, re.DOTALL
                )
                if json_match:
                    try:
                        parsed_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass

            # Attempt 3: Fallback with raw text
            if parsed_data is None:
                logger.warning(
                    "explain_word: Could not parse LLM JSON. Raw response: %s",
                    text[:200],
                )
                parsed_data = {
                    "word": word,
                    "part_of_speech": "unknown",
                    "meaning": text.strip(),
                    "example_sentence_german": "",
                    "example_sentence_english": "",
                    "synonyms": []
                }

            # Normalize keys and ensure defaults
            if parsed_data is not None:
                # Map potential misspelled/alternative keys to standard keys
                mappings = {
                    "example_sentence_sentence_english": "example_sentence_english",
                    "example_sentence_english_translation": "example_sentence_english",
                    "example_english": "example_sentence_english"
                }
                for alt_key, std_key in mappings.items():
                    if alt_key in parsed_data and std_key not in parsed_data:
                        parsed_data[std_key] = parsed_data.pop(alt_key)
                
                # Set default fields if missing
                parsed_data.setdefault("word", word)
                parsed_data.setdefault("part_of_speech", "unknown")
                parsed_data.setdefault("meaning", "")
                parsed_data.setdefault("example_sentence_german", "")
                parsed_data.setdefault("example_sentence_english", "")
                parsed_data.setdefault("synonyms", [])

            parsed_data["latency_s"] = round(latency, 2)
            parsed_data["response"] = text
            return parsed_data

        except Exception as e:
            latency = time.time() - t0
            record_inference(
                endpoint="explain_word",
                language="German",
                level="A2",
                status="error",
                latency_s=latency,
                response_text="",
                token_count=0,
            )
            tb = _write_error("explain_word", e)
            return {
                "error": str(e),
                "type": type(e).__name__,
                "traceback": tb,
            }


    # ----------------------------------------------------------------------
    
