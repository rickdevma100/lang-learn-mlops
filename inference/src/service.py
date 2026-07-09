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
from typing import Annotated

import bentoml
# pyrefly: ignore [missing-import]
from bentoml.validators import ContentType
# pyrefly: ignore [missing-import]
from prometheus_client import make_asgi_app

from .config import MAX_TOKENS, REPO_ROOT, TEMPERATURE
from .metrics import MODEL_LOADED, USER_FEEDBACK, record_inference, record_cache_lookup
from .prompts import load_prompt
from .runner import generate, warmup
from .cache import SemanticCache
from .tts import synthesize_line_sync

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
    resources={"cpu": "4", "memory": "16Gi"},
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

        # Load faster-whisper model for speech-to-text
        try:
            # pyrefly: ignore [missing-import]
            from faster_whisper import WhisperModel
            self.whisper_model = WhisperModel(
                "tiny", device="cpu", compute_type="int8"
            )
            logger.info("faster-whisper 'tiny' model loaded successfully")
        except Exception as e:
            logger.warning("Could not load faster-whisper model: %s", e)
            self.whisper_model = None

    @bentoml.api
    def scenario_dialogue(
        self,
        scenario: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        language: str = "German",
        level: str = "A2",
        bypass_cache: bool = False,
        prompt_template: str = "",
    ) -> dict:
        """Generate an A1/A2 German dialogue for the given scenario.

        Extra parameters `language` and `level` are used for metric labels
        and CEFR scoring. They default to German/A2.

        Set `bypass_cache=True` to skip the semantic cache (used by the
        prompt optimizer during benchmarking so each candidate gets a
        fresh LLM-generated response).

        Set `prompt_template` to override the default prompt loaded from
        disk. The template must contain a `{scenario}` placeholder.
        Used by the prompt optimizer to benchmark candidate prompt
        variations without modifying the ConfigMap.
        """
        t0 = time.time()
        try:
            # 1. Semantic Cache Lookup (skipped when bypass_cache is True)
            if not bypass_cache:
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

            # 2. Generate Dialogue (cache miss or bypass)
            if prompt_template:
                template = prompt_template
            else:
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

            # 3. Store in Semantic Cache (skipped when bypass_cache is True
            #    to avoid polluting the cache with benchmark variations)
            if not bypass_cache:
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
    def clear_cache(self) -> dict:
        """Clear all semantic cache entries from Redis."""
        try:
            success = self.cache.clear()
            return {"status": "success", "cleared": success}
        except Exception as e:
            logger.error("Error clearing cache: %s", e)
            return {"status": "error", "message": str(e)}

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

    @bentoml.api
    def rewrite_prompt(
        self,
        base_prompt: str,
        suffix: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> dict:
        """Rewrite the base prompt to incorporate the winner suffix rules using the LLM."""
        meta_prompt = (
            "You are a prompt engineering expert. Your task is to rewrite a base prompt to seamlessly "
            "incorporate additional instructions or rules.\n\n"
            "Original Base Prompt:\n"
            "\"\"\"\n"
            f"{base_prompt}\n"
            "\"\"\"\n\n"
            "Instructions/Rules to incorporate:\n"
            "\"\"\"\n"
            f"{suffix}\n"
            "\"\"\"\n\n"
            "Generate the final prompt. Do not include any explanation or markdown block quotes (such as ```). "
            "Output only the final rewritten prompt text."
        )
        t0 = time.time()
        try:
            text = generate(meta_prompt, max_tokens=max_tokens, temperature=temperature)
            cleaned = text.strip()
            # Clean markdown code blocks if the LLM outputted them anyway
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()
            
            logger.info("Successfully rewrote prompt via LLM in %.2fs", time.time() - t0)
            return {"prompt": cleaned}
        except Exception as e:
            logger.error("Failed to rewrite prompt: %s", e)
            return {"error": str(e)}

    @bentoml.api
    def reload_prompts(self) -> dict:
        """Clear the prompt LRU cache so prompts are re-read from disk.

        Called by the prompt optimizer after updating the ConfigMap.
        ConfigMap changes propagate to the mounted volume within ~60s,
        so the optimizer waits before calling this endpoint.
        """
        try:
            load_prompt.cache_clear()
            logger.info("Prompt LRU cache cleared — prompts will be re-read from disk.")
            return {
                "status": "reloaded",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        except Exception as e:
            logger.error("Failed to clear prompt cache: %s", e)
            return {"error": str(e)}

    @bentoml.api
    def tts(
        self,
        context: bentoml.Context,
        text: str,
        language: str = "German",
        level: str = "A2",
        speaker: str = "default",
    ) -> Annotated[Path, ContentType("audio/mpeg")]:
        """Synthesize a dialogue line to MP3 audio using edge-tts.

        Checks the Redis audio cache first. Generates via edge-tts on a miss
        and stores the result with a 30-day TTL.

        Args:
            text:     The text to synthesize (German sentence, etc.).
            language: Target language (e.g. 'German').
            level:    CEFR level controlling speech rate (e.g. 'A2').
            speaker:  Voice role — 'person_a', 'person_b', or 'default'.

        Returns:
            MP3 audio file path (served by BentoML as audio/mpeg).
        """
        # 1. Cache hit?
        cached = self.cache.get_audio(text, language, level, speaker)
        if cached:
            logger.debug("TTS audio cache hit for text: %s...", text[:40])
            output_path = Path(context.temp_dir) / "tts.mp3"
            output_path.write_bytes(cached)
            return output_path

        # 2. Synthesize
        audio = synthesize_line_sync(text, language, level, speaker)

        # 3. Store in cache
        if audio:
            self.cache.set_audio(text, language, level, speaker, audio)

        output_path = Path(context.temp_dir) / "tts.mp3"
        output_path.write_bytes(audio)
        return output_path

    @bentoml.api
    def stt(
        self,
        audio: Annotated[Path, ContentType("audio/*")],
        language: str = "de",
    ) -> dict:
        """Transcribe audio to text using faster-whisper.

        Used by Practice Mode for voice input. Accepts any audio format
        supported by FFmpeg (WebM, WAV, MP3, etc.).

        Args:
            audio:    Path to the uploaded audio file.
            language: Language code for transcription (default: 'de' for German).

        Returns:
            Dict with text, language, and duration_s.
        """
        import os

        t0 = time.time()

        if self.whisper_model is None:
            return {
                "error": "Speech-to-text model not available",
                "text": "",
            }

        try:
            segments, info = self.whisper_model.transcribe(
                str(audio),
                language=language,
                beam_size=3,
                vad_filter=True,
            )
            text = " ".join(seg.text.strip() for seg in segments)
            latency = time.time() - t0

            logger.info(
                "STT: transcribed %.1fs audio in %.2fs → '%s'",
                info.duration, latency, text[:80],
            )

            return {
                "text": text,
                "language": info.language,
                "duration_s": round(info.duration, 2),
                "latency_s": round(latency, 2),
            }

        except Exception as e:
            latency = time.time() - t0
            tb = _write_error("stt", e)
            return {
                "error": str(e),
                "text": "",
                "type": type(e).__name__,
                "traceback": tb,
            }
        finally:
            # Explicitly clean up the temp audio file
            try:
                os.unlink(str(audio))
            except OSError:
                pass

    @bentoml.api
    def practice_check(
        self,
        user_text: str,
        expected_english: str,
        scenario: str,
        speaker: str = "Person A",
        preceding_context: str = "",
        language: str = "German",
        level: str = "A2",
        max_tokens: int = 128,
        temperature: float = 0.1,
    ) -> dict:
        """Check a student's practice answer for grammar and context relevance.

        Used by Practice Mode — validates a single German sentence without
        regenerating the dialogue. No Redis caching.

        Args:
            user_text:          The student's German sentence.
            expected_english:   The English translation hint shown to the student.
            scenario:           The conversation scenario (e.g. 'ordering food').
            speaker:            Who the student is speaking as (e.g. 'Person B').
            preceding_context:  The 2-3 dialogue lines before this turn.
            language:           Target language (default: German).
            level:              CEFR level (default: A2).
            max_tokens:         Max tokens for LLM generation.
            temperature:        LLM temperature (low for consistency).

        Returns:
            Dict with grammar_ok, on_topic, feedback, corrected_text, score.
        """
        import json as _json

        t0 = time.time()

        try:
            template = load_prompt("practice_check.txt")
            prompt = template.format(
                level=level,
                scenario=scenario,
                speaker=speaker,
                preceding_context=preceding_context,
                expected_english=expected_english,
                user_text=user_text,
            )

            text = generate(prompt, max_tokens=max_tokens, temperature=temperature)
            latency = time.time() - t0

            # Parse JSON response
            cleaned = text.strip()
            cleaned = re.sub(r"^```(?:json|JSON)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?\s*```\s*$", "", cleaned)
            cleaned = cleaned.strip()

            parsed = None
            try:
                parsed = _json.loads(cleaned)
            except _json.JSONDecodeError:
                json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if json_match:
                    try:
                        parsed = _json.loads(json_match.group())
                    except _json.JSONDecodeError:
                        pass

            if parsed is None:
                logger.warning("practice_check: Could not parse LLM JSON: %s", cleaned[:300])
                return {
                    "error": "Failed to parse LLM response",
                    "raw_response": cleaned[:500],
                    "latency_s": round(latency, 2),
                }

            # Normalize and ensure defaults
            result = {
                "grammar_ok": parsed.get("grammar_ok", True),
                "on_topic": parsed.get("on_topic", True),
                "feedback": parsed.get("feedback", ""),
                "corrected_text": parsed.get("corrected_text"),
                "score": float(parsed.get("score", 1.0)),
                "latency_s": round(latency, 2),
            }

            record_inference(
                endpoint="practice_check",
                language=language,
                level=level,
                status="success",
                latency_s=latency,
                response_text=text,
                token_count=len(text.split()),
            )

            return result

        except Exception as e:
            latency = time.time() - t0
            record_inference(
                endpoint="practice_check",
                language=language,
                level=level,
                status="error",
                latency_s=latency,
                response_text="",
                token_count=0,
            )
            tb = _write_error("practice_check", e)
            return {
                "error": str(e),
                "type": type(e).__name__,
                "traceback": tb,
            }

    # ----------------------------------------------------------------------

