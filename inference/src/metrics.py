"""Prometheus metrics for the lang-learn inference service.

All metrics follow the lang_learn_ prefix convention.
Import this module in service.py and call record_* helpers.
"""
from __future__ import annotations

# pyrefly: ignore [missing-import]
from prometheus_client import Counter, Gauge, Histogram, Info

# ---------------------------------------------------------------------------
# Service information
# ---------------------------------------------------------------------------
SERVICE_INFO = Info(
    "lang_learn_service",
    "Information about the language learning service",
)
SERVICE_INFO.info({
    "version": "1.0.0",
    "model": "gemma-4-gguf",
    "framework": "bentoml",
})

# ---------------------------------------------------------------------------
# Request metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "lang_learn_requests_total",
    "Total inference requests",
    ["endpoint", "language", "level", "status"],
)

LATENCY = Histogram(
    "lang_learn_latency_seconds",
    "Inference request latency in seconds",
    ["endpoint", "language", "level"],
    buckets=(0.5, 1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120),
)

# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------
CEFR_MATCH_SCORE = Gauge(
    "lang_learn_cefr_match_score",
    "Most recent CEFR match score (0-1)",
    ["endpoint", "language", "level"],
)

CEFR_MATCH_HISTOGRAM = Histogram(
    "lang_learn_cefr_match_score_distribution",
    "Distribution of CEFR match scores",
    ["endpoint", "language", "level"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

CEFR_MISMATCH_COUNT = Counter(
    "lang_learn_cefr_mismatches_total",
    "Total responses that did not match requested CEFR level",
    ["endpoint", "language", "level"],
)

# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------
TOKENS_GENERATED = Counter(
    "lang_learn_tokens_generated_total",
    "Total tokens generated",
    ["endpoint", "language", "level"],
)

# ---------------------------------------------------------------------------
# Resource metrics
# ---------------------------------------------------------------------------
GPU_MEMORY_USED = Gauge(
    "lang_learn_gpu_memory_bytes",
    "GPU memory used by inference container",
)

MODEL_LOADED = Gauge(
    "lang_learn_model_loaded",
    "1 if model loaded and ready, 0 otherwise",
)

# ---------------------------------------------------------------------------
# User feedback
# ---------------------------------------------------------------------------
USER_FEEDBACK = Counter(
    "lang_learn_user_feedback_total",
    "User thumbs up/down on responses",
    ["endpoint", "language", "level", "rating"],
)

# ---------------------------------------------------------------------------
# Helper: CEFR scoring
# ---------------------------------------------------------------------------
# A1-level German words (high-frequency, beginner vocabulary)
_A1_WORDS = frozenset([
    "ich", "du", "er", "sie", "es", "wir", "ihr",
    "bin", "bist", "ist", "sind", "seid",
    "habe", "hast", "hat", "haben", "habt",
    "ja", "nein", "bitte", "danke",
    "hallo", "guten", "morgen", "tag", "abend",
    "und", "oder", "aber", "nicht", "kein", "keine",
    "der", "die", "das", "ein", "eine",
    "was", "wer", "wo", "wie", "wann", "warum",
    "gut", "schlecht", "groß", "klein",
    "hier", "dort", "heute", "morgen", "jetzt",
    "essen", "trinken", "gehen", "kommen", "machen",
    "möchte", "kann", "muss", "will", "soll",
    "mit", "von", "zu", "in", "auf", "an", "für",
    "eins", "zwei", "drei", "vier", "fünf",
    "person", "gespräch", "dialog",
])

_B2_WORDS = frozenset([
    "allerdings", "beziehungsweise", "dementsprechend",
    "einverstanden", "folglich", "grundsätzlich",
    "hinsichtlich", "infolgedessen", "jedenfalls",
    "keineswegs", "letztendlich", "möglicherweise",
    "nichtsdestotrotz", "obwohl", "prinzipiell",
    "selbstverständlich", "tatsächlich", "übrigens",
    "vermutlich", "wahrscheinlich", "zufolge",
    "berücksichtigen", "beeinflussen", "entwickeln",
    "erörtern", "feststellen", "gewährleisten",
])

import re as _re


def compute_cefr_score(text: str, target_level: str = "A2") -> float:
    """Estimate a CEFR match score (0.0-1.0) for the generated text.

    Higher = more appropriate for the target level.
    A score of 1.0 means the vocabulary perfectly matches the target.
    """
    words = [w.lower().strip(".,!?;:\"'()") for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return 0.0

    total = len(words)
    a1_count = sum(1 for w in words if w in _A1_WORDS)
    b2_count = sum(1 for w in words if w in _B2_WORDS)
    german_chars = len(_re.findall(r"[äöüÄÖÜß]", text))
    dialogue_turns = len(_re.findall(r"Person [AB]:", text))

    # A1/A2: expect high A1 ratio, few B2, some German chars, multiple turns
    a1_ratio = a1_count / total
    b2_ratio = b2_count / total
    has_german = min(german_chars / max(total / 10, 1), 1.0)
    has_turns = min(dialogue_turns / 10.0, 1.0)  # at least 10 turns ideal

    if target_level in ("A1", "A2"):
        score = (
            0.40 * a1_ratio
            + 0.20 * (1.0 - b2_ratio)
            + 0.20 * has_german
            + 0.20 * has_turns
        )
    elif target_level in ("B1", "B2"):
        score = (
            0.20 * a1_ratio
            + 0.30 * b2_ratio
            + 0.25 * has_german
            + 0.25 * has_turns
        )
    else:
        score = 0.5 * has_german + 0.5 * has_turns

    return round(min(max(score, 0.0), 1.0), 3)


def record_inference(
    endpoint: str,
    language: str,
    level: str,
    status: str,
    latency_s: float,
    response_text: str,
    token_count: int = 0,
) -> float:
    """Record all metrics for a completed inference request.

    Returns the CEFR match score.
    """
    REQUEST_COUNT.labels(
        endpoint=endpoint, language=language, level=level, status=status
    ).inc()

    LATENCY.labels(endpoint=endpoint, language=language, level=level).observe(
        latency_s
    )

    if token_count > 0:
        TOKENS_GENERATED.labels(
            endpoint=endpoint, language=language, level=level
        ).inc(token_count)

    cefr_score = compute_cefr_score(response_text, level)

    CEFR_MATCH_SCORE.labels(
        endpoint=endpoint, language=language, level=level
    ).set(cefr_score)

    CEFR_MATCH_HISTOGRAM.labels(
        endpoint=endpoint, language=language, level=level
    ).observe(cefr_score)

    if cefr_score < 0.60:
        CEFR_MISMATCH_COUNT.labels(
            endpoint=endpoint, language=language, level=level
        ).inc()

    return cefr_score
