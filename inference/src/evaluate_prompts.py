"""Prompt quality evaluation script for DVC experiments.

This script is the core of the DVC pipeline's `evaluate_prompts` stage.
It calls the running inference service (BentoML in Docker) via HTTP,
collects the generated outputs, and computes quality metrics that DVC
tracks across experiments.

Prerequisites:
    The inference service must be running (e.g. via Docker):
        docker run --rm -p 3001:3000 ...

Usage (via DVC):
    dvc repro evaluate_prompts
    dvc exp run --name "baseline"

Usage (standalone):
    python -m inference.src.evaluate_prompts
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = REPO_ROOT / "params.yaml"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
METRICS_DIR = REPO_ROOT / "metrics"


def load_params() -> dict:
    """Load params.yaml from the repo root."""
    with open(PARAMS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# HTTP client — calls the running inference service
# ---------------------------------------------------------------------------

def call_inference_service(
    service_url: str,
    scenario: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    timeout: int = 300,
) -> str:
    """Call the BentoML inference service via HTTP and return the response text.

    Equivalent to:
        curl -X POST http://localhost:3001/scenario_dialogue \
            -H "Content-Type: application/json" \
            -d '{"scenario":"..."}'
    """
    url = f"{service_url.rstrip('/')}/scenario_dialogue"
    payload = json.dumps({
        "scenario": scenario,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot reach inference service at {url}. "
            f"Is the Docker container running?\n"
            f"  Start it with: docker run --rm -p 3001:3000 ...\n"
            f"  Error: {e}"
        ) from e

    if "error" in body:
        raise RuntimeError(f"Inference service error: {body['error']}")

    return body.get("response", "")


# ---------------------------------------------------------------------------
# Text-analysis helpers
# ---------------------------------------------------------------------------

# Common A1-level German words (high-frequency, beginner vocabulary)
A1_WORDS = frozenset([
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
])

# B2-level words (more advanced vocabulary)
B2_WORDS = frozenset([
    "allerdings", "beziehungsweise", "dementsprechend",
    "einverstanden", "folglich", "grundsätzlich",
    "hinsichtlich", "infolgedessen", "jedenfalls",
    "keineswegs", "letztendlich", "möglicherweise",
    "nichtsdestotrotz", "obwohl", "prinzipiell",
    "selbstverständlich", "tatsächlich", "übrigens",
    "vermutlich", "wahrscheinlich", "zufolge",
    "berücksichtigen", "beeinflussen", "entwickeln",
    "erörtern", "feststellen", "gewährleisten",
    "hervorheben", "unterscheiden", "voraussetzen",
    "zusammenfassen", "außerdem", "darüber",
])


def count_words(text: str) -> int:
    """Count words in generated text."""
    words = text.split()
    return len(words)


def count_sentences(text: str) -> int:
    """Count sentences based on terminal punctuation."""
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def compute_vocab_level(text: str) -> dict:
    """Estimate CEFR vocabulary distribution in the generated text."""
    words = [w.lower().strip(".,!?;:\"'()") for w in text.split()]
    words = [w for w in words if w]

    total = len(words) if words else 1
    a1_count = sum(1 for w in words if w in A1_WORDS)
    b2_count = sum(1 for w in words if w in B2_WORDS)
    other_count = total - a1_count - b2_count

    return {
        "a1_word_count": a1_count,
        "b2_word_count": b2_count,
        "other_word_count": other_count,
        "a1_ratio": round(a1_count / total, 3),
        "b2_ratio": round(b2_count / total, 3),
    }


def count_german_chars(text: str) -> float:
    """Estimate the fraction of text that is German (contains umlauts, ß, etc.)."""
    german_specific = len(re.findall(r'[äöüÄÖÜß]', text))
    # A rough heuristic: if there are German-specific chars, it's German
    total_alpha = len(re.findall(r'[a-zA-ZäöüÄÖÜß]', text))
    if total_alpha == 0:
        return 0.0
    # Count Person A/B labels as non-German overhead
    cleaned = re.sub(r'Person [AB]:', '', text)
    alpha_chars = len(re.findall(r'[a-zA-ZäöüÄÖÜß]', cleaned))
    return round(alpha_chars / max(total_alpha, 1), 3)


def count_dialogue_turns(text: str) -> int:
    """Count dialogue turns (Person A: / Person B: lines)."""
    return len(re.findall(r'Person [AB]:', text))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate() -> dict:
    """Run prompt evaluation via the inference service and return metrics."""
    params = load_params()
    model_cfg = params.get("model", {})
    inference_cfg = params.get("inference", {})
    eval_cfg = params.get("evaluate", {})

    service_url = inference_cfg.get("service_url", "http://localhost:3001")
    scenarios = eval_cfg.get("test_scenarios", ["ordering food at a restaurant"])
    num_samples = eval_cfg.get("num_samples", 1)
    max_tokens = inference_cfg.get("max_tokens", 512)
    temperature = inference_cfg.get("temperature", 0.7)

    print(f"Inference service: {service_url}")
    print(f"Model: {model_cfg.get('name', 'unknown')} ({model_cfg.get('path', 'unknown')})")
    print(f"Scenarios: {len(scenarios)}, Samples/scenario: {num_samples}")
    print(f"Max tokens: {max_tokens}, Temperature: {temperature}")
    print()

    # Collect metrics across all scenarios
    all_word_counts = []
    all_sentence_counts = []
    all_a1_words = []
    all_b2_words = []
    all_a1_ratios = []
    all_b2_ratios = []
    all_dialogue_turns = []
    all_german_ratios = []
    total_time = 0.0

    for scenario in scenarios:
        for sample_idx in range(num_samples):
            print(f"  Evaluating: '{scenario}' (sample {sample_idx + 1}/{num_samples})")

            start = time.time()
            output = call_inference_service(
                service_url=service_url,
                scenario=scenario,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed = time.time() - start
            total_time += elapsed

            print(f"    Generated {len(output)} chars in {elapsed:.1f}s")

            # Compute metrics
            wc = count_words(output)
            sc = count_sentences(output)
            vocab = compute_vocab_level(output)
            turns = count_dialogue_turns(output)
            german_ratio = count_german_chars(output)

            all_word_counts.append(wc)
            all_sentence_counts.append(sc)
            all_a1_words.append(vocab["a1_word_count"])
            all_b2_words.append(vocab["b2_word_count"])
            all_a1_ratios.append(vocab["a1_ratio"])
            all_b2_ratios.append(vocab["b2_ratio"])
            all_dialogue_turns.append(turns)
            all_german_ratios.append(german_ratio)

    n = len(all_word_counts) or 1

    metrics = {
        "model_name": model_cfg.get("name", "unknown"),
        "model_path": model_cfg.get("path", "unknown"),
        "prompt_template": "scenario_dialogue",
        "scenarios_evaluated": len(scenarios),
        "samples_per_scenario": num_samples,
        "total_generations": n,
        "avg_word_count": round(sum(all_word_counts) / n, 1),
        "avg_sentence_count": round(sum(all_sentence_counts) / n, 1),
        "avg_dialogue_turns": round(sum(all_dialogue_turns) / n, 1),
        "avg_german_ratio": round(sum(all_german_ratios) / n, 3),
        "a1_words": round(sum(all_a1_words) / n, 1),
        "b2_words": round(sum(all_b2_words) / n, 1),
        "a1_ratio": round(sum(all_a1_ratios) / n, 3),
        "b2_ratio": round(sum(all_b2_ratios) / n, 3),
        "avg_generation_time_s": round(total_time / n, 2),
        "total_time_s": round(total_time, 2),
    }

    return metrics


def main() -> None:
    """Entry point for DVC pipeline stage."""
    print("=" * 60)
    print("DVC Stage: evaluate_prompts")
    print("=" * 60)

    metrics = evaluate()

    # Write metrics
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = METRICS_DIR / "prompt_quality.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nMetrics written to: {metrics_file}")
    print(json.dumps(metrics, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
