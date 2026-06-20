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
    "abend", "abendessen", "aber", "acht", "alle", "allein", "alles", "alt",
    "an", "antworten", "apfel", "apotheke", "arbeit", "arbeiten", "arzt", "auch",
    "auf", "auf Wiedersehen", "auge", "aus", "ausbildung", "auto",
    "bahn", "bahnhof", "ball", "banane", "bank", "baum", "beide", "beim",
    "bein", "beispiel", "beißen", "bekommen", "bereit", "berg", "beruf", "bett",
    "bezahlen", "bier", "bild", "billig", "bin", "birne", "bist", "bitte",
    "blau", "bleiben", "bleistift", "blume", "boden", "brot", "bruder", "buch",
    "bus", "butter", "café", "computer", "da", "danke", "dann", "darf",
    "das", "dein", "dem", "den", "denken", "der", "des", "deutsch",
    "dialog", "dich", "die", "dienstag", "ding", "dir", "doch", "donnerstag",
    "dort", "drei", "du", "dunkel", "durch", "duschen", "durst", "ecke",
    "ei", "ein", "eine", "einem", "einen", "einer", "eines", "einfach",
    "einladen", "einkaufen", "einmal", "eis", "eltern", "ende", "englisch", "entschuldigung",
    "er", "erde", "erst", "es", "essen", "euch", "euer", "fahren",
    "fahrrad", "familie", "fast", "vater", "fehlen", "fenster", "ferien", "fernsehen",
    "fertig", "feuer", "finden", "fisch", "flasche", "fleisch", "fliegen", "flughafen",
    "flugzeug", "fluss", "fragen", "frau", "frei", "freitag", "freizeit", "freund",
    "freundin", "frisch", "froh", "früh", "frühling", "frühstück", "für", "fuß",
    "fußball", "geben", "gegen", "gehen", "gelb", "geld", "gemüse", "genug",
    "gern", "gerne", "gespräch", "gesund", "glauben", "groß", "großeltern", "grün",
    "gut", "guten", "haar", "haben", "habt", "hallo", "hand", "handy",
    "hast", "hat", "hatte", "haus", "heiß", "helfen", "herbst", "herr",
    "heute", "hier", "himmel", "hinter", "hobby", "hoch", "holen", "hören",
    "hotel", "hund", "hunger", "ich", "ihm", "ihn", "ihr", "immer",
    "in", "ist", "ja", "jahr", "jeder", "jetzt", "jung", "kaffee",
    "kalt", "kann", "karte", "kartoffel", "käse", "kaufen", "kein", "keine",
    "kind", "kino", "klasse", "klein", "kochen", "koffer", "kollege", "kommen",
    "können", "kopf", "kosten", "krank", "krankenhaus", "küche", "kuchen", "kugelschreiber",
    "kurz", "lachen", "laden", "land", "lang", "langsam", "laufen", "laut",
    "leben", "legen", "lehrer", "leicht", "leise", "lernen", "lesen", "letzte",
    "leute", "licht", "lieb", "lieben", "lieber", "liegen", "links", "luft",
    "lustig", "machen", "mädchen", "mann", "markt", "mehr", "mein", "mensch",
    "messer", "milch", "minute", "mit", "mitbringen", "mittag", "mittagessen", "mitte",
    "mittwoch", "möchte", "mitnehmen", "monat", "montag", "morgen", "müde", "mund",
    "musik", "müssen", "mutter", "nach", "nachmittag", "nah", "name", "natur",
    "neben", "nehmen", "nein", "nennen", "nett", "neu", "neun", "nicht",
    "nichts", "nie", "noch", "nudeln", "nummer", "nur", "ob", "obst",
    "oder", "offen", "öffnen", "oft", "ohne", "onkel", "orange", "ort",
    "papier", "park", "partner", "person", "pfeffer", "pferd", "platz", "post",
    "preis", "problem", "rad", "regen", "regnen", "reich", "reise", "reisen",
    "restaurant", "richtig", "riechen", "rot", "rufen", "ruhig", "rund", "sache",
    "saft", "sagen", "salat", "salz", "samstag", "sauber", "sauer", "schade",
    "schlafen", "schlecht", "schließen", "schlüssel", "schmecken", "schmutzig", "schnee", "schnell",
    "schon", "schön", "schreiben", "schuh", "schule", "schüler", "schwarz", "schwer",
    "schwester", "schwimmen", "sechs", "see", "sehen", "sehr", "seid", "sein",
    "seit", "seite", "selbst", "senden", "sessel", "sich", "sie", "sieben",
    "sind", "singen", "sitzen", "sohn", "sommer", "sonne", "sonntag", "spaß",
    "spät", "spazieren", "speisekarte", "spielen", "sport", "sprechen", "stadt", "stand",
    "stark", "stehen", "stelle", "stellen", "stift", "straße", "stück", "stuhl",
    "suchen", "suppe", "süß", "tag", "tante", "tasche", "tasse", "tee",
    "telefon", "teller", "teuer", "tief", "tier", "tisch", "tochter", "toll",
    "tomate", "tragen", "treffen", "trinken", "tschüss", "tür", "uhr", "über",
    "um", "und", "uns", "unser", "unter", "urlaub", "verkaufen", "verstehen",
    "viel", "vielleicht", "vier", "vom", "von", "vor", "vormittag", "wagen",
    "wald", "wann", "warm", "warum", "was", "waschen", "wasser", "weg",
    "weich", "wein", "weit", "weiter", "welt", "wenig", "wer", "werden",
    "werfen", "wetter", "wichtig", "wie", "wieder", "will", "wind", "winter",
    "wir", "wissen", "wo", "woche", "wochenende", "woher", "wohin", "wohnen",
    "wohnung", "wollen", "wort", "wunderbar", "zahlen", "zahn", "zehn", "zeigen",
    "zeit", "zeitung", "zimmer", "zu", "zucker", "zug", "zusammen", "zwei",
    "zwischen",
    # Additional conjugated verbs and dialogue vocabulary for CEFR matching
    "möchten", "möchtest", "möchtet", "kannst", "können", "könnt", "könnte", "könnten",
    "muss", "musst", "müssen", "müsst", "soll", "sollst", "sollen", "sollt",
    "will", "willst", "wollen", "wollt", "darf", "darfst", "dürfen", "dürft",
    "werde", "wirst", "wird", "werden", "werdet", "war", "warst", "waren", "wart",
    "hattest", "hatten", "hattet", "trinke", "trinkst", "trinkt", "trinket",
    "esse", "isst", "esst", "esset", "gehe", "gehst", "geht", "gehet",
    "komme", "kommest", "kommt", "kommet", "nehme", "nimmst", "nimmt", "nehmet",
    "kaufe", "kaufst", "kauft", "kaufet", "zahle", "zahlst", "zahlt", "zahlet",
    "bringe", "bringst", "bringt", "bringen", "bringet", "warte", "wartest",
    "wartet", "warten", "pizza", "cola", "rechnung", "bestellen", "vielen", "dank"
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
