"""Async generators for Goethe A2 Exam Teile (Lesen & Schreiben) with diverse topic rotation.
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import random
import re
from typing import Any, Dict, List, Tuple

from ..prompts import load_prompt
from ..runner import generate_exam
from .models import (
    LesenTeil1,
    LesenTeil2,
    LesenTeil3,
    LesenTeil4,
    SchreibenTeil1,
    SchreibenTeil2,
)

logger = logging.getLogger("lang_learn.exam.generators")

# Ring buffer keeping track of recently selected pool items to prevent consecutive duplicates
_recent_pool_history: Dict[str, collections.deque] = {
    "lesen_t1": collections.deque(maxlen=4),
    "lesen_t2": collections.deque(maxlen=4),
    "lesen_t3": collections.deque(maxlen=4),
    "lesen_t4": collections.deque(maxlen=3),
    "schreiben_t1": collections.deque(maxlen=4),
    "schreiben_t2": collections.deque(maxlen=4),
}


def _pick_distinct_pool_item(pool: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """Select a randomized item from the pool ensuring no recent repeats."""
    history = _recent_pool_history.setdefault(key, collections.deque(maxlen=max(1, len(pool) - 1)))
    available_indices = [i for i in range(len(pool)) if i not in history]
    if not available_indices:
        history.clear()
        available_indices = list(range(len(pool)))
    choice_idx = random.choice(available_indices)
    history.append(choice_idx)
    return pool[choice_idx]


def _repair_json_string(text: str) -> str:
    """Repair common JSON formatting errors from small LLMs.

    Fixes:
    1. Missing opening quotes: "c": Ein neues Geschäft" -> "c": "Ein neues Geschäft"
    2. Trailing commas before } or ]
    3. Single quotes used instead of double quotes (in key positions)
    """
    # Fix missing opening quote after colon: "key": value" -> "key": "value"
    # Pattern: colon, optional whitespace, then a non-quote char followed by content ending with quote
    text = re.sub(
        r':\s*(?!")([A-ZÄÖÜa-zäöüß][^",}\]]*?")',
        r': "\1',
        text
    )
    # Fix trailing commas
    text = re.sub(r',\s*([\}\]])', r'\1', text)
    return text


def _try_parse_json(text: str) -> Dict[str, Any] | None:
    """Attempt to parse text as JSON, returning None on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_json(text: str) -> Dict[str, Any] | None:
    """Robustly extract and parse JSON from LLM generation output.

    Handles:
    - Markdown code fences (```json ... ```)
    - Missing opening quotes on string values (Gemma GGUF quirk)
    - Truncated output (max_tokens exhaustion) by closing brackets
    - Trailing commas
    """
    if not text or not text.strip():
        return None

    cleaned = text.strip()
    # Remove markdown code fences
    cleaned = re.sub(r"^```(?:json|JSON)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # 1. Try direct parse
    result = _try_parse_json(cleaned)
    if result:
        return result

    # 2. Try after repairing quotes and commas
    repaired = _repair_json_string(cleaned)
    result = _try_parse_json(repaired)
    if result:
        logger.debug("JSON parsed after quote repair")
        return result

    # 3. Extract largest JSON object substring and repair
    json_match = re.search(r"(\{.*\})", repaired, re.DOTALL)
    if json_match:
        candidate = json_match.group(1).strip()
        result = _try_parse_json(candidate)
        if result:
            return result

    # 4. Handle truncated output: find the JSON start, repair, and close brackets
    start_idx = repaired.find("{")
    if start_idx != -1:
        truncated = repaired[start_idx:]

        # Try progressively removing trailing lines until we get parseable JSON
        lines = truncated.split('\n')
        for trim_count in range(min(len(lines), 12)):
            candidate_lines = lines[:len(lines) - trim_count] if trim_count > 0 else lines
            candidate = '\n'.join(candidate_lines)

            # Clean trailing partial content
            candidate = re.sub(r',\s*$', '', candidate.rstrip())

            # Count unclosed brackets and close them
            open_braces = candidate.count('{') - candidate.count('}')
            open_brackets = candidate.count('[') - candidate.count(']')
            suffix = ']' * max(0, open_brackets) + '}' * max(0, open_braces)

            if suffix or trim_count > 0:
                attempt = candidate + suffix
                result = _try_parse_json(attempt)
                if result:
                    logger.info("JSON recovered from truncated output (trimmed %d lines, closed %d brackets)", trim_count, len(suffix))
                    return result

                # Also try with quote repair applied
                attempt_repaired = _repair_json_string(attempt)
                result = _try_parse_json(attempt_repaired)
                if result:
                    logger.info("JSON recovered from truncated output after repair (trimmed %d lines)", trim_count)
                    return result

    logger.warning("All JSON extraction attempts failed. Output length: %d chars", len(text))
    return None


def _normalize_option_letter(val: Any, valid_set: tuple = ("a", "b", "c")) -> str:
    """Extract and normalize a single option letter from LLM answer_key string."""
    if val is None:
        return valid_set[0]

    s = str(val).strip().lower()
    if s in valid_set:
        return s

    # Handle 'keine', 'none', 'nein', '-' for Teil 4
    if "x" in valid_set and any(k in s for k in ("kein", "nein", "none", "nicht", "x")):
        return "x"

    # Match single character options like 'a)', '(a)', 'a.', 'option a', 'anzeige a'
    m = re.search(r"\b([a-z])\b", s)
    if m and m.group(1) in valid_set:
        return m.group(1)

    # First character if valid
    if s and s[0] in valid_set:
        return s[0]

    return valid_set[0]


def _extract_item_answer(item: Dict[str, Any], default: str = "a", valid_options: tuple = ("a", "b", "c")) -> str:
    """Extract answer key from item regardless of field naming variation."""
    raw = (
        item.get("answer_key")
        or item.get("answer")
        or item.get("correct_answer")
        or item.get("solution")
        or item.get("correct_option")
        or item.get("correct")
        or default
    )
    return _normalize_option_letter(raw, valid_set=valid_options)


def _normalize_options_dict(raw_opts: Any) -> Dict[str, str]:
    """Normalize options to lowercase keys {'a': '...', 'b': '...', 'c': '...'}."""
    if isinstance(raw_opts, list):
        return {chr(ord('a') + i): str(opt) for i, opt in enumerate(raw_opts[:3])}
    elif isinstance(raw_opts, dict):
        normalized = {}
        for k, v in raw_opts.items():
            norm_key = _normalize_option_letter(k, valid_set=("a", "b", "c"))
            normalized[norm_key] = str(v)
        # Ensure at least a, b, c exist
        for k in ("a", "b", "c"):
            if k not in normalized:
                normalized[k] = f"Option {k.upper()}"
        return normalized
    return {"a": "Option A", "b": "Option B", "c": "Option C"}


# ---------------------------------------------------------------------------
# Dynamic Goethe A2 Topic Themes for LLM Injections
# ---------------------------------------------------------------------------
THEMES_TEIL1 = [
    "Fahrradfahren und neue Radwege in deutschen Städten",
    "Frisches Kochen und gesunde Ernährung im Alltag",
    "Wochenendausflüge und Wandern in den Alpen",
    "Haustiere und Hundewiesen im Stadtleben",
    "Ein Wochenende ohne Smartphone und digitale Medien",
    "Sprachen lernen im Sprachcafé und Sprachtandem",
    "Flohmärkte und Second-Hand-Mode bei jungen Leuten",
    "Umweltfreundlich leben und Müll vermeiden im Alltag"
]

THEMES_TEIL2 = [
    "Großes Einkaufszentrum 'Stadt-Galerie' (4 Etagen)",
    "Modernes Kaufhaus am Hauptbahnhof (4 Stockwerke)",
    "Bürgeramt und Dienstleistungszentrum der Stadt",
    "Hauptbibliothek und Kulturzentrum am Marktplatz"
]

THEMES_TEIL3 = [
    "Einladung zu einer Geburtstagsfeier am Samstagabend",
    "Absage und Terminverschiebung für ein Treffen",
    "Urlaubsgrüße und Reisebericht von der Ostsee",
    "Neue Wohnung renovieren und Umzugshelfer gesucht",
    "Gemeinsamer Ausflug ins Museum am Wochenende"
]


# ---------------------------------------------------------------------------
# Generator Functions (Dispatched concurrently with dynamic theme injection)
# ---------------------------------------------------------------------------

async def generate_lesen_teil1(level: str = "A2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Lesen Teil 1 (Newspaper Article, 180-220 words). Returns (sanitized_teil, answer_key)."""
    selected_theme = random.choice(THEMES_TEIL1)
    fallback_choice = _pick_distinct_pool_item(POOL_LESEN_TEIL1, "lesen_t1")
    source = "fallback"
    try:
        template = load_prompt("exam_lesen_teil1.txt")
        prompt_with_theme = f"{template}\n\nTopic: {selected_theme}"
        raw = await asyncio.wait_for(
            asyncio.to_thread(generate_exam, prompt_with_theme, max_tokens=1024, temperature=0.75),
            timeout=120.0
        )
        logger.info("Lesen Teil 1 raw output: %d chars", len(raw))
        parsed = _extract_json(raw)
        text_word_count = len(parsed.get("text", "").split()) if parsed else 0
        if parsed and "text" in parsed and "items" in parsed and len(parsed["items"]) >= 3 and text_word_count >= 100:
            data = parsed
            source = "llm"
            if len(data["items"]) == 4:
                data["items"].append(fallback_choice["items"][4])
            else:
                data["items"] = data["items"][:5]
        else:
            logger.info("Lesen Teil 1 LLM rejected: text=%d words (need 100+), items=%d", text_word_count, len(parsed.get("items", [])) if parsed else 0)
            data = fallback_choice
    except Exception as e:
        logger.warning("Lesen Teil 1 using fallback: %s", e)
        data = fallback_choice

    logger.info("Lesen Teil 1 source: %s (theme: %s)", source, selected_theme)

    sanitized_items = []
    answer_key = {}
    explanations = {}

    for idx, item in enumerate(data["items"][:5], start=1):
        q_id = idx  # Canonical: 1..5
        ans = _extract_item_answer(item, default="a", valid_options=("a", "b", "c"))
        exp = str(item.get("explanation") or item.get("reason") or "Richtige Antwort laut Text.")
        opts = _normalize_options_dict(item.get("options", {}))

        answer_key[str(q_id)] = ans
        explanations[str(q_id)] = exp
        sanitized_items.append({
            "id": q_id,
            "question": str(item.get("question", "")),
            "options": opts
        })

    sanitized = {
        "teil": 1,
        "title": data.get("title", "Lesen Teil 1: Zeitungsartikel"),
        "text": data["text"],
        "items": sanitized_items,
        "source": source
    }
    return sanitized, {"answer_key": answer_key, "explanations": explanations}


async def generate_lesen_teil2(level: str = "A2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Lesen Teil 2 (Kaufhaus Info Board, 5-6 floors). Returns (sanitized_teil, answer_key)."""
    selected_theme = random.choice(THEMES_TEIL2)
    fallback_choice = _pick_distinct_pool_item(POOL_LESEN_TEIL2, "lesen_t2")
    source = "fallback"
    try:
        template = load_prompt("exam_lesen_teil2.txt")
        prompt_with_theme = f"{template}\n\nVenue: {selected_theme}"
        raw = await asyncio.wait_for(
            asyncio.to_thread(generate_exam, prompt_with_theme, max_tokens=1024, temperature=0.75),
            timeout=120.0
        )
        logger.info("Lesen Teil 2 raw output: %d chars", len(raw))
        parsed = _extract_json(raw)
        dir_count = len(parsed.get("directory", [])) if parsed else 0
        if parsed and "directory" in parsed and "items" in parsed and len(parsed["items"]) >= 3 and dir_count >= 4:
            data = parsed
            source = "llm"
            if len(data["items"]) == 4:
                data["items"].append(fallback_choice["items"][4])
            else:
                data["items"] = data["items"][:5]
        else:
            logger.info("Lesen Teil 2 LLM rejected: floors=%d (need 4+), items=%d", dir_count, len(parsed.get("items", [])) if parsed else 0)
            data = fallback_choice
    except Exception as e:
        logger.warning("Lesen Teil 2 using fallback: %s", e)
        data = fallback_choice

    logger.info("Lesen Teil 2 source: %s (venue: %s)", source, selected_theme)

    sanitized_items = []
    answer_key = {}
    explanations = {}

    for idx, item in enumerate(data["items"][:5], start=6):
        q_id = idx  # Canonical: 6..10
        ans = _extract_item_answer(item, default="a", valid_options=("a", "b", "c"))
        exp = str(item.get("explanation") or item.get("reason") or "Richtige Etage laut Wegweiser.")
        opts = _normalize_options_dict(item.get("options", {}))

        answer_key[str(q_id)] = ans
        explanations[str(q_id)] = exp
        sanitized_items.append({
            "id": q_id,
            "question": str(item.get("question", "")),
            "options": opts
        })

    sanitized = {
        "teil": 2,
        "title": data.get("title", "Lesen Teil 2: Kaufhaus-Wegweiser"),
        "directory": data["directory"],
        "items": sanitized_items,
        "source": source
    }
    return sanitized, {"answer_key": answer_key, "explanations": explanations}


async def generate_lesen_teil3(level: str = "A2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Lesen Teil 3 (Personal Email, 200-250 words). Returns (sanitized_teil, answer_key)."""
    selected_theme = random.choice(THEMES_TEIL3)
    fallback_choice = _pick_distinct_pool_item(POOL_LESEN_TEIL3, "lesen_t3")
    source = "fallback"
    try:
        template = load_prompt("exam_lesen_teil3.txt")
        prompt_with_theme = f"{template}\n\nContext: {selected_theme}"
        raw = await asyncio.wait_for(
            asyncio.to_thread(generate_exam, prompt_with_theme, max_tokens=1024, temperature=0.75),
            timeout=120.0
        )
        logger.info("Lesen Teil 3 raw output: %d chars", len(raw))
        parsed = _extract_json(raw)
        text_word_count = len(parsed.get("text", "").split()) if parsed else 0
        if parsed and "text" in parsed and "items" in parsed and len(parsed["items"]) >= 3 and text_word_count >= 100:
            data = parsed
            source = "llm"
            if len(data["items"]) == 4:
                data["items"].append(fallback_choice["items"][4])
            else:
                data["items"] = data["items"][:5]
        else:
            logger.info("Lesen Teil 3 LLM rejected: text=%d words (need 100+), items=%d", text_word_count, len(parsed.get("items", [])) if parsed else 0)
            data = fallback_choice
    except Exception as e:
        logger.warning("Lesen Teil 3 using fallback: %s", e)
        data = fallback_choice

    logger.info("Lesen Teil 3 source: %s (context: %s)", source, selected_theme)

    sanitized_items = []
    answer_key = {}
    explanations = {}

    for idx, item in enumerate(data["items"][:5], start=11):
        q_id = idx  # Canonical: 11..15
        ans = _extract_item_answer(item, default="a", valid_options=("a", "b", "c"))
        exp = str(item.get("explanation") or item.get("reason") or "Richtige Information laut E-Mail.")
        opts = _normalize_options_dict(item.get("options", {}))

        answer_key[str(q_id)] = ans
        explanations[str(q_id)] = exp
        sanitized_items.append({
            "id": q_id,
            "question": str(item.get("question", "")),
            "options": opts
        })

    sanitized = {
        "teil": 3,
        "title": data.get("title", "Lesen Teil 3: E-Mail / Brief"),
        "sender": data.get("sender", "Anna"),
        "recipient": data.get("recipient", "Freund/Freundin"),
        "subject": data.get("subject", "Neuigkeiten"),
        "text": data["text"],
        "items": sanitized_items,
        "source": source
    }
    return sanitized, {"answer_key": answer_key, "explanations": explanations}


async def generate_lesen_teil4(level: str = "A2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Lesen Teil 4 (6 Detailed Website Ads + 5 People Matching). Returns (sanitized_teil, answer_key)."""
    fallback_choice = _pick_distinct_pool_item(POOL_LESEN_TEIL4, "lesen_t4")
    source = "fallback"
    try:
        template = load_prompt("exam_lesen_teil4.txt")
        raw = await asyncio.wait_for(
            asyncio.to_thread(generate_exam, template, max_tokens=1200, temperature=0.75),
            timeout=120.0
        )
        logger.info("Lesen Teil 4 raw output: %d chars", len(raw))
        parsed = _extract_json(raw)
        ads_count = len(parsed.get("ads", [])) if parsed else 0
        # Check ad descriptions are substantial (not just 1-line stubs)
        ads_quality_ok = ads_count >= 5 and all(len(str(ad.get("text", ""))) >= 20 for ad in parsed.get("ads", [])[:5]) if parsed else False
        if parsed and ads_quality_ok and "items" in parsed and len(parsed["items"]) >= 3:
            data = parsed
            source = "llm"
            if len(data["items"]) == 4:
                data["items"].append(fallback_choice["items"][4])
            else:
                data["items"] = data["items"][:5]
            if len(data["ads"]) < 6:
                data["ads"] = fallback_choice["ads"]
        else:
            logger.info("Lesen Teil 4 LLM rejected: ads=%d, quality_ok=%s, items=%d", ads_count, ads_quality_ok, len(parsed.get("items", [])) if parsed else 0)
            data = fallback_choice
    except Exception as e:
        logger.warning("Lesen Teil 4 using fallback: %s", e)
        data = fallback_choice

    logger.info("Lesen Teil 4 source: %s", source)

    sanitized_items = []
    answer_key = {}
    explanations = {}

    for idx, item in enumerate(data["items"][:5], start=16):
        q_id = idx  # Canonical: 16..20
        ans = _extract_item_answer(item, default="x", valid_options=("a", "b", "c", "d", "e", "f", "x"))
        exp = str(item.get("explanation") or item.get("reason") or "Passende Anzeige zu den Bedürfnissen der Person.")
        person_text = str(item.get("person") or item.get("question") or item.get("text") or "")

        answer_key[str(q_id)] = ans
        explanations[str(q_id)] = exp
        sanitized_items.append({
            "id": q_id,
            "person": person_text
        })

    sanitized = {
        "teil": 4,
        "title": data.get("title", "Lesen Teil 4: Anzeigen und Personen"),
        "ads": data["ads"],
        "items": sanitized_items,
        "source": source
    }
    return sanitized, {"answer_key": answer_key, "explanations": explanations}


async def generate_schreiben_teil1(level: str = "A2") -> Dict[str, Any]:
    """Generate Schreiben Teil 1 (Informal SMS / Note)."""
    fallback_choice = _pick_distinct_pool_item(POOL_SCHREIBEN_TEIL1, "schreiben_t1")
    source = "fallback"
    try:
        template = load_prompt("exam_schreiben_teil1.txt")
        raw = await asyncio.wait_for(
            asyncio.to_thread(generate_exam, template, max_tokens=512, temperature=0.75),
            timeout=90.0
        )
        parsed = _extract_json(raw)
        if parsed and "scenario_german" in parsed and "bullet_points" in parsed and len(parsed["bullet_points"]) >= 2:
            logger.info("Schreiben Teil 1 source: llm")
            parsed["source"] = "llm"
            return parsed
        fallback_copy = dict(fallback_choice)
        fallback_copy["source"] = "fallback"
        return fallback_copy
    except Exception as e:
        logger.warning("Schreiben Teil 1 using fallback: %s", e)
        fallback_copy = dict(fallback_choice)
        fallback_copy["source"] = "fallback"
        return fallback_copy


async def generate_schreiben_teil2(level: str = "A2") -> Dict[str, Any]:
    """Generate Schreiben Teil 2 (Formal / Semi-formal Email)."""
    fallback_choice = _pick_distinct_pool_item(POOL_SCHREIBEN_TEIL2, "schreiben_t2")
    try:
        template = load_prompt("exam_schreiben_teil2.txt")
        raw = await asyncio.wait_for(
            asyncio.to_thread(generate_exam, template, max_tokens=512, temperature=0.75),
            timeout=90.0
        )
        parsed = _extract_json(raw)
        if parsed and "scenario_german" in parsed and "bullet_points" in parsed and len(parsed["bullet_points"]) >= 2:
            logger.info("Schreiben Teil 2 source: llm")
            parsed["source"] = "llm"
            return parsed
        fallback_copy = dict(fallback_choice)
        fallback_copy["source"] = "fallback"
        return fallback_copy
    except Exception as e:
        logger.warning("Schreiben Teil 2 using fallback: %s", e)
        fallback_copy = dict(fallback_choice)
        fallback_copy["source"] = "fallback"
        return fallback_copy

THEMES_TEIL2 = [
    "Großes Kaufhaus 'City-Center' mit 5 Etagen",
    "Einkaufszentrum 'Alster-Passage' in Hamburg",
    "Galerie am Hauptbahnhof mit Spezialgeschäften",
    "Shopping-Mall 'Rhein-Center' mit Gastronomie und Sportwelt"
]

THEMES_TEIL3 = [
    "Einladung zur Einweihungsparty in der neuen Wohnung",
    "Wochenend-Grillfest am See mit Freunden",
    "Geburtstagsüberraschung und Picknick im Stadtpark",
    "Gemeinsamer Ausflug und Städtetrip nach München"
]

# ---------------------------------------------------------------------------
# Rich Multi-Paper Fallback Pools (Full Goethe-Zertifikat A2 Certified Standard)
# ---------------------------------------------------------------------------

POOL_LESEN_TEIL1: List[Dict[str, Any]] = [
    {
        "title": "Der TV-Koch Stefan Berger: »Ich versuche immer wieder etwas Neues.«",
        "text": "Bei Stefan Berger gibt es Gerichte, von denen man vorher noch nie gehört hat. Er hat dauernd neue Ideen. Den Gästen gefällt das. Man muss unbedingt vorher anrufen und einen der wenigen Tische bestellen, wenn man in seinem Restaurant „Bremer Lokal“ essen möchte. Er hat viele Gäste, will aber kein zweites Lokal aufmachen. „Klar, ich könnte vielleicht reich damit werden, aber ich habe mich bewusst dagegen entschieden. Ich mag es einfach, wie wir hier arbeiten.“\n\nStefan Berger wurde 1968 im Rheinland geboren, war auf der Realschule und lernte dann in einem großen Hotel kochen.\n\nNach der Berufsausbildung brauchte er erstmal eine zwei-jährige Pause. Er fuhr durch die Welt, hatte verschiedene Jobs und lernte viel Neues kennen. Wegen einer Frau kam er dann nach Bremen. Das „Bremer Lokal“ in seiner Nachbarschaft suchte einen Koch, Berger nahm die Stelle an, und drei Jahre später kaufte er das Restaurant.\n\nDie meisten kennen ihn aber erst durch seine Fernsehshow „Berger kocht“. In der beliebten Sendung besuchen ihn Sänger und Schauspieler und kochen mit ihm ihre Lieblingsrezepte.",
        "items": [
            {
                "id": 1,
                "question": "Im „Bremer Lokal“...",
                "options": {
                    "a": "gibt es traditionelle Gerichte.",
                    "b": "muss man vorher reservieren.",
                    "c": "kann man den Koch selten sehen."
                },
                "answer_key": "b",
                "explanation": "Im Text steht: 'Man muss unbedingt vorher anrufen und einen der wenigen Tische bestellen'."
            },
            {
                "id": 2,
                "question": "Stefan Berger will kein zweites Restaurant eröffnen, weil...",
                "options": {
                    "a": "er nicht genug Geld dafür hat.",
                    "b": "ihm seine jetzige Arbeit gefällt.",
                    "c": "er keine guten Köche findet."
                },
                "answer_key": "b",
                "explanation": "Berger sagt: 'Ich mag es einfach, wie wir hier arbeiten'."
            },
            {
                "id": 3,
                "question": "Nach seiner Ausbildung...",
                "options": {
                    "a": "arbeitete er sofort in Bremen.",
                    "b": "eröffnete er eine Kochschule.",
                    "c": "reiste er durch die Welt."
                },
                "answer_key": "c",
                "explanation": "Laut Text fuhr er zwei Jahre lang durch die Welt und hatte verschiedene Jobs."
            },
            {
                "id": 4,
                "question": "Stefan Berger kam nach Bremen,...",
                "options": {
                    "a": "wegen einer Frau.",
                    "b": "um ein Restaurant zu kaufen.",
                    "c": "weil er ein Hotel eröffnen wollte."
                },
                "answer_key": "a",
                "explanation": "Im Text steht ausdrücklich: 'Wegen einer Frau kam er dann nach Bremen'."
            },
            {
                "id": 5,
                "question": "In seiner Fernsehsendung...",
                "options": {
                    "a": "kochen prominente Gäste mit ihm.",
                    "b": "zeigt er nur schnelle Snacks.",
                    "c": "besucht er Restaurants in ganz Europa."
                },
                "answer_key": "a",
                "explanation": "In der Sendung besuchen ihn bekannte Sänger und Schauspieler und kochen mit ihm."
            }
        ]
    },
    {
        "title": "Mobilität im Wandel: Immer mehr Menschen fahren mit dem Fahrrad zur Arbeit",
        "text": "In vielen deutschen Städten nutzen immer mehr Beschäftigte das Fahrrad für den täglichen Arbeitsweg. Eine aktuelle Studie des Verkehrsministeriums zeigt, dass mittlerweile fast ein Viertel der Berufstätigen regelmäßig mit dem Rad fährt. Vor allem in Ballungsräumen wie Berlin, München und Hamburg verzichten viele Menschen bewusst auf das eigene Auto.\n\nDie Gründe dafür sind vielfältig: Radfahren ist nicht nur gesund und umweltfreundlich, sondern spart auch viel Geld für teures Benzin, Tickets und Parkgebühren. Zudem entfällt das tägliche Ärgernis über Staus im morgendlichen Berufsverkehr.\n\nViele Städte reagieren auf diesen Trend und investieren kräftig in moderne Infrastruktur. Breite Fahrradschnellwege und getrennte Radspuren sorgen für mehr Sicherheit im Straßenverkehr.\n\nDennoch sehen Verkehrsexperten noch großen Verbesserungsbedarf. Vor allem an Bahnhöfen und großen Bürokomplexen fehlen oft überdachte und diebstahlsichere Abstellplätze sowie Ladestationen für moderne E-Bikes.",
        "items": [
            {
                "id": 1,
                "question": "Wie viele Berufstätige fahren laut der Studie regelmäßig mit dem Rad?",
                "options": {
                    "a": "Fast ein Viertel",
                    "b": "Mehr als die Hälfte",
                    "c": "Nur sehr wenige"
                },
                "answer_key": "a",
                "explanation": "Laut Text nutzen 'mittlerweile fast ein Viertel der Berufstätigen regelmäßig das Rad'."
            },
            {
                "id": 2,
                "question": "Warum verzichten viele Beschäftigte auf das Auto?",
                "options": {
                    "a": "Weil Autos in Städten verboten sind",
                    "b": "Weil Radfahren Geld spart und Staus vermeidet",
                    "c": "Weil Benzin überall kostenlos ist"
                },
                "answer_key": "b",
                "explanation": "Radfahren spart Geld für Benzin und Parkplätze und vermeidet Staus."
            },
            {
                "id": 3,
                "question": "Was machen viele Städte, um Radfahrer zu unterstützen?",
                "options": {
                    "a": "Sie verschenken neue Fahrräder",
                    "b": "Sie bauen breite Fahrradschnellwege",
                    "c": "Sie sperren alle Fußgängerzonen"
                },
                "answer_key": "b",
                "explanation": "Viele Städte investieren in breite Fahrradschnellwege und getrennte Radspuren."
            },
            {
                "id": 4,
                "question": "Welches Problem sehen Verkehrsexperten weiterhin?",
                "options": {
                    "a": "Es gibt zu wenige sichere Abstellplätze",
                    "b": "Fahrräder sind zu schnell",
                    "c": "Niemand möchte E-Bikes fahren"
                },
                "answer_key": "a",
                "explanation": "Experten betonen, dass diebstahlsichere Abstellplätze an Bahnhöfen fehlen."
            },
            {
                "id": 5,
                "question": "In welchen Städten ist der Trend laut Text besonders stark?",
                "options": {
                    "a": "Nur in kleinen Dörfern",
                    "b": "In Großstädten wie Berlin, München und Hamburg",
                    "c": "Ausschließlich an der Nordsee"
                },
                "answer_key": "b",
                "explanation": "Im Text werden Ballungsräume wie Berlin, München und Hamburg explizit genannt."
            }
        ]
    },
    {
        "title": "Frische Küche im Alltag: Warum Selberkochen bei jungen Erwachsenen boomt",
        "text": "Immer mehr junge Erwachsene in Deutschland entdecken das Kochen als entspannendes Hobby neu. Statt nach einem langen Arbeitstag schnell zu ungesunden Fertiggerichten oder Lieferdiensten zu greifen, stellen sich viele gerne selbst an den Herd.\n\nEine repräsentative Umfrage ergab, dass über 60 Prozent der 20- bis 35-Jährigen mindestens viermal pro Woche frisch kochen. Besonders beliebt ist der Einkauf auf regionalen Wochenmärkten, wo saisonales Gemüse und frische Kräuter direkt vom Bauern angeboten werden.\n\nBesonders im Trend liegen einfache, mediterrane Gerichte wie Pasta mit hausgemachter Tomatensoße, bunte Ofengemüse-Bleche oder frische Salate mit Nüssen und Ziegenkäse. Die Zubereitung dauert oft nicht länger als 30 Minuten.\n\nDarüber hinaus hat das Kochen auch eine starke soziale Komponente: Viele Freunde treffen sich am Wochenende zum gemeinsamen Kochen und tauschen ihre Lieblingsrezepte und Fotos auf Social Media aus.",
        "items": [
            {
                "id": 1,
                "question": "Was machen viele junge Erwachsene laut dem Text nach der Arbeit?",
                "options": {
                    "a": "Sie gehen jeden Tag ins Luxusrestaurant",
                    "b": "Sie kochen gerne selbst frische Gerichte",
                    "c": "Sie essen ausschließlich Tiefkühlkost"
                },
                "answer_key": "b",
                "explanation": "Im Text steht: 'stellen sich viele gerne selbst an den Herd'."
            },
            {
                "id": 2,
                "question": "Wo kaufen viele junge Leute am liebsten ihre Zutaten ein?",
                "options": {
                    "a": "Auf regionalen Wochenmärkten",
                    "b": "An der nächsten Autobahn-Tankstelle",
                    "c": "In reinen Online-Kaufhäusern"
                },
                "answer_key": "a",
                "explanation": "Laut Text ist der Einkauf auf regionalen Wochenmärkten besonders beliebt."
            },
            {
                "id": 3,
                "question": "Welche Gerichte werden laut Text besonders geschätzt?",
                "options": {
                    "a": "Sehr komplizierte 5-Gänge-Menüs",
                    "b": "Einfache und schnelle mediterrane Speisen",
                    "c": "Nur Fleischgerichte ohne Gemüse"
                },
                "answer_key": "b",
                "explanation": "Im Trend liegen einfache Gerichte wie Pasta und Ofengemüse, die unter 30 Minuten dauern."
            },
            {
                "id": 4,
                "question": "Was machen viele Freunde am Wochenende?",
                "options": {
                    "a": "Sie kochen gemeinsam und tauschen Rezepte",
                    "b": "Sie besuchen Kochschulen im Ausland",
                    "c": "Sie schreiben professionelle Kochbücher"
                },
                "answer_key": "a",
                "explanation": "Freunde treffen sich zum gemeinsamen Kochen und tauschen Rezepte aus."
            },
            {
                "id": 5,
                "question": "Wie viel Prozent der 20- bis 35-Jährigen kochen mehrmals wöchentlich frisch?",
                "options": {
                    "a": "Unter 10 Prozent",
                    "b": "Genau die Hälfte",
                    "c": "Über 60 Prozent"
                },
                "answer_key": "c",
                "explanation": "Laut Umfrage kochen 'über 60 Prozent' mindestens viermal pro Woche frisch."
            }
        ]
    }
]

POOL_LESEN_TEIL2: List[Dict[str, Any]] = [
    {
        "title": "Kaufhaus Alexa Wegweiser",
        "directory": [
            {"floor": "4. Stock", "departments": "Bücher, Geschenke, Spielsachen, Freizeittaschen, Koffer, Brieftaschen und Geldbeutel, Café, Friseur- und Nagelstudio, Kunden-WC, Telefon"},
            {"floor": "3. Stock", "departments": "Handys, Telefone, MP3-Player, CD-Player, DVD-Player, Radios, Fernseher, Computer, Notebooks, Tablets, Software, Drucker, CDs, DVDs, Videospiele, Sportkleidung, Arbeitskleidung"},
            {"floor": "2. Stock", "departments": "Herrenmode, Nachtwäsche für ihn, Unterwäsche für ihn, Möbel für Wohnzimmer, Bad und Küche, Teppiche, Lampen, Gardinen, Kissen, Decken, Stoffe und Dekoartikel, Handtücher"},
            {"floor": "1. Stock", "departments": "Damenmode, Nachtwäsche für sie, Unterwäsche für sie, Mode für Kinder und Jugendliche, Babybekleidung, Kinderwagen, Schuhe, Geschirr und Gläser, Besteck, Töpfe und Pfannen, Grills"},
            {"floor": "Erdgeschoss (EG)", "departments": "Information, Uhren, Schmuck, Parfüm, Kosmetik, Schreibwaren, Glückwunschkarten, Kalender, Schultaschen, Reiseführer, Souvenirs, Schuhwerkstatt, Schlüsseldienst, Blumenladen"},
            {"floor": "Untergeschoss (UG)", "departments": "Bäcker, Supermarkt, Putz- und Waschmittel, Fotoservice, Tabak, Zeitschriften und Zeitungen, Theater- und Konzertkarten, Reisebüro, Geldautomat, Kunden-WC"}
        ],
        "items": [
            {
                "id": 6,
                "question": "Sie möchten eine neue Bratpfanne, Töpfe und schöne Weingläser für die Küche kaufen.",
                "options": {"a": "1. Stock", "b": "2. Stock", "c": "Anderes Stockwerk"},
                "answer_key": "a",
                "explanation": "Geschirr, Gläser, Besteck, Töpfe und Pfannen befinden sich im 1. Stock."
            },
            {
                "id": 7,
                "question": "Sie suchen ein neues Smartphone, ein Tablet und passende Videospiele.",
                "options": {"a": "3. Stock", "b": "Erdgeschoss (EG)", "c": "4. Stock"},
                "answer_key": "a",
                "explanation": "Handys, Tablets, Computer und Videospiele befinden sich im 3. Stock."
            },
            {
                "id": 8,
                "question": "Sie möchten vor Ihrer Reise einen großen Koffer und einen spannenden Roman kaufen.",
                "options": {"a": "1. Stock", "b": "4. Stock", "c": "Anderes Stockwerk"},
                "answer_key": "b",
                "explanation": "Bücher, Geschenke, Freizeittaschen und Koffer befinden sich im 4. Stock."
            },
            {
                "id": 9,
                "question": "Sie suchen einen neuen Esstisch, Küchenlampen und kuschelige Kissen für Ihre Wohnung.",
                "options": {"a": "2. Stock", "b": "1. Stock", "c": "Untergeschoss (UG)"},
                "answer_key": "a",
                "explanation": "Möbel für Wohnzimmer und Küche, Lampen und Kissen sind im 2. Stock."
            },
            {
                "id": 10,
                "question": "Sie möchten sich im Friseurstudio die Haare schneiden lassen und danach im Café entspannen.",
                "options": {"a": "Erdgeschoss (EG)", "b": "3. Stock", "c": "4. Stock"},
                "answer_key": "c",
                "explanation": "Das Café sowie das Friseur- und Nagelstudio befinden sich im 4. Stock."
            }
        ]
    },
    {
        "title": "Einkaufszentrum 'City-Galerie' Wegweiser",
        "directory": [
            {"floor": "Obergeschoss 3", "departments": "Kino 'Astor', Fitnessclub, Bowling-Center, Sky-Restaurant, Panorama-Café, Kunden-WC"},
            {"floor": "Obergeschoss 2", "departments": "Computer & Laptops, Apple Store, Spielekonsolen, Sportartikel, Outdoorbekleidung, Fahrräder & E-Scooter"},
            {"floor": "Obergeschoss 1", "departments": "Herren- und Damenmode, Schuhe, Lederjacken, Handtaschen, Koffer, Reisebüro, Spielwarengeschäft"},
            {"floor": "Erdgeschoss (EG)", "departments": "Information, Parfümerie, Juwelier & Uhren, Optiker & Sonnenbrillen, Buchhandlung, Blumenshop"},
            {"floor": "Untergeschoss (UG)", "departments": "Apotheke, Drogeriemarkt, Bio-Supermarkt, Bäckerei, Schlüsseldienst, Geldautomaten, Parkhaus"}
        ],
        "items": [
            {
                "id": 6,
                "question": "Sie brauchen dringend Kopfschmerztabletten und Sonnencreme vor dem Ausflug.",
                "options": {"a": "Untergeschoss (UG)", "b": "Obergeschoss 1", "c": "Anderes Stockwerk"},
                "answer_key": "a",
                "explanation": "Apotheke und Drogeriemarkt befinden sich im Untergeschoss (UG)."
            },
            {
                "id": 7,
                "question": "Sie möchten am Abend mit Freunden den neuesten Kinofilm anschauen.",
                "options": {"a": "Obergeschoss 3", "b": "Erdgeschoss (EG)", "c": "Obergeschoss 2"},
                "answer_key": "a",
                "explanation": "Das Kino 'Astor' befindet sich im Obergeschoss 3."
            },
            {
                "id": 8,
                "question": "Sie suchen ein neues Ladekabel für Ihr Notebook und eine Sportjacke.",
                "options": {"a": "Erdgeschoss (EG)", "b": "Obergeschoss 2", "c": "Obergeschoss 1"},
                "answer_key": "b",
                "explanation": "Computer, Apple Store und Sportartikel befinden sich im Obergeschoss 2."
            },
            {
                "id": 9,
                "question": "Sie möchten Ihrer Freundin eine neue Halskette oder Ohrringe schenken.",
                "options": {"a": "Obergeschoss 2", "b": "Erdgeschoss (EG)", "c": "Anderes Stockwerk"},
                "answer_key": "b",
                "explanation": "Juwelier & Uhren sowie Parfümerie befinden sich im Erdgeschoss (EG)."
            },
            {
                "id": 10,
                "question": "Sie suchen elegante Lederschuhe und einen Rollkoffer für Ihre Geschäftsreise.",
                "options": {"a": "Untergeschoss (UG)", "b": "Obergeschoss 1", "c": "Obergeschoss 3"},
                "answer_key": "b",
                "explanation": "Schuhe, Handtaschen und Koffer befinden sich im Obergeschoss 1."
            }
        ]
    }
]

POOL_LESEN_TEIL3: List[Dict[str, Any]] = [
    {
        "sender": "Gülcan",
        "recipient": "Sonja",
        "subject": "Mein neues Studentenleben in Hamburg",
        "text": "Liebe Sonja,\n\nich bin jetzt schon vier Wochen in Hamburg und bin noch dabei, mich hier einzuleben. An der Universität ist vieles ganz anders organisiert als zu Hause. Und auch im täglichen Leben musste ich erst einmal lernen, wie einige Dinge hier gemacht werden. Zum Beispiel, wie ich ein Zimmer finde und wo ich was einkaufen kann.\n\nIn der ersten Woche haben ein paar Studenten eine Willkommensführung für uns ausländische Studierende gemacht. Sie haben uns die Uni gezeigt: die Bibliothek, die Cafeteria und die Multimedia-Räume. Hamburg habe ich dann alleine mit dem Stadtplan kennengelernt.\n\nIch wohne mit drei anderen Studenten aus Italien, Japan und Mexiko zusammen. Immer freitags kocht einer von uns etwas aus seinem Land und wir essen zusammen, obwohl wir nur eine winzig kleine Küche haben! Ich finde das super, du weißt ja, wie gerne ich koche!\n\nWir sprechen in der Wohnung nicht nur Deutsch, sondern oft auch Englisch miteinander. Manchmal ist das einfacher, aber mich stört das ein bisschen. Ich möchte dieses Jahr möglichst viel Deutsch lernen. Und weißt du, was mir am meisten Spaß macht? Der Literaturkurs. Der Dozent, Herr Hahn, ist ein total witziger Typ. Den müsstest du mal erleben. :-)\n\nIch freue mich auf deinen Besuch im März. Dann zeige ich dir die Stadt und an einem Nachmittag fahren wir an die Ostsee. Da ist es total schön. Du kannst dann bei Mario schlafen. Das ist der Italiener, der neben mir wohnt. Er ist einverstanden, denn er fährt in den Ferien nach Hause, nach Genua.\n\nSchreib mir bald!\nBis dann\nGülcan",
        "items": [
            {
                "id": 11,
                "question": "Gülcan...",
                "options": {
                    "a": "wohnt erst seit einer Woche in Hamburg.",
                    "b": "lebt seit vier Wochen in Hamburg.",
                    "c": "studiert bereits seit einem Jahr an der Uni."
                },
                "answer_key": "b",
                "explanation": "Gülcan schreibt im ersten Satz: 'ich bin jetzt schon vier Wochen in Hamburg'."
            },
            {
                "id": 12,
                "question": "In ihrer ersten Woche in Hamburg...",
                "options": {
                    "a": "hat sie an einer Willkommensführung an der Uni teilgenommen.",
                    "b": "ist sie direkt an die Ostsee gefahren.",
                    "c": "hat sie mit Sonja eine Wohnung gesucht."
                },
                "answer_key": "a",
                "explanation": "Sie schreibt: 'In der ersten Woche haben ein paar Studenten eine Willkommensführung für uns ausländische Studierende gemacht'."
            },
            {
                "id": 13,
                "question": "Was gefällt Gülcan an ihrer WG besonders gut?",
                "options": {
                    "a": "Dass die Küche riesengroß und modern ist.",
                    "b": "Dass alle Mitbewohner nur Deutsch sprechen.",
                    "c": "Dass freitags gemeinsam internationale Gerichte gekocht werden."
                },
                "answer_key": "c",
                "explanation": "Sie schreibt: 'Immer freitags kocht einer von uns etwas aus seinem Land und wir essen zusammen ... Ich finde das super'."
            },
            {
                "id": 14,
                "question": "Was stört Gülcan im Zusammenleben ein bisschen?",
                "options": {
                    "a": "Dass in der Wohnung oft Englisch statt Deutsch gesprochen wird.",
                    "b": "Dass die Mitbewohner nie sauber machen.",
                    "c": "Dass Herr Hahn zu strenge Noten vergibt."
                },
                "answer_key": "a",
                "explanation": "Sie schreibt: 'Wir sprechen ... oft auch Englisch miteinander ... mich stört das ein bisschen'."
            },
            {
                "id": 15,
                "question": "Wenn Sonja im März zu Besuch kommt,...",
                "options": {
                    "a": "müssen sie ein teures Hotelzimmer buchen.",
                    "b": "kann Sonja im Zimmer des Italieners Mario schlafen.",
                    "c": "fahren sie sofort gemeinsam nach Genua."
                },
                "answer_key": "b",
                "explanation": "Gülcan schreibt: 'Du kannst dann bei Mario schlafen ... Er fährt in den Ferien nach Hause, nach Genua'."
            }
        ]
    },
    {
        "sender": "Anna Schneider",
        "recipient": "Markus",
        "subject": "Neuigkeiten aus meiner neuen Wohnung in Freiburg!",
        "text": "Lieber Markus,\n\nendlich habe ich etwas Zeit zum Schreiben. Der Umzug letzte Woche war ganz schön anstrengend, aber meine Freunde haben mir super geholfen. Alle Kisten sind inzwischen ausgepackt und die Möbel stehen an ihrem Platz.\n\nMeine neue Wohnung in Freiburg ist wirklich toll! Sie hat zwei helle Zimmer, einen kleinen Balkon und liegt direkt neben einem wunderschönen Stadtpark. Zur Universität brauche ich mit dem Fahrrad nur zehn Minuten.\n\nAm nächsten Samstag mache ich eine kleine Einweihungsparty ab 18 Uhr. Ich grille auf dem Balkon und es gibt leckere hausgemachte Salate und Getränke. Ein paar Studienkollegen und Nachbarn kommen auch vorbei.\n\nHast du Zeit und Lust zu kommen? Du kannst auch gerne bei mir auf dem gemütlichen Schlafsofa im Wohnzimmer übernachten, wenn es spät wird. Sag mir bitte bis Donnerstag kurz Bescheid, damit ich genug einkaufen kann!\n\nHerzliche Grüße,\nAnna",
        "items": [
            {
                "id": 11,
                "question": "Wie beurteilt Anna ihren Umzug?",
                "options": {
                    "a": "Als sehr entspannt und mühelos",
                    "b": "Als ziemlich anstrengend, aber mit toller Hilfe",
                    "c": "Als viel zu teuer"
                },
                "answer_key": "b",
                "explanation": "Anna schreibt: 'Der Umzug letzte Woche war ganz schön anstrengend, aber meine Freunde haben mir super geholfen'."
            },
            {
                "id": 12,
                "question": "Was gefällt Anna an ihrer neuen Wohnung?",
                "options": {
                    "a": "Sie liegt direkt neben der Autobahn",
                    "b": "Sie hat zwei helle Zimmer, einen Balkon und liegt am Park",
                    "c": "Sie hat einen riesigen Swimmingpool"
                },
                "answer_key": "b",
                "explanation": "Laut Text hat die Wohnung zwei helle Zimmer, einen Balkon und liegt neben einem Stadtpark."
            },
            {
                "id": 13,
                "question": "Was plant Anna für kommenden Samstag?",
                "options": {
                    "a": "Eine Einweihungsparty mit Grillen auf dem Balkon",
                    "b": "Einen großen Ausflug ins Museum",
                    "c": "Eine Zugreise nach Berlin"
                },
                "answer_key": "a",
                "explanation": "Sie plant eine Einweihungsparty ab 18 Uhr mit Grillen auf dem Balkon."
            },
            {
                "id": 14,
                "question": "Wie lange braucht Anna mit dem Rad zur Uni?",
                "options": {
                    "a": "Über eine Stunde",
                    "b": "Nur zehn Minuten",
                    "c": "Sie muss den Bus nehmen"
                },
                "answer_key": "b",
                "explanation": "Im Text steht: 'Zur Universität brauche ich mit dem Fahrrad nur zehn Minuten'."
            },
            {
                "id": 15,
                "question": "Was bietet Anna Markus für die Übernachtung an?",
                "options": {
                    "a": "Ein Hotelzimmer in der Innenstadt",
                    "b": "Ein Schlafsofa im Wohnzimmer",
                    "c": "Ein Zelt im Park"
                },
                "answer_key": "b",
                "explanation": "Sie schreibt: 'Du kannst auch gerne bei mir auf dem gemütlichen Schlafsofa im Wohnzimmer übernachten'."
            }
        ]
    }
]

POOL_LESEN_TEIL4: List[Dict[str, Any]] = [
    {
        "title": "Lokale und Restaurants im Internet",
        "ads": [
            {
                "id": "a",
                "title": "www.park-cafe.de",
                "text": "Selbstgemachte Torten, Kuchen und italienisches Eis. Große Sonnenterrasse mit Spielplatz. Alles auch zum Mitnehmen! Täglich außer montags von 14 bis 19 Uhr geöffnet. Bergstraße 7, 89312 Günzburg, Tel. 08221 36152"
            },
            {
                "id": "b",
                "title": "www.feine-speisen.de",
                "text": "Egal, wo Sie feiern wollen, wir liefern für Ihre Hochzeit oder andere private Feiern bestes Essen. Z. B. Hochzeitsmenü ab 30 € p. P.; bayerisches Buffet 20,50 € p. P. Wir bieten außerdem Tische und Stühle, Dekoration, Servicepersonal und Kinderbetreuung an."
            },
            {
                "id": "c",
                "title": "www.weinhaus-walter.de",
                "text": "Internationale Spezialitäten. Beste Weine. Jetzt neu: Jeden Tag anderes 3-Gänge-Menü mit Getränk ab 20 € pro Person. Im Sommer auch in unserem ruhigen Garten. Sie finden uns direkt hinter dem Rathaus. Schöner Raum für kleine Feiern."
            },
            {
                "id": "d",
                "title": "www.cafe-sand.de",
                "text": "Urlaub in der Stadtmitte – direkt am Fluss, täglich ab 10.00 Uhr geöffnet. Jeden Samstag und Sonntag gibt es das stadtbekannte große Frühstück. Ab Mai jeden Sonnabend Party mit Live-Musik, ab 22 Uhr. Tischreservierung Tel. 785 43 65."
            },
            {
                "id": "e",
                "title": "www.towabu.de",
                "text": "Spiel + Spaß bei Towabu. Auf über 2500 qm auch bei schlechtem Wetter spielen und toben! Tolle Geburtstagspartys mit Super-Programm. Getränke inklusive. Täglich 10 bis 20 Uhr. Auch in den Sommerferien geöffnet."
            },
            {
                "id": "f",
                "title": "www.hansen-im-moor.de",
                "text": "Das Ausflugsrestaurant im Teufelsmoor. Mit dem Bus nur 20 Minuten vom Zentrum! Norddeutsche Küche. Mit Terrasse direkt am See. Sie suchen einen Ort für Ihr Familienfest, Ihre Hochzeit, Ihre Firmenfeier? Sprechen Sie uns an! Unsere Räume bieten Platz für 150 Personen."
            }
        ],
        "items": [
            {
                "id": 16,
                "person": "Sarah heiratet bald und möchte mit vielen Gästen in einem Lokal feiern.",
                "answer_key": "f",
                "explanation": "Anzeige F (Hansen im Moor) bietet große Räume für Hochzeiten mit bis zu 150 Personen direkt am See."
            },
            {
                "id": 17,
                "person": "Petra will mit Geschäftspartnern in der Stadt essen gehen und über die Arbeit sprechen.",
                "answer_key": "c",
                "explanation": "Anzeige C (Weinhaus Walter) bietet einen ruhigen Garten und Menüs direkt hinter dem Rathaus."
            },
            {
                "id": 18,
                "person": "Jens feiert seinen Geburtstag zu Hause und möchte guten Wein anbieten.",
                "answer_key": "x",
                "explanation": "Keine der Anzeigen bietet Weinlieferungen nach Hause an, daher ist die richtige Antwort 'x'."
            },
            {
                "id": 19,
                "person": "Karsten lädt am Abend Gäste zu sich nach Hause ein, möchte aber nicht kochen.",
                "answer_key": "b",
                "explanation": "Anzeige B (Feine Speisen) liefert Essen, Buffets und Geschirr direkt nach Hause für private Feiern."
            },
            {
                "id": 20,
                "person": "Gabriele und ihre Tochter feiern Kindergeburtstag und möchten Kuchen essen gehen.",
                "answer_key": "a",
                "explanation": "Anzeige A (Park-Café) bietet selbstgemachte Torten, Kuchen und einen Spielplatz für Kinder."
            }
        ]
    },
    {
        "title": "Anzeigen und Dienstleistungen in der Region",
        "ads": [
            {
                "id": "a",
                "title": "www.salsa-club-ritmo.de",
                "text": "Salsa & Bachata Tanzkurse für Einsteiger! Jeden Freitag ab 19:30 Uhr. Kein fester Partner nötig. Schnupperstunde nur 10 €. Leopoldstraße 45, Tel. 089 332211"
            },
            {
                "id": "b",
                "title": "www.mathe-hilfe-plus.de",
                "text": "Qualifizierte Mathematik- und Physik-Nachhilfe für Schüler bis zum Abitur. Einzelunterricht und Hausbesuche möglich. Tel. 0172 445566"
            },
            {
                "id": "c",
                "title": "www.camping-ausruestung.de",
                "text": "Zelte, Schlafsäcke und Outdoor-Kocher günstig mieten. Ideal für Festival- und Wochenendtrips an den See. Ab 15 €/Wochenende. info@outdoor-rent.de"
            },
            {
                "id": "d",
                "title": "www.fotostudio-blickfang.de",
                "text": "Fotoworkshops für Anfänger in der Natur. Lerne Landschaftsfotografie am Wochenende mit Profi-Ausrüstung. www.blickfang-foto.de"
            },
            {
                "id": "e",
                "title": "www.auto-schnaeppchen.de",
                "text": "Verkaufe sparsamen VW Polo, Baujahr 2019, 45.000 km, TÜV neu, Klimaanlage, 8-fach bereift. Preis: 6.200 € VB. Tel. 0151 998877"
            },
            {
                "id": "f",
                "title": "www.gartenprofi-service.de",
                "text": "Zuverlässige Gartenpflege: Rasenmähen, Heckenschnitt, Baumpflege und Unkrautbeseitigung. Termine kurzfristig frei. 18 €/Stunde. Tel. 0176 112233"
            }
        ],
        "items": [
            {
                "id": 16,
                "person": "Tim sucht ein großes Zelt für einen Wochenendausflug an den See mit seinen Freunden.",
                "answer_key": "c",
                "explanation": "Anzeige C (Camping-Ausrüstung) vermietet Zelte und Outdoor-Ausrüstung für Wochenenden."
            },
            {
                "id": 17,
                "person": "Elena möchte gerne tanzen lernen, hat aber im Moment keinen festen Tanzpartner.",
                "answer_key": "a",
                "explanation": "Anzeige A (Salsa Club) bietet Anfängerkurse an, für die kein fester Partner nötig ist."
            },
            {
                "id": 18,
                "person": "Familie Berger braucht Hilfe beim Rasenmähen und Heckenschnitt in ihrem großen Garten.",
                "answer_key": "f",
                "explanation": "Anzeige F (Gartenprofi) bietet zuverlässiges Rasenmähen und Heckenschnitt an."
            },
            {
                "id": 19,
                "person": "Simon hat eine neue Kamera und möchte lernen, wie man schöne Naturfotos macht.",
                "answer_key": "d",
                "explanation": "Anzeige D (Fotostudio Blickfang) bietet Naturfotografie-Workshops für Anfänger an."
            },
            {
                "id": 20,
                "person": "Laura sucht einen Reitkurs für Anfänger auf einem Reiterhof.",
                "answer_key": "x",
                "explanation": "Keine der Anzeigen bietet Reitkurse an, daher ist die Antwort 'x'."
            }
        ]
    }
]

POOL_SCHREIBEN_TEIL1: List[Dict[str, Any]] = [
    {
        "teil": 1,
        "title": "Schreiben Teil 1: Verspätung ankündigen",
        "scenario_german": "Sie haben sich heute Abend mit Ihrem Freund Michael im Kino verabredet, können aber nicht pünktlich sein.",
        "instructions_german": "Schreiben Sie eine kurze Nachricht an Michael (ca. 20–30 Wörter). Schreiben Sie zu allen drei Punkten:",
        "bullet_points": [
            "Entschuldigen Sie sich für die Verspätung.",
            "Nennen Sie den Grund (z.B. Zugausfall oder Überstunden).",
            "Schlagen Sie einen neuen Treffpunkt oder eine neue Uhrzeit vor."
        ],
        "target_word_count": "20–30 Wörter",
        "tips_english": "Write a short SMS/note (approx. 20-30 words). Address all 3 bullet points, using an informal greeting and sign-off."
    },
    {
        "teil": 1,
        "title": "Schreiben Teil 1: Einladung ablehnen und neues Treffen vorschlagen",
        "scenario_german": "Ihre Kollegin Maria hat Sie zum Abendessen am Freitag eingeladen. Sie haben aber leider keine Zeit.",
        "instructions_german": "Schreiben Sie eine kurze Nachricht an Maria (ca. 20–30 Wörter). Schreiben Sie zu allen drei Punkten:",
        "bullet_points": [
            "Bedanken Sie sich herzlich für die Einladung.",
            "Erklären Sie höflich, warum Sie am Freitag nicht kommen können.",
            "Schlagen Sie ein Treffen am nächsten Wochenende vor."
        ],
        "target_word_count": "20–30 Wörter",
        "tips_english": "Write a short note (approx. 20-30 words). Thank for the invite, give your reason, and propose an alternative date."
    },
    {
        "teil": 1,
        "title": "Schreiben Teil 1: Sporttraining absagen",
        "scenario_german": "Sie trainieren regelmäßig mit Ihrem Freund Lukas im Fitnessstudio, sind heute aber krank.",
        "instructions_german": "Schreiben Sie eine Nachricht an Lukas (ca. 20–30 Wörter). Schreiben Sie zu allen drei Punkten:",
        "bullet_points": [
            "Sagen Sie das gemeinsame Training für heute ab.",
            "Erklären Sie kurz Ihren Grund (z.B. Erkältung oder Kopfschmerzen).",
            "Vereinbaren Sie einen neuen Termin für die nächste Woche."
        ],
        "target_word_count": "20–30 Wörter",
        "tips_english": "Write an informal note (20-30 words) cancelling training, stating why, and rescheduling."
    }
]

POOL_SCHREIBEN_TEIL2: List[Dict[str, Any]] = [
    {
        "teil": 2,
        "title": "Schreiben Teil 2: Sprachkurs anfragen",
        "scenario_german": "Sie möchten im nächsten Monat an einer Sprachschule in Heidelberg einen Deutschkurs (Stufe B1) besuchen. Schreiben Sie an die Sprachschule.",
        "instructions_german": "Schreiben Sie eine formelle E-Mail an Frau Weber von der Sprachschule (ca. 30–40 Wörter). Schreiben Sie zu allen vier Punkten:",
        "bullet_points": [
            "Grund für Ihr Schreiben nennen",
            "Informationen zum Kurstermin und Beginn erfragen",
            "Nach den Gesamtkosten und Unterkünften fragen",
            "Passende formelle Anrede und Grußformel verwenden"
        ],
        "target_word_count": "30–40 Wörter",
        "tips_english": "Write a formal email (approx. 30-40 words). Address all points, use formal greetings (Sehr geehrte Frau Weber) and polite closings (Mit freundlichen Grüßen)."
    },
    {
        "teil": 2,
        "title": "Schreiben Teil 2: Zimmerreservierung im Hotel",
        "scenario_german": "Sie möchten für einen Wochenendurlaub mit Ihrer Familie zwei Zimmer im Hotel 'Alpenblick' buchen.",
        "instructions_german": "Schreiben Sie eine formelle E-Mail an das Hotel (ca. 30–40 Wörter). Schreiben Sie zu allen vier Punkten:",
        "bullet_points": [
            "Ankunftstag und Anzahl der Personen / Zimmer nennen",
            "Nach dem Frühstücksangebot und den Zimmerpreisen fragen",
            "Nach Parkplätzen direkt am Hotel fragen",
            "Höfliche formelle Anrede und Schlussformel"
        ],
        "target_word_count": "30–40 Wörter",
        "tips_english": "Write a formal email requesting hotel reservation, inquiring about prices, breakfast and parking."
    },
    {
        "teil": 2,
        "title": "Schreiben Teil 2: Wohnungsbesichtigung anfragen",
        "scenario_german": "Sie haben im Internet eine Anzeige für eine schöne 2-Zimmer-Wohnung gesehen und möchten die Wohnung gerne besichtigen.",
        "instructions_german": "Schreiben Sie eine E-Mail an den Vermieter, Herrn Müller (ca. 30–40 Wörter). Schreiben Sie zu allen vier Punkten:",
        "bullet_points": [
            "Sich kurz vorstellen (Beruf und Personenzahl)",
            "Großes Interesse an der Wohnung bekunden",
            "Nach einem Termin für eine Besichtigung fragen",
            "Passende formelle Grußformel verwenden"
        ],
        "target_word_count": "30–40 Wörter",
        "tips_english": "Write a formal apartment inquiry email to Herr Müller introducing yourself and requesting a viewing appointment."
    }
]



