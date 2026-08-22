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
from ..runner import generate
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


def _extract_json(text: str) -> Dict[str, Any] | None:
    """Safely extract and parse JSON from LLM generation output."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json|JSON)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt to extract largest JSON object substring
    json_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if json_match:
        candidate = json_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Repair common trailing comma issues
            repaired = re.sub(r",\s*([\}\]])", r"\1", candidate)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    # Fallback: if ends abruptly, try closing brackets
    start_idx = cleaned.find("{")
    if start_idx != -1:
        truncated = cleaned[start_idx:]
        for suffix in ["}]}", "]}", "}", '"]}', '"}']:
            try:
                fixed = re.sub(r",\s*$", "", truncated) + suffix
                fixed = re.sub(r",\s*([\}\]])", r"\1", fixed)
                return json.loads(fixed)
            except Exception:
                continue

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
    """Generate Lesen Teil 1 (Newspaper Article). Returns (sanitized_teil, answer_key)."""
    selected_theme = random.choice(THEMES_TEIL1)
    fallback_choice = _pick_distinct_pool_item(POOL_LESEN_TEIL1, "lesen_t1")
    source = "fallback"
    try:
        template = load_prompt("exam_lesen_teil1.txt")
        prompt_with_theme = f"{template}\n\nTopic: {selected_theme}"
        raw = await asyncio.wait_for(
            asyncio.to_thread(generate, prompt_with_theme, max_tokens=350, temperature=0.75),
            timeout=10.0
        )
        parsed = _extract_json(raw)
        if parsed and "text" in parsed and "items" in parsed and len(parsed["items"]) >= 4:
            data = parsed
            source = "llm"
            if len(data["items"]) == 4:
                data["items"].append(fallback_choice["items"][4])
            else:
                data["items"] = data["items"][:5]
        else:
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
    """Generate Lesen Teil 2 (Kaufhaus Info Board). Returns (sanitized_teil, answer_key)."""
    selected_theme = random.choice(THEMES_TEIL2)
    fallback_choice = _pick_distinct_pool_item(POOL_LESEN_TEIL2, "lesen_t2")
    source = "fallback"
    try:
        template = load_prompt("exam_lesen_teil2.txt")
        prompt_with_theme = f"{template}\n\nVenue: {selected_theme}"
        raw = await asyncio.wait_for(
            asyncio.to_thread(generate, prompt_with_theme, max_tokens=350, temperature=0.75),
            timeout=10.0
        )
        parsed = _extract_json(raw)
        if parsed and "directory" in parsed and "items" in parsed and len(parsed["items"]) >= 4:
            data = parsed
            source = "llm"
            if len(data["items"]) == 4:
                data["items"].append(fallback_choice["items"][4])
            else:
                data["items"] = data["items"][:5]
        else:
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
    """Generate Lesen Teil 3 (Personal Email). Returns (sanitized_teil, answer_key)."""
    selected_theme = random.choice(THEMES_TEIL3)
    fallback_choice = _pick_distinct_pool_item(POOL_LESEN_TEIL3, "lesen_t3")
    source = "fallback"
    try:
        template = load_prompt("exam_lesen_teil3.txt")
        prompt_with_theme = f"{template}\n\nContext: {selected_theme}"
        raw = await asyncio.wait_for(
            asyncio.to_thread(generate, prompt_with_theme, max_tokens=350, temperature=0.75),
            timeout=10.0
        )
        parsed = _extract_json(raw)
        if parsed and "text" in parsed and "items" in parsed and len(parsed["items"]) >= 4:
            data = parsed
            source = "llm"
            if len(data["items"]) == 4:
                data["items"].append(fallback_choice["items"][4])
            else:
                data["items"] = data["items"][:5]
        else:
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
    """Generate Lesen Teil 4 (6 Ads + 5 People Matching). Returns (sanitized_teil, answer_key)."""
    fallback_choice = _pick_distinct_pool_item(POOL_LESEN_TEIL4, "lesen_t4")
    source = "fallback"
    try:
        template = load_prompt("exam_lesen_teil4.txt")
        raw = await asyncio.wait_for(
            asyncio.to_thread(generate, template, max_tokens=400, temperature=0.75),
            timeout=10.0
        )
        parsed = _extract_json(raw)
        if parsed and "ads" in parsed and len(parsed["ads"]) >= 5 and "items" in parsed and len(parsed["items"]) >= 4:
            data = parsed
            source = "llm"
            if len(data["items"]) == 4:
                data["items"].append(fallback_choice["items"][4])
            else:
                data["items"] = data["items"][:5]
            if len(data["ads"]) < 6:
                data["ads"] = fallback_choice["ads"]
        else:
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
            asyncio.to_thread(generate, template, max_tokens=220, temperature=0.75),
            timeout=8.0
        )
        parsed = _extract_json(raw)
        if parsed and "scenario_german" in parsed and "bullet_points" in parsed and len(parsed["bullet_points"]) == 3:
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
            asyncio.to_thread(generate, template, max_tokens=250, temperature=0.75),
            timeout=8.0
        )
        parsed = _extract_json(raw)
        if parsed and "scenario_german" in parsed and "bullet_points" in parsed and len(parsed["bullet_points"]) >= 3:
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
# Rich Multi-Paper Fallback Pools (Ensures 100% variety even in offline mode)
# ---------------------------------------------------------------------------

POOL_LESEN_TEIL1: List[Dict[str, Any]] = [
    {
        "title": "Immer mehr Menschen fahren mit dem Fahrrad zur Arbeit",
        "text": "In deutschen Großstädten nutzen immer mehr Menschen das Fahrrad für den täglichen Weg zur Arbeit. Eine neue Studie zeigt, dass fast 25 Prozent der Beschäftigten regelmäßig mit dem Rad fahren. Viele Städte bauen deshalb neue, breite Radwege. Besonders junge Leute finden das Radfahren praktisch, gesund und umweltfreundlich. Sie sparen Geld für Benzin und Parkplätze und müssen nicht im Stau stehen. Allerdings fordern Experten mehr sichere Abstellplätze an Bahnhöfen und Bürogebäuden.",
        "items": [
            {
                "id": 1,
                "question": "Wie viele Beschäftigte fahren laut der Studie regelmäßig mit dem Rad?",
                "options": {"a": "Fast ein Viertel", "b": "Mehr als die Hälfte", "c": "Nur sehr wenige"},
                "answer_key": "a",
                "explanation": "Laut Text nutzen 'fast 25 Prozent' (also fast ein Viertel) das Rad."
            },
            {
                "id": 2,
                "question": "Warum bauen viele Städte neue Radwege?",
                "options": {"a": "Weil Autos verboten werden", "b": "Weil mehr Menschen Rad fahren", "c": "Weil Radwege billiger sind"},
                "answer_key": "b",
                "explanation": "Wegen der wachsenden Zahl an Radfahrern bauen viele Städte neue Radwege."
            },
            {
                "id": 3,
                "question": "Was finden junge Leute am Radfahren besonders gut?",
                "options": {"a": "Es ist teuer", "b": "Man muss viel reparieren", "c": "Es ist gesund und spart Geld"},
                "answer_key": "c",
                "explanation": "Sie finden es 'praktisch, gesund und umweltfreundlich' und sparen Geld."
            },
            {
                "id": 4,
                "question": "Was müssen Radfahrer nicht tun?",
                "options": {"a": "Im Stau stehen", "b": "Einen Helm tragen", "c": "Auf die Ampeln achten"},
                "answer_key": "a",
                "explanation": "Im Text steht: 'Sie ... müssen nicht im Stau stehen'."
            },
            {
                "id": 5,
                "question": "Was fordern Experten für die Zukunft?",
                "options": {"a": "Höhere Preise für Fahrräder", "b": "Mehr sichere Abstellplätze", "c": "Schnellere Züge"},
                "answer_key": "b",
                "explanation": "Experten fordern 'mehr sichere Abstellplätze an Bahnhöfen'."
            }
        ]
    },
    {
        "title": "Selber kochen: Warum frisches Essen im Trend liegt",
        "text": "Immer mehr junge Erwachsene in Deutschland kochen wieder täglich zu Hause selbst. Statt Fertiggerichte aus der Mikrowelle zu essen, kaufen viele frisches Gemüse auf dem Wochenmarkt. Eine Umfrage zeigt, dass 60 Prozent der Befragten Kochen als entspannendes Hobby nach der Arbeit sehen. Beliebt sind vor allem einfache Gerichte wie Nudeln mit Tomatensoße oder bunte Gemüsesuppen. Viele teilen außerdem Fotos ihrer Gerichte in sozialen Medien und tauschen Rezepte aus.",
        "items": [
            {
                "id": 1,
                "question": "Was machen immer mehr junge Erwachsene laut dem Text?",
                "options": {"a": "Sie gehen jeden Tag ins Restaurant", "b": "Sie kochen selbst zu Hause", "c": "Sie essen nur Fertiggerichte"},
                "answer_key": "b",
                "explanation": "Der Text sagt: 'Immer mehr junge Erwachsene in Deutschland kochen wieder täglich zu Hause selbst'."
            },
            {
                "id": 2,
                "question": "Wo kaufen viele Menschen frische Zutaten?",
                "options": {"a": "Auf dem Wochenmarkt", "b": "An der Tankstelle", "c": "Im Internet"},
                "answer_key": "a",
                "explanation": "Laut Text kaufen viele 'frisches Gemüse auf dem Wochenmarkt'."
            },
            {
                "id": 3,
                "question": "Wie empfinden 60 Prozent der Befragten das Kochen?",
                "options": {"a": "Als stressige Pflicht", "b": "Als sehr teuer", "c": "Als entspannendes Hobby"},
                "answer_key": "c",
                "explanation": "Im Text steht: '60 Prozent der Befragten sehen Kochen als entspannendes Hobby nach der Arbeit'."
            },
            {
                "id": 4,
                "question": "Welche Gerichte sind besonders beliebt?",
                "options": {"a": "Schwierige 5-Gänge-Menüs", "b": "Einfache Nudeln und Suppen", "c": "Ausschließlich Fleischgerichte"},
                "answer_key": "b",
                "explanation": "Beliebt sind einfache Gerichte wie Nudeln mit Tomatensoße oder Gemüsesuppen."
            },
            {
                "id": 5,
                "question": "Was machen viele Menschen in sozialen Medien?",
                "options": {"a": "Kochbücher verkaufen", "b": "Fotos und Rezepte teilen", "c": "Restaurants kritisieren"},
                "answer_key": "b",
                "explanation": "Sie teilen Fotos ihrer Gerichte und tauschen Rezepte aus."
            }
        ]
    },
    {
        "title": "Wochenende in der Natur: Wandern begeistert junge Leute",
        "text": "Früher galt Wandern als Hobby für Senioren, doch heute packen immer mehr junge Menschen ihren Rucksack für eine Tour in die Berge. Besonders an den Wochenenden sind Wanderwege im Schwarzwald und in den Alpen gut besucht. Viele schätzen die frische Luft, die Bewegung und die Ruhe abseits der lauten Städte. Moderne Apps helfen bei der Planung der besten Routen. Nach einer langen Wanderung kehren viele in gemütliche Berghütten ein, um regionale Spezialitäten wie Käsespätzle zu genießen.",
        "items": [
            {
                "id": 1,
                "question": "Wer geht heute laut Text immer öfter wandern?",
                "options": {"a": "Nur ältere Senioren", "b": "Immer mehr junge Menschen", "c": "Ausschließlich Profisportler"},
                "answer_key": "b",
                "explanation": "Im Text steht: 'heute packen immer mehr junge Menschen ihren Rucksack'."
            },
            {
                "id": 2,
                "question": "Welche Regionen sind am Wochenende besonders beliebt?",
                "options": {"a": "Große Einkaufszentren", "b": "Der Schwarzwald und die Alpen", "c": "Strände an der Nordsee"},
                "answer_key": "b",
                "explanation": "Laut Text sind Wanderwege 'im Schwarzwald und in den Alpen gut besucht'."
            },
            {
                "id": 3,
                "question": "Was schätzen die Wanderer besonders an den Bergtouren?",
                "options": {"a": "Laute Musik", "b": "Frische Luft und Ruhe", "c": "Günstige Busreisen"},
                "answer_key": "b",
                "explanation": "Viele schätzen 'die frische Luft, die Bewegung und die Ruhe'."
            },
            {
                "id": 4,
                "question": "Womit planen viele Wanderer ihre Routen?",
                "options": {"a": "Mit modernen Apps", "b": "Mit alten Reiseführern aus Papier", "c": "Gar nicht"},
                "answer_key": "a",
                "explanation": "Im Text steht: 'Moderne Apps helfen bei der Planung der besten Routen'."
            },
            {
                "id": 5,
                "question": "Was machen viele nach einer Wanderung?",
                "options": {"a": "Sofort nach Hause fliegen", "b": "In Berghütten essen", "c": "Ein neues Zelt kaufen"},
                "answer_key": "b",
                "explanation": "Sie kehren in Berghütten ein und genießen regionale Spezialitäten."
            }
        ]
    },
    {
        "title": "Deutschlandticket: Mit 49 Euro flexibel durch das ganze Land",
        "text": "Seit der Einführung des Deutschlandtickets nutzen Millionen Menschen Busse und Regionalzüge viel häufiger. Für 49 Euro im Monat kann man im gesamten Nahverkehr flexibel fahren. Viele Pendler lassen das Auto in der Garage und fahren stressfrei mit dem Zug zur Arbeit. Auch für Wochenendausflüge in Nachbarstädte ist das Ticket sehr beliebt. Umweltverbände loben das Ticket, fordern aber mehr Züge und pünktlichere Verbindungen auf den Hauptstrecken.",
        "items": [
            {
                "id": 1,
                "question": "Wie viel kostet das Deutschlandticket pro Monat?",
                "options": {"a": "49 Euro", "b": "100 Euro", "c": "Es ist kostenlos"},
                "answer_key": "a",
                "explanation": "Laut Text kostet das Ticket 49 Euro im Monat."
            },
            {
                "id": 2,
                "question": "In welchen Verkehrsmitteln gilt das Ticket?",
                "options": {"a": "Nur in Flugzeugen", "b": "Im gesamten Nahverkehr mit Bus und Bahn", "c": "Nur in Taxis"},
                "answer_key": "b",
                "explanation": "Man kann im gesamten Nahverkehr (Busse und Regionalzüge) fahren."
            },
            {
                "id": 3,
                "question": "Was machen viele Pendler laut dem Artikel?",
                "options": {"a": "Sie kaufen ein zweites Auto", "b": "Sie fahren mit dem Zug zur Arbeit", "c": "Sie arbeiten nur noch zu Hause"},
                "answer_key": "b",
                "explanation": "Viele Pendler lassen das Auto stehen und fahren mit dem Zug zur Arbeit."
            },
            {
                "id": 4,
                "question": "Wofür wird das Ticket am Wochenende genutzt?",
                "options": {"a": "Für Ausflüge in Nachbarstädte", "b": "Für Flugreisen ins Ausland", "c": "Gar nicht"},
                "answer_key": "a",
                "explanation": "Auch für Wochenendausflüge in Nachbarstädte ist das Ticket sehr beliebt."
            },
            {
                "id": 5,
                "question": "Was fordern Umweltverbände für die Zukunft?",
                "options": {"a": "Höhere Ticketpreise", "b": "Mehr Züge und pünktlichere Verbindungen", "c": "Weniger Bahnhöfe"},
                "answer_key": "b",
                "explanation": "Umweltverbände fordern mehr Züge und pünktlichere Verbindungen."
            }
        ]
    },
    {
        "title": "Sprachen lernen im Tandem: Neue Freunde und lebendige Gespräche",
        "text": "Immer mehr Sprachbegeisterte treffen sich in gemütlichen Cafés zum sogenannten Sprachtandem. Das Prinzip ist einfach: Zwei Personen mit unterschiedlichen Muttersprachen helfen sich gegenseitig beim Lernen. Eine Stunde spricht man Deutsch, die nächste Stunde zum Beispiel Spanisch oder Italienisch. Im Gegensatz zum traditionellen Unterricht lernt man hier besonders die Alltagssprache, Redewendungen und die Kultur des Partnerlandes kennen.",
        "items": [
            {
                "id": 1,
                "question": "Wo treffen sich viele Menschen zum Sprachtandem?",
                "options": {"a": "In großen Bibliotheken", "b": "In gemütlichen Cafés", "c": "Im Reisebüro"},
                "answer_key": "b",
                "explanation": "Sie treffen sich in gemütlichen Cafés."
            },
            {
                "id": 2,
                "question": "Wie funktioniert das Sprachtandem?",
                "options": {"a": "Ein Lehrer gibt Noten", "b": "Zwei Personen mit verschiedenen Muttersprachen helfen sich", "c": "Man lernt nur Grammatik"},
                "answer_key": "b",
                "explanation": "Zwei Personen mit unterschiedlichen Muttersprachen helfen sich gegenseitig."
            },
            {
                "id": 3,
                "question": "Wie wird die Zeit beim Tandem meistens aufgeteilt?",
                "options": {"a": "Jede Sprache wird jeweils eine Stunde gesprochen", "b": "Nur Deutsch wird gesprochen", "c": "Man spricht nur 5 Minuten"},
                "answer_key": "a",
                "explanation": "Eine Stunde spricht man Deutsch, die nächste Stunde die andere Sprache."
            },
            {
                "id": 4,
                "question": "Was lernt man beim Tandem besonders gut?",
                "options": {"a": "Schwere Prüfungsregeln", "b": "Alltagssprache und Redewendungen", "c": "Alte Gedichte"},
                "answer_key": "b",
                "explanation": "Man lernt besonders die Alltagssprache, Redewendungen und die Kultur kennen."
            },
            {
                "id": 5,
                "question": "Was ist der Vorteil gegenüber traditionellem Unterricht?",
                "options": {"a": "Es ist teurer", "b": "Man lernt lebendiger und ungezwungener", "c": "Es gibt viele Hausaufgaben"},
                "answer_key": "b",
                "explanation": "Man lernt in entspannter Atmosphäre von Muttersprachlern."
            }
        ]
    },
    {
        "title": "Flohmärkte am Sonntag: Schätze finden und die Umwelt schonen",
        "text": "An sonnigen Sonntagen zieht es Tausende Menschen auf die großen Flohmärkte der Stadt. Zwischen alten Büchern, Schallplatten, Fahrrädern und Vintage-Kleidung kann man echte Schnäppchen entdecken. Viele Besucher schätzen die besondere Atmosphäre beim Stöbern und Handeln. Der Kauf gebrauchter Dinge schont nicht nur den Geldbeutel, sondern ist auch nachhaltig und spart wertvolle Ressourcen für unsere Umwelt.",
        "items": [
            {
                "id": 1,
                "question": "Wann besuchen viele Menschen den Flohmarkt?",
                "options": {"a": "An sonnigen Sonntagen", "b": "Nur montags früh", "c": "Mitten in der Nacht"},
                "answer_key": "a",
                "explanation": "Laut Text zieht es die Menschen an sonnigen Sonntagen auf den Flohmarkt."
            },
            {
                "id": 2,
                "question": "Was kann man auf dem Flohmarkt kaufen?",
                "options": {"a": "Nur neue Luxusautos", "b": "Bücher, Kleidung, Fahrräder und Schallplatten", "c": "Ausschließlich Lebensmittel"},
                "answer_key": "b",
                "explanation": "Man findet alte Bücher, Schallplatten, Fahrräder und Vintage-Kleidung."
            },
            {
                "id": 3,
                "question": "Was macht den Besuchern besonders viel Spaß?",
                "options": {"a": "Das Stöbern und Handeln", "b": "Die laute Musik", "c": "Die hohen Preise"},
                "answer_key": "a",
                "explanation": "Viele Besucher schätzen die besondere Atmosphäre beim Stöbern und Handeln."
            },
            {
                "id": 4,
                "question": "Warum ist der Kauf gebrauchter Dinge gut für die Umwelt?",
                "options": {"a": "Weil man weniger Abfall produziert und Ressourcen spart", "b": "Weil Flohmärkte im Wald liegen", "c": "Weil man kein Geld braucht"},
                "answer_key": "a",
                "explanation": "Es ist nachhaltig und spart wertvolle Ressourcen."
            },
            {
                "id": 5,
                "question": "Welchen Vorteil hat der Flohmarkt für das eigene Budget?",
                "options": {"a": "Er kostet viel Eintritt", "b": "Man schont den Geldbeutel durch günstige Preise", "c": "Man muss Schulden machen"},
                "answer_key": "b",
                "explanation": "Der Kauf gebrauchter Dinge schont den Geldbeutel."
            }
        ]
    }
]

POOL_LESEN_TEIL2: List[Dict[str, Any]] = [
    {
        "title": "Kaufhaus-Wegweiser (City-Center)",
        "directory": [
            {"floor": "3. Stock", "departments": "Restaurant, Café, Kundentoiletten, Reisebüro, Event-Lounge"},
            {"floor": "2. Stock", "departments": "Kinderbekleidung, Spielzeug, Sportkleidung, Outdoorschuhe, Fahrräder"},
            {"floor": "1. Stock", "departments": "Damen- und Herrenmode, Schuhe, Lederjacken, Koffer und Taschen"},
            {"floor": "Erdgeschoss", "departments": "Parfümerie, Uhren, Schmuck, Information, Zeitschriften, Foto-Pass"},
            {"floor": "Untergeschoss", "departments": "Supermarkt, Bäckerei, Haushaltswaren, Elektrogeräte, Fernseher"}
        ],
        "items": [
            {
                "id": 6,
                "question": "Sie möchten eine warme Winterjacke für Ihre Tochter kaufen.",
                "options": {"a": "1. Stock", "b": "2. Stock", "c": "Anderes Stockwerk"},
                "answer_key": "b",
                "explanation": "Kinderbekleidung befindet sich im 2. Stock."
            },
            {
                "id": 7,
                "question": "Sie haben Hunger und möchten zu Mittag essen.",
                "options": {"a": "3. Stock", "b": "Erdgeschoss", "c": "1. Stock"},
                "answer_key": "a",
                "explanation": "Restaurant und Café befinden sich im 3. Stock."
            },
            {
                "id": 8,
                "question": "Sie suchen eine Kaffeemaschine für Ihre neue Küche.",
                "options": {"a": "Untergeschoss", "b": "1. Stock", "c": "2. Stock"},
                "answer_key": "a",
                "explanation": "Elektrogeräte und Haushaltswaren sind im Untergeschoss."
            },
            {
                "id": 9,
                "question": "Sie möchten ein Geburtstagsgeschenk: eine schöne Damenuhr.",
                "options": {"a": "3. Stock", "b": "Erdgeschoss", "c": "Untergeschoss"},
                "answer_key": "b",
                "explanation": "Uhren und Schmuck sind im Erdgeschoss."
            },
            {
                "id": 10,
                "question": "Sie suchen einen neuen Lederkoffer für Ihren Urlaub.",
                "options": {"a": "1. Stock", "b": "2. Stock", "c": "Anderes Stockwerk"},
                "answer_key": "a",
                "explanation": "Koffer, Taschen und Lederwaren befinden sich im 1. Stock."
            }
        ]
    },
    {
        "title": "Einkaufszentrum 'Alster-Passage' Wegweiser",
        "directory": [
            {"floor": "Obergeschoss 2", "departments": "Kino, Fitnessstudio, Bowlingbahn, Eisdiele, Sushi-Bar"},
            {"floor": "Obergeschoss 1", "departments": "Buchhandlung, Schreibwaren, Musikinstrumente, Computer & Laptops, Handyzubehör"},
            {"floor": "Erdgeschoss", "departments": "Damenbekleidung, Herrenmode, Schuhe, Juwelier, Optiker & Brillen"},
            {"floor": "Untergeschoss", "departments": "Apotheke, Drogeriemarkt, Blumengeschäft, Schlüsseldienst & Schuhreparatur"}
        ],
        "items": [
            {
                "id": 6,
                "question": "Sie brauchen dringend Kopfschmerztabletten und Duschgel.",
                "options": {"a": "Untergeschoss", "b": "Obergeschoss 1", "c": "Anderes Stockwerk"},
                "answer_key": "a",
                "explanation": "Apotheke und Drogeriemarkt befinden sich im Untergeschoss."
            },
            {
                "id": 7,
                "question": "Sie möchten am Abend mit Freunden einen Film ansehen.",
                "options": {"a": "Obergeschoss 2", "b": "Erdgeschoss", "c": "Untergeschoss"},
                "answer_key": "a",
                "explanation": "Das Kino befindet sich im Obergeschoss 2."
            },
            {
                "id": 8,
                "question": "Sie suchen ein neues Ladekabel für Ihr Smartphone.",
                "options": {"a": "Erdgeschoss", "b": "Obergeschoss 1", "c": "Obergeschoss 2"},
                "answer_key": "b",
                "explanation": "Handyzubehör und Computer sind im Obergeschoss 1."
            },
            {
                "id": 9,
                "question": "Sie brauchen eine neue Sonnenbrille für den Sommer.",
                "options": {"a": "Obergeschoss 2", "b": "Erdgeschoss", "c": "Anderes Stockwerk"},
                "answer_key": "b",
                "explanation": "Optiker & Brillen befinden sich im Erdgeschoss."
            },
            {
                "id": 10,
                "question": "Sie möchten einen Blumenstrauß zum Geburtstag verschenken.",
                "options": {"a": "Untergeschoss", "b": "Obergeschoss 1", "c": "Obergeschoss 2"},
                "answer_key": "a",
                "explanation": "Das Blumengeschäft befindet sich im Untergeschoss."
            }
        ]
    },
    {
        "title": "Einkaufszentrum am Hauptbahnhof",
        "directory": [
            {"floor": "3. Etage", "departments": "Zahnarztpraxis, Physiotherapie, Reisezentrum DB, Fundbüro"},
            {"floor": "2. Etage", "departments": "Sportartikel, Wanderausrüstung, Fahrräder, Campingbedarf"},
            {"floor": "1. Etage", "departments": "Elektronikfachmarkt, Handyreparatur, Kameras, Videospiele"},
            {"floor": "Erdgeschoss", "departments": "Bäckerei, Zeitschriften & Zeitungen, Café, Geldautomat"},
            {"floor": "Untergeschoss", "departments": "Schließfächer, Apotheke, Drogerie, Bio-Supermarkt"}
        ],
        "items": [
            {
                "id": 6,
                "question": "Sie möchten Ihr Gepäck für ein paar Stunden sicher einschließen.",
                "options": {"a": "Untergeschoss", "b": "1. Etage", "c": "3. Etage"},
                "answer_key": "a",
                "explanation": "Schließfächer befinden sich im Untergeschoss."
            },
            {
                "id": 7,
                "question": "Sie brauchen ein Zugticket nach Berlin und Beratung.",
                "options": {"a": "3. Etage", "b": "Erdgeschoss", "c": "2. Etage"},
                "answer_key": "a",
                "explanation": "Das DB Reisezentrum befindet sich in der 3. Etage."
            },
            {
                "id": 8,
                "question": "Das Display Ihres Smartphones ist kaputt und muss repariert werden.",
                "options": {"a": "Erdgeschoss", "b": "1. Etage", "c": "Anderes Stockwerk"},
                "answer_key": "b",
                "explanation": "Die Handyreparatur befindet sich in der 1. Etage."
            },
            {
                "id": 9,
                "question": "Sie möchten vor der Zugfahrt einen frischen Kaffee und Brezeln kaufen.",
                "options": {"a": "2. Etage", "b": "Erdgeschoss", "c": "3. Etage"},
                "answer_key": "b",
                "explanation": "Bäckerei und Café sind im Erdgeschoss."
            },
            {
                "id": 10,
                "question": "Sie suchen ein neues Zelt für einen Wochenendausflug in die Natur.",
                "options": {"a": "2. Etage", "b": "Untergeschoss", "c": "Anderes Stockwerk"},
                "answer_key": "a",
                "explanation": "Wanderausrüstung und Campingbedarf befinden sich in der 2. Etage."
            }
        ]
    },
    {
        "title": "Bürger- und Kulturzentrum am Marktplatz",
        "directory": [
            {"floor": "3. Obergeschoss", "departments": "Stadtbibliothek, Lesesaal, Kostenloses WLAN, Hörbuchausleihe"},
            {"floor": "2. Obergeschoss", "departments": "Volkshochschule (VHS), Sprachkurse, Computerräume, Beratungsbüro"},
            {"floor": "1. Obergeschoss", "departments": "Bürgeramt, Passstelle, Wohnsitzanmeldung, Führerscheinstelle"},
            {"floor": "Erdgeschoss", "departments": "Theaterkasse, Tourist-Information, Café am Markt, Stadtplanverkauf"}
        ],
        "items": [
            {
                "id": 6,
                "question": "Sie sind neu in die Stadt gezogen und müssen Ihren Wohnsitz anmelden.",
                "options": {"a": "1. Obergeschoss", "b": "3. Obergeschoss", "c": "Anderes Stockwerk"},
                "answer_key": "a",
                "explanation": "Bürgeramt und Wohnsitzanmeldung befinden sich im 1. Obergeschoss."
            },
            {
                "id": 7,
                "question": "Sie möchten sich für einen Abend-Deutschkurs anmelden.",
                "options": {"a": "Erdgeschoss", "b": "2. Obergeschoss", "c": "3. Obergeschoss"},
                "answer_key": "b",
                "explanation": "Die Volkshochschule mit Sprachkursen befindet sich im 2. Obergeschoss."
            },
            {
                "id": 8,
                "question": "Sie möchten Tickets für das Konzert am Samstagabend kaufen.",
                "options": {"a": "Erdgeschoss", "b": "1. Obergeschoss", "c": "2. Obergeschoss"},
                "answer_key": "a",
                "explanation": "Die Theaterkasse befindet sich im Erdgeschoss."
            },
            {
                "id": 9,
                "question": "Sie suchen einen ruhigen Platz, um ein Buch zu lesen und im Internet zu recherchieren.",
                "options": {"a": "1. Obergeschoss", "b": "3. Obergeschoss", "c": "Anderes Stockwerk"},
                "answer_key": "b",
                "explanation": "Stadtbibliothek und Lesesaal befinden sich im 3. Obergeschoss."
            },
            {
                "id": 10,
                "question": "Sie brauchen einen neuen Reisepass für Ihren Urlaub.",
                "options": {"a": "1. Obergeschoss", "b": "2. Obergeschoss", "c": "Erdgeschoss"},
                "answer_key": "a",
                "explanation": "Das Bürgeramt mit Passstelle ist im 1. Obergeschoss."
            }
        ]
    }
]

POOL_LESEN_TEIL3: List[Dict[str, Any]] = [
    {
        "sender": "Anna Schneider",
        "recipient": "Markus",
        "subject": "Neuigkeiten aus meiner neuen Wohnung!",
        "text": "Lieber Markus,\nendlich habe ich etwas Zeit zum Schreiben. Der Umzug letzte Woche war ganz schön anstrengend, aber meine Freunde haben mir super geholfen. Meine neue Wohnung in Freiburg ist wirklich toll! Sie hat zwei helle Zimmer, einen kleinen Balkon und liegt direkt neben einem schönen Park. Am nächsten Samstag mache ich eine kleine Einweihungsparty ab 18 Uhr. Ich grille auf dem Balkon und es gibt leckere Salate. Hast du Zeit und Lust zu kommen? Du kannst auch gerne bei mir auf dem Sofa übernachten, wenn es spät wird.\nHerzliche Grüße,\nAnna",
        "items": [
            {
                "id": 11,
                "question": "Wie fand Anna ihren Umzug?",
                "options": {"a": "Sehr einfach", "b": "Ziemlich anstrengend", "c": "Langweilig"},
                "answer_key": "b",
                "explanation": "Anna schreibt: 'Der Umzug letzte Woche war ganz schön anstrengend'."
            },
            {
                "id": 12,
                "question": "Was gefällt Anna an ihrer neuen Wohnung?",
                "options": {"a": "Sie hat einen Garten", "b": "Sie ist sehr groß mit 4 Zimmern", "c": "Sie hat zwei helle Zimmer und einen Balkon"},
                "answer_key": "c",
                "explanation": "Laut Text hat die Wohnung 'zwei helle Zimmer' und einen 'kleinen Balkon'."
            },
            {
                "id": 13,
                "question": "Was plant Anna für nächsten Samstag?",
                "options": {"a": "Eine Einweihungsparty mit Grillen", "b": "Einen Ausflug ins Museum", "c": "Eine Reise nach Freiburg"},
                "answer_key": "a",
                "explanation": "Sie plant eine Einweihungsparty ab 18 Uhr und will auf dem Balkon grillen."
            },
            {
                "id": 14,
                "question": "Wo liegt die neue Wohnung von Anna?",
                "options": {"a": "Am Bahnhof", "b": "Direkt neben einem Park", "c": "Im Industriegebiet"},
                "answer_key": "b",
                "explanation": "Anna schreibt, die Wohnung 'liegt direkt neben einem schönen Park'."
            },
            {
                "id": 15,
                "question": "Was bietet Anna Markus für die Nacht an?",
                "options": {"a": "Ein Hotelzimmer zu buchen", "b": "Auf dem Sofa zu schlafen", "c": "Den letzten Zug zu nehmen"},
                "answer_key": "b",
                "explanation": "Sie schreibt: 'Du kannst auch gerne bei mir auf dem Sofa übernachten'."
            }
        ]
    },
    {
        "sender": "Tobias Weber",
        "recipient": "Sarah",
        "subject": "Sommerfest am Badesee am Samstag!",
        "text": "Hallo Sarah,\nwie geht es dir? Da das Wetter am Samstag fantastisch werden soll (über 28 Grad!), planen wir ab 14 Uhr ein großes Sommerfest am Badesee. Wir bringen Decken, Musik und einen Volleyball mit. Jeder bringt etwas zum Essen mit – ich mache meinen berühmten Nudelsalat und backe Muffins. Bringst du vielleicht ein paar kalte Getränke oder Obst mit? Vergiss deine Badesachen und Sonnencreme nicht! Wir treffen uns direkt am Parkplatz vor dem Kiosk.\nBis Samstag,\nTobias",
        "items": [
            {
                "id": 11,
                "question": "Warum organisiert Tobias das Fest am Samstag?",
                "options": {"a": "Weil es regnen soll", "b": "Weil das Wetter warm und sonnig wird", "c": "Weil er Geburtstag hat"},
                "answer_key": "b",
                "explanation": "Tobias schreibt, dass das Wetter fantastisch wird mit über 28 Grad."
            },
            {
                "id": 12,
                "question": "Wann beginnt das Fest am Badesee?",
                "options": {"a": "Um 10 Uhr morgens", "b": "Um 14 Uhr", "c": "Erst um 20 Uhr"},
                "answer_key": "b",
                "explanation": "Im Text steht: 'planen wir ab 14 Uhr ein großes Sommerfest'."
            },
            {
                "id": 13,
                "question": "Was bringt Tobias zum Essen mit?",
                "options": {"a": "Nudelsalat und Muffins", "b": "Nur belegte Brötchen", "c": "Gekaufte Pizza"},
                "answer_key": "a",
                "explanation": "Tobias bringt seinen 'berühmten Nudelsalat' und backt 'Muffins'."
            },
            {
                "id": 14,
                "question": "Worum bittet Tobias Sarah?",
                "options": {"a": "Einen Grill zu kaufen", "b": "Kalte Getränke oder Obst mitzubringen", "c": "Ihn mit dem Auto abzuholen"},
                "answer_key": "b",
                "explanation": "Er fragt: 'Bringst du vielleicht ein paar kalte Getränke oder Obst mit?'"
            },
            {
                "id": 15,
                "question": "Wo ist der Treffpunkt am Samstag?",
                "options": {"a": "Am Bahnhof", "b": "Am Parkplatz vor dem Kiosk", "c": "Bei Tobias zu Hause"},
                "answer_key": "b",
                "explanation": "Im Text steht: 'Wir treffen uns direkt am Parkplatz vor dem Kiosk'."
            }
        ]
    },
    {
        "sender": "Maximilian Braun",
        "recipient": "Julia",
        "subject": "Wochenendausflug in den Harz",
        "text": "Liebe Julia,\nich habe tolle Neuigkeiten: Für unseren Ausflug in den Harz am kommenden Wochenende habe ich schon eine gemütliche Ferienhütte reserviert. Wir fahren am Samstagmorgen um 8:15 Uhr mit dem Regionalzug ab Hauptbahnhof. Bitte denk an feste Wanderschuhe und eine Regenjacke, da das Wetter in den Bergen schnell wechseln kann. Das Bahnticket habe ich bereits für uns beide online gekauft. Melde dich kurz, ob die Uhrzeit für dich passt!\nViele Grüße,\nMaximilian",
        "items": [
            {
                "id": 11,
                "question": "Was hat Maximilian bereits für den Ausflug organisiert?",
                "options": {"a": "Ein Luxushotel gebucht", "b": "Eine Ferienhütte reserviert und Tickets gekauft", "c": "Ein Auto gemietet"},
                "answer_key": "b",
                "explanation": "Er hat die Ferienhütte reserviert und das Bahnticket gekauft."
            },
            {
                "id": 12,
                "question": "Wann fährt der Zug am Samstag ab?",
                "options": {"a": "Um 8:15 Uhr", "b": "Um 12:00 Uhr", "c": "Erst am Nachmittag"},
                "answer_key": "a",
                "explanation": "Im Text steht: 'am Samstagmorgen um 8:15 Uhr'."
            },
            {
                "id": 13,
                "question": "Welche Kleidung empfiehlt Maximilian Julia?",
                "options": {"a": "Badekleidung", "b": "Feste Wanderschuhe und eine Regenjacke", "c": "Nur ein Sommerkleid"},
                "answer_key": "b",
                "explanation": "Er bittet sie, an feste Wanderschuhe und eine Regenjacke zu denken."
            },
            {
                "id": 14,
                "question": "Wie reisen Maximilian und Julia in den Harz?",
                "options": {"a": "Mit dem Flugzeug", "b": "Mit dem Regionalzug ab Hauptbahnhof", "c": "Mit dem Fahrrad"},
                "answer_key": "b",
                "explanation": "Sie fahren mit dem Regionalzug ab Hauptbahnhof."
            },
            {
                "id": 15,
                "question": "Worum bittet Maximilian Julia am Ende?",
                "options": {"a": "Um eine kurze Rückmeldung zur Abfahrtszeit", "b": "Ihm Geld zu überweisen", "c": "Essen für alle zu kochen"},
                "answer_key": "a",
                "explanation": "Er schreibt: 'Melde dich kurz, ob die Uhrzeit für dich passt!'."
            }
        ]
    }
]

POOL_LESEN_TEIL4: List[Dict[str, Any]] = [
    {
        "ads": [
            {"id": "a", "title": "Gitarrenunterricht für Anfänger", "text": "Lerne deine Lieblingssongs! Erfahrener Musiker gibt Einzelunterricht. Flexible Termine abends. Tel: 0171-234567"},
            {"id": "b", "title": "Schöne 2-Zimmer-Wohnung im Zentrum", "text": "Helle 60 qm Wohnung mit Balkon, voll möbliert, ab sofort frei. Warmmiete 750 Euro. info@wohnen-city.de"},
            {"id": "c", "title": "Babysitterin sucht Familie", "text": "Pädagogik-Studentin (22 Jahre) hat nachmittags und am Wochenende Zeit für Kinderbetreuung. Tel: 0152-987654"},
            {"id": "d", "title": "Citybike 28 Zoll zu verkaufen", "text": "Gebrauchtes, sehr gepflegtes Damenfahrrad mit 7-Gang-Schaltung und Korb. Preis: 95 Euro VB. Tel: 0160-112233"},
            {"id": "e", "title": "Fitnessstudio 'Bodyfit' Spezialangebot", "text": "Jetzt anmelden und 3 Monate gratis trainieren! Großer Saunabereich und moderne Geräte. www.bodyfit-club.de"},
            {"id": "f", "title": "Italienischer Kochkurs am Samstag", "text": "Lerne echte Pasta und Pizza selbst machen! 4-stündiger Workshop in kleiner Gruppe. Anmeldung unter: kochschule-roma.de"}
        ],
        "items": [
            {
                "id": 16,
                "person": "Claudia sucht jemanden, der an zwei Nachmittagen auf ihren 4-jährigen Sohn aufpasst.",
                "answer_key": "c",
                "explanation": "Anzeige C bietet Kinderbetreuung / Babysitting durch eine Studentin an."
            },
            {
                "id": 17,
                "person": "Maria sucht ein preiswertes Fahrrad für den täglichen Weg zur Arbeit.",
                "answer_key": "d",
                "explanation": "Anzeige D verkauft ein Citybike für 95 Euro."
            },
            {
                "id": 18,
                "person": "Jan möchte am Wochenende lernen, wie man echte italienische Pasta kocht.",
                "answer_key": "f",
                "explanation": "Anzeige F bietet einen Samstags-Kochkurs für Pasta und Pizza."
            },
            {
                "id": 19,
                "person": "Sarah möchte nach Feierabend ein Instrument lernen, am liebsten Gitarre.",
                "answer_key": "a",
                "explanation": "Anzeige A bietet Gitarrenunterricht mit flexiblen Abendterminen ab 18:00 Uhr."
            },
            {
                "id": 20,
                "person": "Felix sucht einen Schwimmkurs für Fortgeschrittene am Wochenende.",
                "answer_key": "x",
                "explanation": "Keine der Anzeigen bietet einen Schwimmkurs an, daher ist die richtige Antwort 'x'."
            }
        ]
    },
    {
        "ads": [
            {"id": "a", "title": "Spanisch lernen in Kleingruppen", "text": "Abendkurs für Anfänger ab Oktober. Nette Atmosphäre und muttersprachliche Lehrerin. www.spanisch-lernen.de"},
            {"id": "b", "title": "Verkaufe Mountainbike 26 Zoll", "text": "Top Zustand, neue Reifen, Scheibenbremsen. Ideal für Touren im Wald. Preis 140 Euro. Tel: 0176-554433"},
            {"id": "c", "title": "Klavierlehrer erteilt Privatunterricht", "text": "Klassik und Pop für Kinder und Erwachsene. Hausbesuche möglich. Tel: 0179-887766"},
            {"id": "d", "title": "Hundebetreuung mit Herz", "text": "Liebevolle Betreuung für Ihren Hund während Ihres Urlaubs oder bei der Arbeit. Großer Garten vorhanden. Tel: 0151-332211"},
            {"id": "e", "title": "Günstige Möbel abzugeben", "text": "Esstisch mit 4 Stühlen aus Holz sowie großes Bücherregal wegen Umzug günstig zu verkaufen. Tel: 0170-998877"},
            {"id": "f", "title": "Yoga am Morgen im Park", "text": "Entspannter Start in den Tag jeden Dienstag und Donnerstag um 7:30 Uhr. Für alle Level geeignet. www.yoga-park.de"}
        ],
        "items": [
            {
                "id": 16,
                "person": "Stefan fährt für zwei Wochen in den Urlaub und sucht jemanden für seinen Labrador.",
                "answer_key": "d",
                "explanation": "Anzeige D bietet liebevolle Hundebetreuung während des Urlaubs."
            },
            {
                "id": 17,
                "person": "Lisa zieht in eine neue Wohnung und braucht noch einen Esstisch und Stühle.",
                "answer_key": "e",
                "explanation": "Anzeige E verkauft günstigen Esstisch mit 4 Stühlen."
            },
            {
                "id": 18,
                "person": "Jonas möchte vor der Arbeit Sport treiben und sich mit Dehnübungen entspannen.",
                "answer_key": "f",
                "explanation": "Anzeige F bietet Yoga am Morgen um 7:30 Uhr im Park an."
            },
            {
                "id": 19,
                "person": "Katrin plant eine Reise nach Madrid und möchte davor die spanische Sprache lernen.",
                "answer_key": "a",
                "explanation": "Anzeige A bietet Spanischkurse in Kleingruppen an."
            },
            {
                "id": 20,
                "person": "Michael sucht einen Tanzkurs für Standardtänze am Wochenende.",
                "answer_key": "x",
                "explanation": "Keine der Anzeigen bietet Standardtanzkurse an, daher ist die Antwort 'x'."
            }
        ]
    },
    {
        "ads": [
            {"id": "a", "title": "Salsa & Bachata Tanzkurs", "text": "Neue Anfängerkurse jeden Freitag ab 19 Uhr. Kein fester Tanzpartner erforderlich! www.salsa-club.de"},
            {"id": "b", "title": "Mathematik-Nachhilfe bis Klasse 10", "text": "Geduldiger Student hilft bei Prüfungsvorbereitung und Hausaufgaben. Günstige Tarife. Tel: 0162-443322"},
            {"id": "c", "title": "Campingzelt für 4 Personen zu vermieten", "text": "Großes wasserdichtes Familienzelt inklusive Luftmatratzen für Wochenenden. 30 Euro/Wochenende. Tel: 0173-889900"},
            {"id": "d", "title": "Fotokurs für Einsteiger", "text": "Lerne die Grundlagen der digitalen Spiegelreflexkamera am Wochenende in der Natur. www.fotowelt-online.de"},
            {"id": "e", "title": "Gepflegter Kleinwagen VW Polo", "text": "Baujahr 2018, wenig Kilometer, TÜV neu, sparsamer Benzinverbrauch. Preis: 5.800 Euro. Tel: 0157-123456"},
            {"id": "f", "title": "Gartenhilfe gesucht", "text": "Suchen zuverlässige Unterstützung für Rasenmähen und Hecke schneiden 2x im Monat. 15 Euro/Stunde. Tel: 0178-654321"}
        ],
        "items": [
            {
                "id": 16,
                "person": "Tim sucht ein großes Zelt für einen Wochenendausflug an den See mit Freunden.",
                "answer_key": "c",
                "explanation": "Anzeige C vermietet ein 4-Personen-Familienzelt für Wochenenden."
            },
            {
                "id": 17,
                "person": "Elena möchte gerne tanzen lernen, hat aber im Moment keinen festen Tanzpartner.",
                "answer_key": "a",
                "explanation": "Anzeige A bietet Salsa-Kurse an, bei denen kein fester Partner nötig ist."
            },
            {
                "id": 18,
                "person": "Familie Berger braucht Hilfe beim Rasenmähen in ihrem großen Garten.",
                "answer_key": "f",
                "explanation": "Anzeige F sucht Gartenhilfe für Rasenmähen und Hecke schneiden."
            },
            {
                "id": 19,
                "person": "Simon hat eine neue Kamera bekommen und möchte lernen, wie man schöne Naturfotos macht.",
                "answer_key": "d",
                "explanation": "Anzeige D bietet Fotokurse für Einsteiger in der Natur an."
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
        "scenario_german": "Sie können heute Abend nicht pünktlich zum Treffen mit Ihrem Freund Michael kommen.",
        "instructions_german": "Schreiben Sie eine kurze Nachricht an Michael (ca. 20–30 Wörter). Schreiben Sie zu allen drei Punkten:",
        "bullet_points": [
            "Entschuldigen Sie sich für die Verspätung.",
            "Nennen Sie den Grund dafür.",
            "Schlagen Sie einen neuen Treffpunkt oder eine neue Uhrzeit vor."
        ],
        "target_word_count": "20–30 Wörter",
        "tips_english": "Write a short SMS/note (approx. 20-30 words). Address all 3 bullet points, using an informal greeting and sign-off."
    },
    {
        "teil": 1,
        "title": "Schreiben Teil 1: Einladung ablehnen und neues Treffen vorschlagen",
        "scenario_german": "Ihre Kollegin Maria hat Sie zum Abendessen am Freitag eingeladen. Sie haben aber keine Zeit.",
        "instructions_german": "Schreiben Sie eine kurze Nachricht an Maria (ca. 20–30 Wörter). Schreiben Sie zu allen drei Punkten:",
        "bullet_points": [
            "Bedanken Sie sich für die nette Einladung.",
            "Erklären Sie höflich, warum Sie am Freitag nicht kommen können.",
            "Schlagen Sie ein Treffen am nächsten Wochenende vor."
        ],
        "target_word_count": "20–30 Wörter",
        "tips_english": "Write a short note (approx. 20-30 words). Thank for the invite, give your reason, and propose an alternative date."
    },
    {
        "teil": 1,
        "title": "Schreiben Teil 1: Sporttraining absagen",
        "scenario_german": "Sie trainieren regelmäßig mit Ihrem Freund Lukas im Fitnessstudio, können aber heute nicht kommen.",
        "instructions_german": "Schreiben Sie eine Nachricht an Lukas (ca. 20–30 Wörter). Schreiben Sie zu allen drei Punkten:",
        "bullet_points": [
            "Sagen Sie das gemeinsame Training für heute ab.",
            "Nennen Sie den Grund (z.B. Erkältung oder Überstunden).",
            "Vereinbaren Sie einen Termin für die nächste Woche."
        ],
        "target_word_count": "20–30 Wörter",
        "tips_english": "Write an informal note (20-30 words) cancelling training, stating why, and rescheduling."
    },
    {
        "teil": 1,
        "title": "Schreiben Teil 1: Hilfe beim Umzug erbitten",
        "scenario_german": "Sie ziehen am kommenden Samstag in eine neue Wohnung und brauchen Unterstützung beim Tragen von Kartons.",
        "instructions_german": "Schreiben Sie eine Nachricht an Ihre Freundin Julia (ca. 20–30 Wörter). Schreiben Sie zu allen drei Punkten:",
        "bullet_points": [
            "Erklären Sie, dass Sie am Samstag umziehen.",
            "Bitten Sie Julia höflich um Hilfe beim Kistentragen.",
            "Laden Sie sie als Dankeschön zum Pizzaessen ein."
        ],
        "target_word_count": "20–30 Wörter",
        "tips_english": "Write a short note asking for moving help and inviting for pizza afterwards."
    },
    {
        "teil": 1,
        "title": "Schreiben Teil 1: Schlüssel an Nachbarn übergeben",
        "scenario_german": "Sie fahren für eine Woche in den Urlaub und möchten Ihren Wohnungsschlüssel Ihrem Nachbarn Herrn Schmidt geben.",
        "instructions_german": "Schreiben Sie eine Nachricht an Herrn Schmidt (ca. 20–30 Wörter). Schreiben Sie zu allen drei Punkten:",
        "bullet_points": [
            "Informieren Sie über Ihren Urlaub ab Montag.",
            "Bitten Sie ihn, nach der Post und den Blumen zu schauen.",
            "Schlagen Sie vor, wann Sie den Schlüssel vorbeibringen können."
        ],
        "target_word_count": "20–30 Wörter",
        "tips_english": "Write a friendly note regarding apartment keys, watering plants, and handing over the key."
    }
]

POOL_SCHREIBEN_TEIL2: List[Dict[str, Any]] = [
    {
        "teil": 2,
        "title": "Schreiben Teil 2: Sprachkurs anfragen",
        "scenario_german": "Sie möchten im nächsten Monat an einer Sprachschule in Heidelberg einen Deutschkurs (Stufe B1) besuchen. Schreiben Sie an die Sprachschule.",
        "instructions_german": "Schreiben Sie eine E-Mail an Frau Weber von der Sprachschule (ca. 30–40 Wörter). Schreiben Sie zu allen vier Punkten:",
        "bullet_points": [
            "Grund für Ihr Schreiben",
            "Informationen zum Kurstermin und Beginn erfragen",
            "Nach den Kosten und Unterkünften fragen",
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
            "Nach dem Frühstück und Preisen fragen",
            "Nach Parkplätzen am Hotel fragen",
            "Höfliche formelle Anrede und Schlussformel"
        ],
        "target_word_count": "30–40 Wörter",
        "tips_english": "Write a formal email requesting hotel reservation, inquiring about prices, breakfast and parking."
    },
    {
        "teil": 2,
        "title": "Schreiben Teil 2: Wohnungsbesichtigung anfragen",
        "scenario_german": "Sie haben im Internet eine Anzeige für eine schöne 2-Zimmer-Wohnung gesehen und möchten die Wohnung besichtigen.",
        "instructions_german": "Schreiben Sie eine E-Mail an den Vermieter, Herrn Müller (ca. 30–40 Wörter). Schreiben Sie zu allen vier Punkten:",
        "bullet_points": [
            "Sich kurz vorstellen (Beruf / Personenzahl)",
            "Interesse an der Wohnung bekunden",
            "Nach einem Termin für eine Besichtigung fragen",
            "Passende formelle Grußformel"
        ],
        "target_word_count": "30–40 Wörter",
        "tips_english": "Write a formal apartment inquiry email to Herr Müller introducing yourself and requesting a viewing appointment."
    },
    {
        "teil": 2,
        "title": "Schreiben Teil 2: Reparaturmeldung an Hausverwaltung",
        "scenario_german": "In Ihrer Mietwohnung ist seit zwei Tagen die Heizung im Wohnzimmer defekt. Schreiben Sie an die Hausverwaltung.",
        "instructions_german": "Schreiben Sie eine formelle E-Mail an die Hausverwaltung Schneider (ca. 30–40 Wörter). Schreiben Sie zu allen vier Punkten:",
        "bullet_points": [
            "Problem mit der Heizung beschreiben",
            "Erklären, wie kalt es in der Wohnung ist",
            "Um eine schnelle Reparatur bitten",
            "Erreichbarkeit für den Handwerker angeben"
        ],
        "target_word_count": "30–40 Wörter",
        "tips_english": "Write a formal email reporting a broken heating system and requesting urgent repair."
    },
    {
        "teil": 2,
        "title": "Schreiben Teil 2: Bewerbung für Minijob im Café",
        "scenario_german": "Sie suchen einen Nebenjob am Wochenende und haben einen Aushang für eine Servicekraft im 'Café Central' gesehen.",
        "instructions_german": "Schreiben Sie eine E-Mail an den Café-Inhaber, Herrn Fischer (ca. 30–40 Wörter). Schreiben Sie zu allen vier Punkten:",
        "bullet_points": [
            "Grund für die Bewerbung nennen",
            "Eigene Erfahrung in der Gastronomie erwähnen",
            "Verfügbarkeit am Wochenende angeben",
            "Formelle Anrede und Schlussformel"
        ],
        "target_word_count": "30–40 Wörter",
        "tips_english": "Write a formal job application email to Herr Fischer expressing interest in the weekend service job."
    }
]



