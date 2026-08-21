"""Async generators for Goethe A2 Exam Teile (Lesen & Schreiben).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, Tuple

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


def _extract_json(text: str) -> Dict[str, Any] | None:
    """Safely extract and parse JSON from LLM generation output."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json|JSON)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Fallback Templates (Ensures 100% Reliability if LLM output is malformed)
# ---------------------------------------------------------------------------

FALLBACK_LESEN_TEIL1 = {
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
}

FALLBACK_LESEN_TEIL2 = {
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
}

FALLBACK_LESEN_TEIL3 = {
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
            "question": "Wo liegt Annas neue Wohnung?",
            "options": {"a": "Neben einem Park", "b": "Direkt am Bahnhof", "c": "Im Stadtzentrum"},
            "answer_key": "a",
            "explanation": "Die Wohnung liegt 'direkt neben einem schönen Park'."
        },
        {
            "id": 13,
            "question": "Was plant Anna für nächsten Samstag?",
            "options": {"a": "Einen Ausflug ins Museum", "b": "Einen Kochkurs", "c": "Eine Einweihungsparty"},
            "answer_key": "c",
            "explanation": "Sie macht 'eine kleine Einweihungsparty ab 18 Uhr'."
        },
        {
            "id": 14,
            "question": "Was möchte Anna auf der Party machen?",
            "options": {"a": "Auf dem Balkon grillen", "b": "Pizza bestellen", "c": "In ein Restaurant gehen"},
            "answer_key": "a",
            "explanation": "Sie schreibt: 'Ich grille auf dem Balkon'."
        },
        {
            "id": 15,
            "question": "Was bietet Anna Markus an?",
            "options": {"a": "Ihn mit dem Auto abzuholen", "b": "Auf dem Sofa zu übernachten", "c": "Ihm beim Umzug zu helfen"},
            "answer_key": "b",
            "explanation": "Anna schreibt: 'Du kannst auch gerne bei mir auf dem Sofa übernachten'."
        }
    ]
}

FALLBACK_LESEN_TEIL4 = {
    "title": "Anzeigen und Personen (Matching Ads)",
    "instructions": "Finden Sie für jede Person die passende Anzeige (a-f). Wenn keine Anzeige passt, wählen Sie 'x'.",
    "ads": [
        {"id": "a", "title": "Gitarrenunterricht in Berlin", "text": "Erfahrener Musiker bietet privaten Gitarrenunterricht für Anfänger. Flexible Abendtermine ab 18:00 Uhr. Tel: 0176-123456"},
        {"id": "b", "title": "Yoga im Stadtpark", "text": "Morgen-Yoga jeden Dienstag & Donnerstag von 7:30 bis 8:30 Uhr im Park. Für alle Levels. Beitrag: 8 Euro/Stunde. info@yoga-park.de"},
        {"id": "c", "title": "Spanisch lernen für Einsteiger", "text": "Sprachschule Lingua: Abendkurse A1-A2 für Reisende und Berufstätige. Start jeden ersten Montag im Monat. www.lingua-kurse.de"},
        {"id": "d", "title": "Günstiges Damenfahrrad", "text": "Verkaufe gut gepflegtes 7-Gang-Citybike mit Korb und Licht. 95 Euro VB. Abholung in Hamburg-Nord. Tel: 0151-998877"},
        {"id": "e", "title": "Wohnung zur Zwischenmiete", "text": "Möblierte 2-Zimmer-Wohnung in Köln für 3 Monate frei (Juni-August). 680 Euro warm. Nichtraucher. kontakt@wohnen-koeln.de"},
        {"id": "f", "title": "Italienischer Kochkurs", "text": "Lernen Sie echte Pizza und frische Pasta selbst machen! Samstags 14:00-18:00 Uhr inkl. Getränke und Zutaten. 45 Euro. www.kochstudio.de"}
    ],
    "items": [
        {
            "id": 16,
            "person": "Lukas möchte im Sommerurlaub nach Madrid reisen und vorher Grundkenntnisse in Spanisch lernen.",
            "answer_key": "c",
            "explanation": "Anzeige C bietet Spanisch-Abendkurse für Einsteiger und Reisende."
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
}

FALLBACK_SCHREIBEN_TEIL1 = {
    "teil": 1,
    "title": "Schreiben Teil 1: Kurze Mitteilung (SMS / Nachricht)",
    "scenario_german": "Sie können heute Abend nicht pünktlich zum Treffen mit Ihrem Freund Michael kommen.",
    "instructions_german": "Schreiben Sie eine kurze Nachricht an Michael (ca. 20–30 Wörter). Schreiben Sie zu allen drei Punkten:",
    "bullet_points": [
        "Entschuldigen Sie sich für die Verspätung.",
        "Nennen Sie den Grund dafür.",
        "Schlagen Sie einen neuen Treffpunkt oder eine neue Uhrzeit vor."
    ],
    "target_word_count": "20–30 Wörter",
    "tips_english": "Write a short SMS/note (approx. 20-30 words). Address all 3 bullet points, using an informal greeting and sign-off."
}

FALLBACK_SCHREIBEN_TEIL2 = {
    "teil": 2,
    "title": "Schreiben Teil 2: Formelle / Halbformelle E-Mail",
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
}


# ---------------------------------------------------------------------------
# Generator Functions (Dispatched concurrently via asyncio.to_thread)
# ---------------------------------------------------------------------------

async def generate_lesen_teil1(level: str = "A2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Lesen Teil 1 (Newspaper Article). Returns (sanitized_teil, answer_key)."""
    try:
        template = load_prompt("exam_lesen_teil1.txt")
        raw = await asyncio.to_thread(generate, template, max_tokens=600, temperature=0.6)
        parsed = _extract_json(raw)
        if parsed and "text" in parsed and "items" in parsed and len(parsed["items"]) == 5:
            data = parsed
        else:
            data = FALLBACK_LESEN_TEIL1
    except Exception as e:
        logger.warning("Error generating Lesen Teil 1, using fallback: %s", e)
        data = FALLBACK_LESEN_TEIL1

    # Extract answer key
    answer_key = {str(item["id"]): item.get("answer_key", "a") for item in data["items"]}
    explanations = {str(item["id"]): item.get("explanation", "") for item in data["items"]}

    # Build sanitized frontend data
    sanitized_items = []
    for item in data["items"]:
        sanitized_items.append({
            "id": item["id"],
            "question": item["question"],
            "options": item["options"]
        })

    sanitized = {
        "teil": 1,
        "title": data.get("title", "Lesen Teil 1: Zeitungsartikel"),
        "text": data["text"],
        "items": sanitized_items
    }
    return sanitized, {"answer_key": answer_key, "explanations": explanations}


async def generate_lesen_teil2(level: str = "A2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Lesen Teil 2 (Kaufhaus Info Board). Returns (sanitized_teil, answer_key)."""
    try:
        template = load_prompt("exam_lesen_teil2.txt")
        raw = await asyncio.to_thread(generate, template, max_tokens=650, temperature=0.6)
        parsed = _extract_json(raw)
        if parsed and "directory" in parsed and "items" in parsed and len(parsed["items"]) == 5:
            data = parsed
        else:
            data = FALLBACK_LESEN_TEIL2
    except Exception as e:
        logger.warning("Error generating Lesen Teil 2, using fallback: %s", e)
        data = FALLBACK_LESEN_TEIL2

    answer_key = {str(item["id"]): item.get("answer_key", "a") for item in data["items"]}
    explanations = {str(item["id"]): item.get("explanation", "") for item in data["items"]}

    sanitized_items = []
    for item in data["items"]:
        sanitized_items.append({
            "id": item["id"],
            "question": item["question"],
            "options": item["options"]
        })

    sanitized = {
        "teil": 2,
        "title": data.get("title", "Lesen Teil 2: Kaufhaus-Wegweiser"),
        "directory": data["directory"],
        "items": sanitized_items
    }
    return sanitized, {"answer_key": answer_key, "explanations": explanations}


async def generate_lesen_teil3(level: str = "A2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Lesen Teil 3 (Personal Email). Returns (sanitized_teil, answer_key)."""
    try:
        template = load_prompt("exam_lesen_teil3.txt")
        raw = await asyncio.to_thread(generate, template, max_tokens=600, temperature=0.6)
        parsed = _extract_json(raw)
        if parsed and "text" in parsed and "items" in parsed and len(parsed["items"]) == 5:
            data = parsed
        else:
            data = FALLBACK_LESEN_TEIL3
    except Exception as e:
        logger.warning("Error generating Lesen Teil 3, using fallback: %s", e)
        data = FALLBACK_LESEN_TEIL3

    answer_key = {str(item["id"]): item.get("answer_key", "a") for item in data["items"]}
    explanations = {str(item["id"]): item.get("explanation", "") for item in data["items"]}

    sanitized_items = []
    for item in data["items"]:
        sanitized_items.append({
            "id": item["id"],
            "question": item["question"],
            "options": item["options"]
        })

    sanitized = {
        "teil": 3,
        "title": "Lesen Teil 3: E-Mail / Brief",
        "sender": data.get("sender", "Anna"),
        "recipient": data.get("recipient", "Freund"),
        "subject": data.get("subject", "Neuigkeiten"),
        "text": data["text"],
        "items": sanitized_items
    }
    return sanitized, {"answer_key": answer_key, "explanations": explanations}


async def generate_lesen_teil4(level: str = "A2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Lesen Teil 4 (6 Ads + 5 People matching). Returns (sanitized_teil, answer_key)."""
    try:
        template = load_prompt("exam_lesen_teil4.txt")
        raw = await asyncio.to_thread(generate, template, max_tokens=750, temperature=0.6)
        parsed = _extract_json(raw)
        if parsed and "ads" in parsed and "items" in parsed and len(parsed["ads"]) >= 6 and len(parsed["items"]) == 5:
            data = parsed
        else:
            data = FALLBACK_LESEN_TEIL4
    except Exception as e:
        logger.warning("Error generating Lesen Teil 4, using fallback: %s", e)
        data = FALLBACK_LESEN_TEIL4

    answer_key = {str(item["id"]): item.get("answer_key", "x") for item in data["items"]}
    explanations = {str(item["id"]): item.get("explanation", "") for item in data["items"]}

    sanitized_items = []
    for item in data["items"]:
        sanitized_items.append({
            "id": item["id"],
            "person": item["person"]
        })

    sanitized = {
        "teil": 4,
        "title": data.get("title", "Lesen Teil 4: Anzeigen und Personen"),
        "instructions": data.get("instructions", "Finden Sie für jede Person die passende Anzeige (a-f) oder wählen Sie 'x'."),
        "ads": data["ads"],
        "items": sanitized_items
    }
    return sanitized, {"answer_key": answer_key, "explanations": explanations}


async def generate_schreiben_teil1(level: str = "A2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Schreiben Teil 1 (SMS / Short Note)."""
    try:
        template = load_prompt("exam_schreiben_teil1.txt")
        raw = await asyncio.to_thread(generate, template, max_tokens=400, temperature=0.6)
        parsed = _extract_json(raw)
        if parsed and "bullet_points" in parsed and len(parsed["bullet_points"]) >= 3:
            data = parsed
        else:
            data = FALLBACK_SCHREIBEN_TEIL1
    except Exception as e:
        logger.warning("Error generating Schreiben Teil 1, using fallback: %s", e)
        data = FALLBACK_SCHREIBEN_TEIL1

    return data, {}


async def generate_schreiben_teil2(level: str = "A2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Schreiben Teil 2 (Formal / Semi-formal Email)."""
    try:
        template = load_prompt("exam_schreiben_teil2.txt")
        raw = await asyncio.to_thread(generate, template, max_tokens=450, temperature=0.6)
        parsed = _extract_json(raw)
        if parsed and "bullet_points" in parsed and len(parsed["bullet_points"]) >= 3:
            data = parsed
        else:
            data = FALLBACK_SCHREIBEN_TEIL2
    except Exception as e:
        logger.warning("Error generating Schreiben Teil 2, using fallback: %s", e)
        data = FALLBACK_SCHREIBEN_TEIL2

    return data, {}
