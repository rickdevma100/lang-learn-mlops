"""Async generators for Goethe A2 Exam Teile (Lesen & Schreiben).

Hybrid architecture:
- Lesen: Certified pool texts from Redis + LLM-generated questions (2B model)
- Schreiben: LLM-only generation with fallback pool
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
from .text_pool import TextPool

logger = logging.getLogger("lang_learn.exam.generators")

# Singleton text pool instance (initialized on first use)
_text_pool: TextPool | None = None


def _get_text_pool() -> TextPool:
    """Get or create the text pool singleton."""
    global _text_pool
    if _text_pool is None:
        _text_pool = TextPool()
        # Seed pool on first access
        stats = _text_pool.seed_pool()
        logger.info("Text pool initialized: %s", stats)
    return _text_pool


# Ring buffer for Schreiben pools (kept for backward compat)
_recent_pool_history: Dict[str, collections.deque] = {
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


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _repair_json_string(text: str) -> str:
    """Repair common JSON formatting errors from small LLMs."""
    text = re.sub(
        r':\s*(?!")([A-ZÄÖÜa-zäöüß][^",}\]]*?")',
        r': "\1',
        text
    )
    text = re.sub(r',\s*([\}\]])', r'\1', text)
    return text


def _try_parse_json(text: str) -> Any | None:
    """Attempt to parse text as JSON, returning None on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_json(text: str) -> Any | None:
    """Robustly extract and parse JSON from LLM generation output.

    Handles markdown fences, missing quotes, truncated output, trailing commas.
    Returns either a dict or list depending on what was parsed.
    """
    if not text or not text.strip():
        return None

    cleaned = text.strip()
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
        return result

    # 3. Extract JSON array if present
    arr_match = re.search(r"(\[.*\])", repaired, re.DOTALL)
    if arr_match:
        result = _try_parse_json(arr_match.group(1))
        if result:
            return result

    # 4. Extract JSON object if present
    obj_match = re.search(r"(\{.*\})", repaired, re.DOTALL)
    if obj_match:
        result = _try_parse_json(obj_match.group(1))
        if result:
            return result

    # 5. Handle truncated output: find start, close brackets
    for start_char in ("[", "{"):
        start_idx = repaired.find(start_char)
        if start_idx == -1:
            continue
        truncated = repaired[start_idx:]
        lines = truncated.split('\n')
        for trim_count in range(min(len(lines), 12)):
            candidate_lines = lines[:len(lines) - trim_count] if trim_count > 0 else lines
            candidate = '\n'.join(candidate_lines)
            candidate = re.sub(r',\s*$', '', candidate.rstrip())
            open_braces = candidate.count('{') - candidate.count('}')
            open_brackets = candidate.count('[') - candidate.count(']')
            suffix = ']' * max(0, open_brackets) + '}' * max(0, open_braces)
            if suffix or trim_count > 0:
                attempt = candidate + suffix
                result = _try_parse_json(attempt)
                if result:
                    logger.info("JSON recovered from truncated output (trimmed %d lines)", trim_count)
                    return result
                attempt_repaired = _repair_json_string(attempt)
                result = _try_parse_json(attempt_repaired)
                if result:
                    return result

    logger.warning("All JSON extraction attempts failed. Output length: %d chars", len(text))
    return None


def _normalize_option_letter(val: Any, valid_set: tuple = ("a", "b", "c")) -> str:
    """Extract and normalize a single option letter."""
    if val is None:
        return valid_set[0]
    s = str(val).strip().lower()
    if s in valid_set:
        return s
    if "x" in valid_set and any(k in s for k in ("kein", "nein", "none", "nicht", "x")):
        return "x"
    m = re.search(r"\b([a-z])\b", s)
    if m and m.group(1) in valid_set:
        return m.group(1)
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
        for k in ("a", "b", "c"):
            if k not in normalized:
                normalized[k] = f"Option {k.upper()}"
        return normalized
    return {"a": "Option A", "b": "Option B", "c": "Option C"}


# ---------------------------------------------------------------------------
# LLM Question Generation (shared by all Lesen generators)
# ---------------------------------------------------------------------------

async def _generate_questions_for_text(
    text: str,
    teil_type: str,
    num_questions: int = 5,
    id_start: int = 1,
    max_retries: int = 3,
) -> List[Dict[str, Any]] | None:
    """Generate multiple-choice questions for a given text using the 2B LLM.
    
    Args:
        text: The German text to generate questions about.
        teil_type: Description like 'newspaper article' or 'email' for prompt context.
        num_questions: Number of questions to generate.
        id_start: Starting question ID (1 for Teil 1, 6 for Teil 2, etc.)
        max_retries: Number of LLM retries on failure.
    
    Returns:
        List of question dicts, or None if all retries failed.
    """
    id_end = id_start + num_questions - 1
    
    # Build example items for the prompt
    example_items = []
    for i in range(id_start, id_end + 1):
        ans = random.choice(["a", "b", "c"])
        example_items.append(
            f'{{"id":{i},"question":"Frage auf Deutsch?","options":{{"a":"Antwort A","b":"Antwort B","c":"Antwort C"}},"answer_key":"{ans}","explanation":"Begründung aus dem Text."}}'
        )
    example_json = "[" + ",".join(example_items) + "]"
    
    prompt = (
        f"Read this German {teil_type}:\n\n"
        f"{text}\n\n"
        f"Generate {num_questions} multiple-choice reading comprehension questions (numbered {id_start}-{id_end}) about this text.\n"
        "Each question must have exactly 3 options (a, b, c) with ONLY ONE correct answer.\n"
        "The explanation MUST quote the specific words from the text that prove the answer.\n"
        f"Return ONLY a valid JSON array:\n{example_json}"
    )
    
    for attempt in range(1, max_retries + 1):
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(generate, prompt, max_tokens=1024, temperature=0.4),
                timeout=90.0
            )
            logger.info("Question generation attempt %d: %d chars output", attempt, len(raw))
            
            parsed = _extract_json(raw)
            items = None
            
            if isinstance(parsed, list) and len(parsed) >= 3:
                items = parsed
            elif isinstance(parsed, dict):
                if "items" in parsed:
                    items = parsed["items"]
                elif "questions" in parsed:
                    items = parsed["questions"]
            
            if items and len(items) >= 3:
                logger.info("Question generation SUCCESS: %d items on attempt %d", len(items), attempt)
                return items[:num_questions]
            else:
                logger.warning("Question generation attempt %d: parsed but insufficient items (got %d)", 
                             attempt, len(items) if items else 0)
                
        except asyncio.TimeoutError:
            logger.warning("Question generation attempt %d timed out", attempt)
        except Exception as e:
            logger.warning("Question generation attempt %d error: %s", attempt, e)
    
    logger.error("Question generation FAILED after %d attempts", max_retries)
    return None


# ---------------------------------------------------------------------------
# Lesen Generators (Hybrid: Pool Text + LLM Questions)
# ---------------------------------------------------------------------------

async def generate_lesen_teil1(level: str = "A2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Lesen Teil 1 (Newspaper Article + 5 MCQ).
    
    Hybrid: Text from certified pool, questions from 2B LLM.
    """
    pool = _get_text_pool()
    text_data = pool.get_random_text("lesen_teil1")
    
    # Generate fresh questions using LLM
    items_raw = await _generate_questions_for_text(
        text=text_data["text"],
        teil_type="newspaper article",
        num_questions=5,
        id_start=1,
    )
    
    if not items_raw:
        raise RuntimeError("Failed to generate questions for Lesen Teil 1 after all retries")
    
    # Sanitize items
    sanitized_items = []
    answer_key = {}
    explanations = {}
    
    for idx, item in enumerate(items_raw[:5], start=1):
        q_id = idx
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
        "title": text_data.get("title", "Lesen Teil 1: Zeitungsartikel"),
        "text": text_data["text"],
        "items": sanitized_items,
        "source": "hybrid"
    }
    return sanitized, {"answer_key": answer_key, "explanations": explanations}


async def generate_lesen_teil2(level: str = "A2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Lesen Teil 2 (Directory/Floor Guide + 5 MCQ).
    
    Hybrid: Directory from certified pool, questions from 2B LLM.
    """
    pool = _get_text_pool()
    text_data = pool.get_random_text("lesen_teil2")
    
    # Build a text description of the directory for LLM
    directory_text = ""
    for floor in text_data.get("directory", []):
        directory_text += f"{floor['floor']}: {floor['departments']}\n"
    
    items_raw = await _generate_questions_for_text(
        text=directory_text,
        teil_type="department store floor directory",
        num_questions=5,
        id_start=6,
    )
    
    if not items_raw:
        raise RuntimeError("Failed to generate questions for Lesen Teil 2 after all retries")
    
    sanitized_items = []
    answer_key = {}
    explanations = {}
    
    for idx, item in enumerate(items_raw[:5], start=6):
        q_id = idx
        ans = _extract_item_answer(item, default="a", valid_options=("a", "b", "c"))
        exp = str(item.get("explanation") or item.get("reason") or "Richtige Antwort laut Wegweiser.")
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
        "title": text_data.get("title", "Lesen Teil 2: Kaufhaus-Wegweiser"),
        "directory": text_data["directory"],
        "items": sanitized_items,
        "source": "hybrid"
    }
    return sanitized, {"answer_key": answer_key, "explanations": explanations}


async def generate_lesen_teil3(level: str = "A2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Lesen Teil 3 (Personal Email + 5 MCQ).
    
    Hybrid: Email from certified pool, questions from 2B LLM.
    """
    pool = _get_text_pool()
    text_data = pool.get_random_text("lesen_teil3")
    
    items_raw = await _generate_questions_for_text(
        text=text_data["text"],
        teil_type="personal email",
        num_questions=5,
        id_start=11,
    )
    
    if not items_raw:
        raise RuntimeError("Failed to generate questions for Lesen Teil 3 after all retries")
    
    sanitized_items = []
    answer_key = {}
    explanations = {}
    
    for idx, item in enumerate(items_raw[:5], start=11):
        q_id = idx
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
        "title": text_data.get("title", "Lesen Teil 3: E-Mail / Brief"),
        "sender": text_data.get("sender", "Anna"),
        "recipient": text_data.get("recipient", "Freund/Freundin"),
        "subject": text_data.get("subject", "Neuigkeiten"),
        "text": text_data["text"],
        "items": sanitized_items,
        "source": "hybrid"
    }
    return sanitized, {"answer_key": answer_key, "explanations": explanations}


async def generate_lesen_teil4(level: str = "A2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Lesen Teil 4 (Classified Ads + 5 People Matching).
    
    Hybrid: Ads from certified pool, matching questions from 2B LLM.
    """
    pool = _get_text_pool()
    text_data = pool.get_random_text("lesen_teil4")
    
    # Build text description of ads for LLM
    ads_text = ""
    for ad in text_data.get("ads", []):
        ads_text += f"Anzeige {ad['id'].upper()}: {ad['title']} - {ad['text']}\n\n"
    
    # Teil 4 uses a different question format: match people to ads
    prompt = (
        f"Read these 6 classified ads:\n\n{ads_text}\n"
        "Generate 5 questions. Each question describes a person looking for something specific.\n"
        "The person must match EXACTLY ONE ad (a-f), or NO ad (answer 'x').\n"
        "At least 4 questions must match an ad. Maximum 1 question can have 'x' (no match).\n"
        "Return ONLY a valid JSON array:\n"
        '[{"id":16,"question":"Person description in German...","answer_key":"b","explanation":"Why this ad matches."},'
        '{"id":17,"question":"...","answer_key":"d","explanation":"..."},'
        '{"id":18,"question":"...","answer_key":"a","explanation":"..."},'
        '{"id":19,"question":"...","answer_key":"f","explanation":"..."},'
        '{"id":20,"question":"...","answer_key":"x","explanation":"No ad matches because..."}]'
    )
    
    items_raw = None
    for attempt in range(1, 4):
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(generate, prompt, max_tokens=1024, temperature=0.4),
                timeout=90.0
            )
            logger.info("Teil 4 question generation attempt %d: %d chars", attempt, len(raw))
            parsed = _extract_json(raw)
            
            if isinstance(parsed, list) and len(parsed) >= 3:
                items_raw = parsed
                break
            elif isinstance(parsed, dict) and "items" in parsed:
                items_raw = parsed["items"]
                break
        except Exception as e:
            logger.warning("Teil 4 question generation attempt %d: %s", attempt, e)
    
    if not items_raw:
        raise RuntimeError("Failed to generate questions for Lesen Teil 4 after all retries")
    
    sanitized_items = []
    answer_key = {}
    explanations = {}
    
    valid_teil4_options = ("a", "b", "c", "d", "e", "f", "x")
    for idx, item in enumerate(items_raw[:5], start=16):
        q_id = idx
        ans = _extract_item_answer(item, default="a", valid_options=valid_teil4_options)
        exp = str(item.get("explanation") or item.get("reason") or "Passende Anzeige.")
        
        answer_key[str(q_id)] = ans
        explanations[str(q_id)] = exp
        sanitized_items.append({
            "id": q_id,
            "question": str(item.get("question", "")),
        })
    
    sanitized = {
        "teil": 4,
        "title": text_data.get("title", "Lesen Teil 4: Anzeigen & Personen"),
        "ads": text_data["ads"],
        "items": sanitized_items,
        "source": "hybrid"
    }
    return sanitized, {"answer_key": answer_key, "explanations": explanations}


# ---------------------------------------------------------------------------
# Schreiben Generators (LLM-only, unchanged)
# ---------------------------------------------------------------------------

async def generate_schreiben_teil1(level: str = "A2") -> Dict[str, Any]:
    """Generate Schreiben Teil 1 (Informal SMS / Note)."""
    fallback_choice = _pick_distinct_pool_item(POOL_SCHREIBEN_TEIL1, "schreiben_t1")
    source = "fallback"
    try:
        template = load_prompt("exam_schreiben_teil1.txt")
        raw = await asyncio.wait_for(
            asyncio.to_thread(generate, template, max_tokens=512, temperature=0.75),
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
            asyncio.to_thread(generate, template, max_tokens=512, temperature=0.75),
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


# ---------------------------------------------------------------------------
# Schreiben Fallback Pools (kept for Schreiben generators)
# ---------------------------------------------------------------------------

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
