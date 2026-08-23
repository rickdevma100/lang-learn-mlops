"""Async generators for Goethe A2 Exam Teile (Lesen & Schreiben).

Architecture:
- ALL Lesen Teile: Pool text/data + PROGRAMMATIC questions (instant, 100% correct)
- Schreiben: LLM-only with fallback pool

ZERO LLM calls during exam generation for Lesen.
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

# Singleton text pool instance
_text_pool: TextPool | None = None


def _get_text_pool() -> TextPool:
    global _text_pool
    if _text_pool is None:
        _text_pool = TextPool()
        stats = _text_pool.seed_pool()
        logger.info("Text pool initialized: %s", stats)
    return _text_pool


# Schreiben ring buffers
_recent_pool_history: Dict[str, collections.deque] = {
    "schreiben_t1": collections.deque(maxlen=4),
    "schreiben_t2": collections.deque(maxlen=4),
}


def _pick_distinct_pool_item(pool: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    history = _recent_pool_history.setdefault(key, collections.deque(maxlen=max(1, len(pool) - 1)))
    available_indices = [i for i in range(len(pool)) if i not in history]
    if not available_indices:
        history.clear()
        available_indices = list(range(len(pool)))
    choice_idx = random.choice(available_indices)
    history.append(choice_idx)
    return pool[choice_idx]


# ---------------------------------------------------------------------------
# JSON parsing helpers (used by Schreiben only now)
# ---------------------------------------------------------------------------

def _repair_json_string(text: str) -> str:
    text = re.sub(r':\s*(?!")([A-ZÄÖÜa-zäöüß][^",}\]]*?")', r': "\1', text)
    text = re.sub(r',\s*([\}\]])', r'\1', text)
    return text


def _try_parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_json(text: str) -> Any | None:
    if not text or not text.strip():
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json|JSON)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    result = _try_parse_json(cleaned)
    if result:
        return result
    repaired = _repair_json_string(cleaned)
    result = _try_parse_json(repaired)
    if result:
        return result

    arr_match = re.search(r"(\[.*\])", repaired, re.DOTALL)
    if arr_match:
        result = _try_parse_json(arr_match.group(1))
        if result:
            return result
    obj_match = re.search(r"(\{.*\})", repaired, re.DOTALL)
    if obj_match:
        result = _try_parse_json(obj_match.group(1))
        if result:
            return result

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
                    return result
                attempt_repaired = _repair_json_string(attempt)
                result = _try_parse_json(attempt_repaired)
                if result:
                    return result
    return None


# ---------------------------------------------------------------------------
# PROGRAMMATIC Question Generators — NO LLM needed
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Split German text into sentences, filtering out very short ones."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = []
    for s in raw:
        s = s.strip()
        # Skip very short fragments, quotes, greetings
        if len(s) > 25 and not s.startswith("Liebe") and not s.startswith("Bis ") and not s.startswith("Viele Grüße"):
            sentences.append(s)
    return sentences


def _extract_key_facts(text: str) -> List[Dict[str, str]]:
    """Extract factual statements from text for question generation."""
    sentences = _split_sentences(text)
    facts = []
    for sent in sentences:
        # Extract numbers, names, places for factual questions
        has_number = bool(re.search(r'\d+', sent))
        has_quote = '„' in sent or '\"' in sent or '»' in sent
        has_name = bool(re.search(r'[A-ZÄÖÜ][a-zäöüß]{2,}\s+[A-ZÄÖÜ]', sent))
        
        fact = {
            "sentence": sent,
            "has_number": has_number,
            "has_quote": has_quote,
            "has_name": has_name,
            "type": "quote" if has_quote else ("number" if has_number else ("name" if has_name else "general"))
        }
        facts.append(fact)
    return facts


def _generate_wrong_option(correct_sentence: str, all_sentences: List[str]) -> str:
    """Generate a plausible but wrong option based on the correct sentence."""
    # Strategy 1: Pick a sentence from a different part of the text
    other = [s for s in all_sentences if s != correct_sentence]
    if other:
        wrong = random.choice(other)
        # Truncate to roughly same length as correct
        if len(wrong) > 80:
            wrong = wrong[:77] + "..."
        return wrong
    return "Das steht nicht im Text."


def _generate_teil1_questions_programmatic(text: str, title: str, id_start: int = 1) -> List[Dict[str, Any]]:
    """Generate 5 MCQ questions from a newspaper article — FULLY PROGRAMMATIC.
    
    Uses the Goethe A2 format: Richtig / Falsch / Steht nicht im Text.
    Creates statements about the article and asks whether they're correct.
    """
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        # Fallback: split on newlines
        sentences = [s.strip() for s in text.split('\n') if len(s.strip()) > 25]
    
    random.shuffle(sentences)
    
    # Question templates
    templates_richtig = [
        "Im Artikel steht: {statement}",
        "Laut dem Text: {statement}",
        "{statement}",
    ]
    
    # Generate "steht nicht im Text" statements
    not_in_text_statements = [
        f"Der Artikel wurde in einer Fachzeitschrift für Wissenschaft veröffentlicht.",
        f"Der Autor des Artikels lebt seit 20 Jahren in Australien.",
        f"Im Text wird ein neues Gesetz der Europäischen Union beschrieben.",
        f"Der Artikel berichtet über ein Sportereignis in Asien.",
        f"Im Text geht es hauptsächlich um die Geschichte des Mittelalters.",
        f"Der Artikel beschreibt eine neue Methode der Weltraumforschung.",
        f"Im Text wird über ein Musikfestival in Südamerika berichtet.",
        f"Der Artikel handelt von einem Vulkanausbruch auf Island.",
    ]
    
    questions = []
    used_sentences = set()
    
    for i in range(5):
        q_id = id_start + i
        question_type = random.choices(["richtig", "falsch", "nicht_im_text"], weights=[3, 1, 1])[0]
        
        if question_type == "richtig" and len(sentences) > len(used_sentences):
            # Pick an unused sentence — this IS in the text (Richtig)
            available = [s for s in sentences if s not in used_sentences]
            if not available:
                available = sentences
            chosen = random.choice(available)
            used_sentences.add(chosen)
            
            # Truncate for display
            statement = chosen if len(chosen) <= 100 else chosen[:97] + "..."
            template = random.choice(templates_richtig)
            
            correct_answer = "a"
            questions.append({
                "id": q_id,
                "question": template.format(statement=statement),
                "options": {
                    "a": "Richtig",
                    "b": "Falsch",
                    "c": "Steht nicht im Text"
                },
                "answer_key": correct_answer,
                "explanation": f"Diese Aussage steht im Text: \"{chosen[:60]}...\""
            })
            
        elif question_type == "nicht_im_text":
            # Statement NOT in the text
            statement = random.choice(not_in_text_statements)
            not_in_text_statements.remove(statement)
            
            correct_answer = "c"
            questions.append({
                "id": q_id,
                "question": statement,
                "options": {
                    "a": "Richtig",
                    "b": "Falsch",
                    "c": "Steht nicht im Text"
                },
                "answer_key": correct_answer,
                "explanation": "Diese Information steht nicht im Text."
            })
        else:
            # "Falsch" — modify a real sentence to make it wrong
            available = [s for s in sentences if s not in used_sentences]
            if not available:
                available = sentences
            chosen = random.choice(available)
            used_sentences.add(chosen)
            
            # Create a false version by negation or number change
            false_statement = _make_false_statement(chosen)
            
            correct_answer = "b"
            questions.append({
                "id": q_id,
                "question": false_statement,
                "options": {
                    "a": "Richtig",
                    "b": "Falsch",
                    "c": "Steht nicht im Text"
                },
                "answer_key": correct_answer,
                "explanation": f"Falsch. Im Text steht: \"{chosen[:80]}...\""
            })
    
    return questions


def _make_false_statement(sentence: str) -> str:
    """Modify a German sentence to make it factually incorrect."""
    # Strategy 1: Change numbers
    numbers = re.findall(r'\d+', sentence)
    if numbers:
        num = random.choice(numbers)
        wrong_num = str(int(num) * 2 + 7)  # Make it clearly different
        return sentence.replace(num, wrong_num, 1)
    
    # Strategy 2: Add "nicht" or remove it
    if " nicht " in sentence:
        return sentence.replace(" nicht ", " ", 1)
    
    # Strategy 3: Change positive to negative
    replacements = [
        ("gut", "schlecht"), ("viele", "wenige"), ("groß", "klein"),
        ("schnell", "langsam"), ("beliebt", "unbeliebt"), ("mehr", "weniger"),
        ("gern", "ungern"), ("oft", "selten"), ("neu", "alt"),
        ("teuer", "billig"), ("wichtig", "unwichtig"),
    ]
    for pos, neg in replacements:
        if pos in sentence.lower():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(pos), re.IGNORECASE)
            return pattern.sub(neg, sentence, count=1)
    
    # Strategy 4: Add "nicht" before the verb area
    words = sentence.split()
    if len(words) > 4:
        insert_pos = min(3, len(words) - 1)
        words.insert(insert_pos, "nicht")
        return " ".join(words)
    
    return sentence + " Das stimmt nicht."


def _generate_teil2_questions_programmatic(directory: List[Dict[str, Any]], id_start: int = 6) -> List[Dict[str, Any]]:
    """Generate 5 MCQ from a floor directory — FULLY PROGRAMMATIC, 100% correct."""
    all_items = []
    all_floors = []
    for floor_info in directory:
        floor_name = floor_info["floor"]
        all_floors.append(floor_name)
        departments = [d.strip() for d in floor_info["departments"].split(",")]
        for dept in departments:
            if dept and len(dept) > 2:
                all_items.append((dept, floor_name))

    random.shuffle(all_items)

    selected = []
    used_floors = set()
    for dept, floor in all_items:
        if floor not in used_floors or len(selected) < 5:
            selected.append((dept, floor))
            used_floors.add(floor)
        if len(selected) >= 5:
            break
    if len(selected) < 5:
        for dept, floor in all_items:
            if (dept, floor) not in selected:
                selected.append((dept, floor))
            if len(selected) >= 5:
                break

    scenarios = [
        "Sie möchten {} kaufen. Wohin gehen Sie?",
        "Sie suchen {}. In welchem Stock finden Sie das?",
        "Ihr Freund braucht {}. Wo im Kaufhaus finden Sie das?",
        "Sie möchten sich {} ansehen. Wohin müssen Sie gehen?",
        "Eine Kundin fragt nach {}. In welchem Stock ist das?",
    ]

    questions = []
    for i, (dept, correct_floor) in enumerate(selected[:5]):
        q_id = id_start + i
        scenario = scenarios[i % len(scenarios)]
        wrong_floors = [f for f in all_floors if f != correct_floor]
        random.shuffle(wrong_floors)
        options_list = [correct_floor] + wrong_floors[:2]
        random.shuffle(options_list)
        correct_letter = chr(ord('a') + options_list.index(correct_floor))

        questions.append({
            "id": q_id,
            "question": scenario.format(dept),
            "options": {"a": options_list[0], "b": options_list[1], "c": options_list[2]},
            "answer_key": correct_letter,
            "explanation": f"{dept} befindet sich im {correct_floor}."
        })
    return questions


def _generate_teil3_questions_programmatic(text: str, sender: str, recipient: str, subject: str, id_start: int = 11) -> List[Dict[str, Any]]:
    """Generate 5 MCQ from a personal email — FULLY PROGRAMMATIC.
    
    Creates comprehension questions about the email content.
    Uses Richtig / Falsch / Steht nicht im Text format (authentic Goethe A2).
    """
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        sentences = [s.strip() for s in text.split('\n') if len(s.strip()) > 20]
    
    random.shuffle(sentences)
    
    # Email-specific "not in text" statements
    not_in_text = [
        f"{sender} hat einen neuen Job in einer Bank gefunden.",
        f"{sender} plant eine Reise nach Japan im nächsten Sommer.",
        f"{sender} hat letzte Woche geheiratet.",
        f"{sender} studiert jetzt Medizin an der Universität.",
        f"In der E-Mail geht es um einen Autounfall.",
        f"{sender} bittet um Geld für eine neue Wohnung.",
        f"{sender} hat ein Haustier (eine Katze) gekauft.",
        f"{sender} möchte in ein anderes Land umziehen.",
    ]
    
    questions = []
    used_sentences = set()
    
    for i in range(5):
        q_id = id_start + i
        
        if i == 0:
            # First question: about the purpose of the email
            questions.append({
                "id": q_id,
                "question": f"Warum schreibt {sender} diese E-Mail?",
                "options": {
                    "a": f"{sender} möchte Neuigkeiten erzählen.",
                    "b": f"{sender} braucht dringend Hilfe bei einem Problem.",
                    "c": f"{sender} möchte sich über etwas beschweren."
                },
                "answer_key": "a",
                "explanation": f"{sender} schreibt, um Neuigkeiten und persönliche Erfahrungen mitzuteilen."
            })
        elif i < 4 and len(sentences) > len(used_sentences):
            # Middle questions: Richtig/Falsch about specific facts
            available = [s for s in sentences if s not in used_sentences]
            if not available:
                available = sentences
            chosen = random.choice(available)
            used_sentences.add(chosen)
            
            # Alternate between Richtig and Falsch
            if i % 2 == 1:
                statement = chosen if len(chosen) <= 90 else chosen[:87] + "..."
                questions.append({
                    "id": q_id,
                    "question": statement,
                    "options": {"a": "Richtig", "b": "Falsch", "c": "Steht nicht im Text"},
                    "answer_key": "a",
                    "explanation": f"Diese Aussage steht in der E-Mail: \"{chosen[:60]}...\""
                })
            else:
                false_statement = _make_false_statement(chosen)
                questions.append({
                    "id": q_id,
                    "question": false_statement,
                    "options": {"a": "Richtig", "b": "Falsch", "c": "Steht nicht im Text"},
                    "answer_key": "b",
                    "explanation": f"Falsch. In der E-Mail steht: \"{chosen[:60]}...\""
                })
        else:
            # Last question: not in text
            stmt = random.choice(not_in_text)
            not_in_text.remove(stmt)
            questions.append({
                "id": q_id,
                "question": stmt,
                "options": {"a": "Richtig", "b": "Falsch", "c": "Steht nicht im Text"},
                "answer_key": "c",
                "explanation": "Diese Information steht nicht in der E-Mail."
            })
    
    return questions


def _generate_teil4_questions_programmatic(ads: List[Dict[str, Any]], id_start: int = 16) -> List[Dict[str, Any]]:
    """Generate 5 person-matching questions from classified ads — FULLY PROGRAMMATIC."""
    _KEYWORD_SCENARIOS = [
        (["Kinder", "Kind", "Spielplatz", "Spielsachen", "Geburtstag", "Spielen", "spielen"],
         "{name} möchte mit seinen Kindern einen schönen Nachmittag verbringen und sucht einen Ort mit Spielmöglichkeiten."),
        (["Kurs", "lernen", "Unterricht", "Schule", "Sprachkurs", "Training"],
         "{name} möchte einen Kurs besuchen und sucht Informationen über Termine und Preise."),
        (["Wochenende", "Samstag", "Sonntag", "Frühstück"],
         "{name} sucht ein Lokal, das am Wochenende ein gutes Frühstück anbietet."),
        (["Reparatur", "kaputt", "Notdienst", "reparieren", "Werkstatt"],
         "{name} braucht eine Reparatur und sucht einen zuverlässigen Service."),
        (["Essen", "Restaurant", "Küche", "kochen", "Menü", "Spezialitäten"],
         "{name} möchte mit Freunden in einem guten Restaurant essen gehen."),
        (["Sport", "Fitness", "Training", "Schwimmen", "Fahrrad"],
         "{name} möchte regelmäßig Sport machen und sucht ein passendes Angebot."),
        (["Musik", "Konzert", "Party", "Live", "Veranstaltung", "Abend"],
         "{name} möchte abends ausgehen und sucht eine Veranstaltung mit Unterhaltung."),
        (["Hochzeit", "Feier", "feiern", "Fest", "Gäste"],
         "{name} plant eine große Feier und sucht einen passenden Ort."),
        (["Liefern", "liefern", "bestellen", "Catering", "Lieferung"],
         "{name} möchte Essen für eine private Feier bestellen."),
        (["Garten", "Terrasse", "draußen", "Sonnenterrasse", "Natur"],
         "{name} möchte bei schönem Wetter draußen sitzen und etwas essen."),
        (["Kuchen", "Torte", "Eis", "Café", "Kaffee"],
         "{name} möchte Kuchen essen und Kaffee trinken gehen."),
        (["Tiere", "Hund", "Katze", "Tierarzt", "Haustier"],
         "{name} sucht Hilfe für sein Haustier."),
        (["Markt", "einkaufen", "Supermarkt", "Geschäft"],
         "{name} möchte frische Lebensmittel einkaufen."),
        (["Ausflug", "Urlaub", "Reise", "Ausflugsziel"],
         "{name} plant einen Tagesausflug mit der Familie."),
    ]

    _NAMES = ["Peter", "Maria", "Thomas", "Sarah", "Jens", "Laura", "Markus",
              "Petra", "Stefan", "Anna", "Karsten", "Gabriele", "Hans", "Sophie"]

    used_names = set()
    questions = []

    shuffled_ads = list(ads)
    random.shuffle(shuffled_ads)

    for i in range(min(4, len(shuffled_ads))):
        ad = shuffled_ads[i]
        ad_id = ad["id"].lower()
        ad_text_lower = ad["text"].lower()
        ad_title = ad.get("title", "")

        best_scenario = None
        for keywords, template in _KEYWORD_SCENARIOS:
            if any(kw.lower() in ad_text_lower or kw.lower() in ad_title.lower() for kw in keywords):
                best_scenario = template
                break
        if not best_scenario:
            best_scenario = "{name} sucht Informationen über das Angebot von " + ad_title + "."

        name = random.choice([n for n in _NAMES if n not in used_names])
        used_names.add(name)

        q_id = id_start + len(questions)
        questions.append({
            "id": q_id,
            "question": best_scenario.format(name=name),
            "answer_key": ad_id,
            "explanation": f"Anzeige {ad_id.upper()} ({ad_title}) passt: {ad['text'][:80]}..."
        })

    # 1 "no match" question
    name = random.choice([n for n in _NAMES if n not in used_names])
    no_match = [
        f"{name} sucht einen Zahnarzt, der auch am Abend Termine hat.",
        f"{name} möchte privaten Klavierunterricht nehmen.",
        f"{name} braucht einen Anwalt für Mietrecht.",
        f"{name} sucht eine Tagesmutter für sein zweijähriges Kind.",
        f"{name} möchte Japanisch lernen.",
        f"{name} sucht eine Reinigungsfirma für sein Büro.",
    ]
    q_id = id_start + len(questions)
    questions.append({
        "id": q_id,
        "question": random.choice(no_match),
        "answer_key": "x",
        "explanation": "Keine der Anzeigen passt zu dieser Person."
    })

    random.shuffle(questions)
    for i, q in enumerate(questions):
        q["id"] = id_start + i
    return questions[:5]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_option_letter(val: Any, valid_set: tuple = ("a", "b", "c")) -> str:
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
    raw = (
        item.get("answer_key") or item.get("answer") or item.get("correct_answer")
        or item.get("solution") or item.get("correct_option") or item.get("correct") or default
    )
    return _normalize_option_letter(raw, valid_set=valid_options)


def _normalize_options_dict(raw_opts: Any) -> Dict[str, str]:
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


def _sanitize_mcq_items(items_raw: List[Dict], id_start: int, count: int = 5,
                         valid_options: tuple = ("a", "b", "c")) -> Tuple[List[Dict], Dict, Dict]:
    sanitized_items = []
    answer_key = {}
    explanations = {}
    for idx, item in enumerate(items_raw[:count], start=id_start):
        q_id = idx
        ans = _extract_item_answer(item, default="a", valid_options=valid_options)
        exp = str(item.get("explanation") or item.get("reason") or "Richtige Antwort.")
        if valid_options == ("a", "b", "c"):
            opts = _normalize_options_dict(item.get("options", {}))
            sanitized_items.append({"id": q_id, "question": str(item.get("question", "")), "options": opts})
        else:
            sanitized_items.append({"id": q_id, "question": str(item.get("question", ""))})
        answer_key[str(q_id)] = ans
        explanations[str(q_id)] = exp
    return sanitized_items, answer_key, explanations


# ---------------------------------------------------------------------------
# Lesen Generators — ALL PROGRAMMATIC (zero LLM calls)
# ---------------------------------------------------------------------------

async def generate_lesen_teil1(level: str = "A2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Lesen Teil 1 — Newspaper Article + 5 MCQ. FULLY PROGRAMMATIC."""
    pool = _get_text_pool()
    text_data = pool.get_random_text("lesen_teil1")

    items_raw = _generate_teil1_questions_programmatic(
        text=text_data["text"],
        title=text_data.get("title", ""),
        id_start=1,
    )
    sanitized_items, answer_key, explanations = _sanitize_mcq_items(items_raw, id_start=1)

    sanitized = {
        "teil": 1,
        "title": text_data.get("title", "Lesen Teil 1: Zeitungsartikel"),
        "text": text_data["text"],
        "items": sanitized_items,
        "source": "programmatic"
    }
    return sanitized, {"answer_key": answer_key, "explanations": explanations}


async def generate_lesen_teil2(level: str = "A2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Lesen Teil 2 — Floor Directory + 5 MCQ. FULLY PROGRAMMATIC."""
    pool = _get_text_pool()
    text_data = pool.get_random_text("lesen_teil2")

    items_raw = _generate_teil2_questions_programmatic(text_data["directory"], id_start=6)
    sanitized_items, answer_key, explanations = _sanitize_mcq_items(items_raw, id_start=6)

    sanitized = {
        "teil": 2,
        "title": text_data.get("title", "Lesen Teil 2: Kaufhaus-Wegweiser"),
        "directory": text_data["directory"],
        "items": sanitized_items,
        "source": "programmatic"
    }
    return sanitized, {"answer_key": answer_key, "explanations": explanations}


async def generate_lesen_teil3(level: str = "A2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Lesen Teil 3 — Personal Email + 5 MCQ. FULLY PROGRAMMATIC."""
    pool = _get_text_pool()
    text_data = pool.get_random_text("lesen_teil3")

    items_raw = _generate_teil3_questions_programmatic(
        text=text_data["text"],
        sender=text_data.get("sender", "Anna"),
        recipient=text_data.get("recipient", "Freund/in"),
        subject=text_data.get("subject", "Neuigkeiten"),
        id_start=11,
    )
    sanitized_items, answer_key, explanations = _sanitize_mcq_items(items_raw, id_start=11)

    sanitized = {
        "teil": 3,
        "title": text_data.get("title", "Lesen Teil 3: E-Mail / Brief"),
        "sender": text_data.get("sender", "Anna"),
        "recipient": text_data.get("recipient", "Freund/Freundin"),
        "subject": text_data.get("subject", "Neuigkeiten"),
        "text": text_data["text"],
        "items": sanitized_items,
        "source": "programmatic"
    }
    return sanitized, {"answer_key": answer_key, "explanations": explanations}


async def generate_lesen_teil4(level: str = "A2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Lesen Teil 4 — Classified Ads + 5 Person Matching. FULLY PROGRAMMATIC."""
    pool = _get_text_pool()
    text_data = pool.get_random_text("lesen_teil4")

    items_raw = _generate_teil4_questions_programmatic(text_data["ads"], id_start=16)
    valid_options = ("a", "b", "c", "d", "e", "f", "x")
    sanitized_items, answer_key, explanations = _sanitize_mcq_items(
        items_raw, id_start=16, valid_options=valid_options
    )

    sanitized = {
        "teil": 4,
        "title": text_data.get("title", "Lesen Teil 4: Anzeigen & Personen"),
        "ads": text_data["ads"],
        "items": sanitized_items,
        "source": "programmatic"
    }
    return sanitized, {"answer_key": answer_key, "explanations": explanations}


# ---------------------------------------------------------------------------
# Schreiben Generators (LLM-only, unchanged)
# ---------------------------------------------------------------------------

async def generate_schreiben_teil1(level: str = "A2") -> Dict[str, Any]:
    fallback_choice = _pick_distinct_pool_item(POOL_SCHREIBEN_TEIL1, "schreiben_t1")
    try:
        template = load_prompt("exam_schreiben_teil1.txt")
        raw = await asyncio.wait_for(
            asyncio.to_thread(generate, template, max_tokens=512, temperature=0.75),
            timeout=90.0
        )
        parsed = _extract_json(raw)
        if parsed and "scenario_german" in parsed and "bullet_points" in parsed and len(parsed["bullet_points"]) >= 2:
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
    fallback_choice = _pick_distinct_pool_item(POOL_SCHREIBEN_TEIL2, "schreiben_t2")
    try:
        template = load_prompt("exam_schreiben_teil2.txt")
        raw = await asyncio.wait_for(
            asyncio.to_thread(generate, template, max_tokens=512, temperature=0.75),
            timeout=90.0
        )
        parsed = _extract_json(raw)
        if parsed and "scenario_german" in parsed and "bullet_points" in parsed and len(parsed["bullet_points"]) >= 2:
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
# Schreiben Fallback Pools
# ---------------------------------------------------------------------------

POOL_SCHREIBEN_TEIL1: List[Dict[str, Any]] = [
    {
        "teil": 1,
        "title": "Schreiben Teil 1: Verspätung ankündigen",
        "scenario_german": "Sie haben sich heute Abend mit Ihrem Freund Michael im Kino verabredet, können aber nicht pünktlich sein.",
        "instructions_german": "Schreiben Sie eine kurze Nachricht an Michael (ca. 20–30 Wörter). Schreiben Sie zu allen drei Punkten:",
        "bullet_points": ["Entschuldigen Sie sich für die Verspätung.", "Nennen Sie den Grund.", "Schlagen Sie einen neuen Treffpunkt oder eine neue Uhrzeit vor."],
        "target_word_count": "20–30 Wörter",
        "tips_english": "Write a short SMS/note (approx. 20-30 words). Address all 3 bullet points."
    },
    {
        "teil": 1,
        "title": "Schreiben Teil 1: Einladung ablehnen",
        "scenario_german": "Ihre Kollegin Maria hat Sie zum Abendessen am Freitag eingeladen. Sie haben aber leider keine Zeit.",
        "instructions_german": "Schreiben Sie eine kurze Nachricht an Maria (ca. 20–30 Wörter). Schreiben Sie zu allen drei Punkten:",
        "bullet_points": ["Bedanken Sie sich für die Einladung.", "Erklären Sie, warum Sie nicht kommen können.", "Schlagen Sie ein Treffen am nächsten Wochenende vor."],
        "target_word_count": "20–30 Wörter",
        "tips_english": "Write a short note (approx. 20-30 words). Thank, give reason, propose alternative."
    },
    {
        "teil": 1,
        "title": "Schreiben Teil 1: Sporttraining absagen",
        "scenario_german": "Sie trainieren regelmäßig mit Ihrem Freund Lukas im Fitnessstudio, sind heute aber krank.",
        "instructions_german": "Schreiben Sie eine Nachricht an Lukas (ca. 20–30 Wörter). Schreiben Sie zu allen drei Punkten:",
        "bullet_points": ["Sagen Sie das Training für heute ab.", "Erklären Sie kurz Ihren Grund.", "Vereinbaren Sie einen neuen Termin."],
        "target_word_count": "20–30 Wörter",
        "tips_english": "Write an informal note (20-30 words) cancelling training, stating why, and rescheduling."
    }
]

POOL_SCHREIBEN_TEIL2: List[Dict[str, Any]] = [
    {
        "teil": 2,
        "title": "Schreiben Teil 2: Sprachkurs anfragen",
        "scenario_german": "Sie möchten im nächsten Monat einen Deutschkurs (Stufe B1) besuchen. Schreiben Sie an die Sprachschule.",
        "instructions_german": "Schreiben Sie eine formelle E-Mail an Frau Weber (ca. 30–40 Wörter). Schreiben Sie zu allen vier Punkten:",
        "bullet_points": ["Grund für Ihr Schreiben nennen", "Nach Kurstermin und Beginn fragen", "Nach Kosten und Unterkünften fragen", "Formelle Anrede und Grußformel"],
        "target_word_count": "30–40 Wörter",
        "tips_english": "Write a formal email (30-40 words). Address all points with formal greetings."
    },
    {
        "teil": 2,
        "title": "Schreiben Teil 2: Zimmerreservierung",
        "scenario_german": "Sie möchten mit Ihrer Familie zwei Zimmer im Hotel 'Alpenblick' buchen.",
        "instructions_german": "Schreiben Sie eine formelle E-Mail an das Hotel (ca. 30–40 Wörter). Schreiben Sie zu allen vier Punkten:",
        "bullet_points": ["Ankunftstag und Personenzahl nennen", "Nach Frühstück und Zimmerpreisen fragen", "Nach Parkplätzen fragen", "Höfliche formelle Schlussformel"],
        "target_word_count": "30–40 Wörter",
        "tips_english": "Write a formal hotel reservation email."
    },
    {
        "teil": 2,
        "title": "Schreiben Teil 2: Wohnungsbesichtigung",
        "scenario_german": "Sie haben eine Anzeige für eine 2-Zimmer-Wohnung gesehen und möchten die Wohnung besichtigen.",
        "instructions_german": "Schreiben Sie eine E-Mail an den Vermieter Herrn Müller (ca. 30–40 Wörter). Schreiben Sie zu allen vier Punkten:",
        "bullet_points": ["Sich kurz vorstellen", "Interesse an der Wohnung bekunden", "Nach einem Besichtigungstermin fragen", "Formelle Grußformel verwenden"],
        "target_word_count": "30–40 Wörter",
        "tips_english": "Write a formal apartment inquiry email."
    }
]
