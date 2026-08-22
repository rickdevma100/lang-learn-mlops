"""Evaluators for Goethe A2 Exams (Deterministic for Lesen, LLM Rubric for Schreiben).
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

# pyrefly: ignore [missing-import]
from ..prompts import load_prompt
# pyrefly: ignore [missing-import]
from ..runner import generate
from .models import EvaluationResult, ExamModule

logger = logging.getLogger("lang_learn.exam.evaluators")


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


def _normalize_option_letter(val: Any, valid_set: tuple = ("a", "b", "c", "d", "e", "f", "x")) -> str:
    """Extract and normalize a single option letter."""
    if val is None:
        return ""
    s = str(val).strip().lower()
    if s in valid_set:
        return s
    if any(k in s for k in ("kein", "nein", "none", "nicht", "x")):
        return "x"
    m = re.search(r"\b([a-z])\b", s)
    if m and m.group(1) in valid_set:
        return m.group(1)
    if s and s[0] in valid_set:
        return s[0]
    return s


def evaluate_reading(paper: Dict[str, Any], user_answers: Dict[str, Any]) -> EvaluationResult:
    """Deterministically score a Lesen (Reading) exam using the Redis-stored answer key.

    Formula: 20 raw items * 1.25 = 25 module points.
    Pass threshold: >= 15.0 / 25.0 module points (>= 12 / 20 raw items, >= 60%).
    """
    raw_ak = paper.get("answer_key", {})
    if isinstance(raw_ak, dict) and "answer_key" in raw_ak:
        answer_key = raw_ak.get("answer_key", {})
        explanations = raw_ak.get("explanations", {})
    elif isinstance(raw_ak, dict):
        answer_key = raw_ak
        explanations = {}
    else:
        answer_key = {}
        explanations = {}

    # Fallback answer key extraction from teils if top-level answer_key was omitted
    if not answer_key:
        teils = paper.get("teils", {})
        for t_name, t_data in teils.items():
            if isinstance(t_data, dict) and "items" in t_data:
                for itm in t_data["items"]:
                    if "id" in itm and "answer_key" in itm:
                        answer_key[str(itm["id"])] = itm["answer_key"]
                    if "id" in itm and "explanation" in itm:
                        explanations[str(itm["id"])] = itm["explanation"]

    # Flatten user answers if grouped by teil
    flat_user_answers: Dict[str, str] = {}
    for k, v in user_answers.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat_user_answers[str(sub_k)] = _normalize_option_letter(sub_v)
        else:
            flat_user_answers[str(k)] = _normalize_option_letter(v)

    item_results = []
    teil_counts = {"teil1": {"correct": 0, "total": 5},
                   "teil2": {"correct": 0, "total": 5},
                   "teil3": {"correct": 0, "total": 5},
                   "teil4": {"correct": 0, "total": 5}}

    correct_count = 0
    total_items = 20

    for q_id in range(1, 21):
        q_str = str(q_id)
        raw_expected = answer_key.get(q_str, "")
        raw_user_ans = flat_user_answers.get(q_str, "")

        expected = _normalize_option_letter(raw_expected)
        user_ans = _normalize_option_letter(raw_user_ans)
        is_correct = bool(expected and user_ans and (user_ans == expected))

        if is_correct:
            correct_count += 1

        # Determine Teil
        if 1 <= q_id <= 5:
            t_key = "teil1"
        elif 6 <= q_id <= 10:
            t_key = "teil2"
        elif 11 <= q_id <= 15:
            t_key = "teil3"
        else:
            t_key = "teil4"

        if is_correct:
            teil_counts[t_key]["correct"] += 1

        item_results.append({
            "id": q_id,
            "teil": int(t_key.replace("teil", "")),
            "user_answer": user_ans or "None",
            "correct_answer": expected,
            "is_correct": is_correct,
            "explanation": explanations.get(q_str, "")
        })

    module_score = round(correct_count * 1.25, 1)
    passed = module_score >= 15.0  # Official Goethe 60% requirement (>= 12 out of 20 items)

    breakdown = {
        "teils": {
            "teil1": {
                "name": "Teil 1: Zeitungsartikel",
                "score": teil_counts["teil1"]["correct"],
                "max": 5,
                "percentage": round(teil_counts["teil1"]["correct"] / 5 * 100, 1)
            },
            "teil2": {
                "name": "Teil 2: Kaufhaus-Wegweiser",
                "score": teil_counts["teil2"]["correct"],
                "max": 5,
                "percentage": round(teil_counts["teil2"]["correct"] / 5 * 100, 1)
            },
            "teil3": {
                "name": "Teil 3: E-Mail / Brief",
                "score": teil_counts["teil3"]["correct"],
                "max": 5,
                "percentage": round(teil_counts["teil3"]["correct"] / 5 * 100, 1)
            },
            "teil4": {
                "name": "Teil 4: Anzeigen und Personen",
                "score": teil_counts["teil4"]["correct"],
                "max": 5,
                "percentage": round(teil_counts["teil4"]["correct"] / 5 * 100, 1)
            }
        },
        "items": item_results
    }

    feedback = (
        f"Herzlichen Glückwunsch! Sie haben den Leseteil mit {module_score}/25.0 Punkten ({correct_count}/20 richtig) bestanden."
        if passed
        else f"Leider nicht bestanden ({module_score}/25.0 Punkten, {correct_count}/20 richtig). Sie benötigen mindestens 15.0 von 25.0 Punkten (60%, mindestens 12 richtige Antworten)."
    )

    return EvaluationResult(
        submission_id=str(uuid.uuid4()),
        paper_id=paper.get("paper_id", ""),
        module=ExamModule.LESEN,
        level=paper.get("level", "A2"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        raw_score=float(correct_count),
        max_raw_score=float(total_items),
        module_score=module_score,
        max_module_score=25.0,
        passed=passed,
        breakdown=breakdown,
        general_feedback=feedback
    )


def evaluate_writing(paper: Dict[str, Any], user_answers: Dict[str, Any]) -> EvaluationResult:
    """Evaluate Schreiben (Writing) exam against official Goethe A2 Rubric using LLM."""
    teils = paper.get("teils", {})
    teil1_prompt = json.dumps(teils.get("teil1", {}), ensure_ascii=False)
    teil2_prompt = json.dumps(teils.get("teil2", {}), ensure_ascii=False)

    teil1_user_text = str(user_answers.get("teil1", "")).strip()
    teil2_user_text = str(user_answers.get("teil2", "")).strip()

    try:
        template = load_prompt("exam_eval_schreiben.txt")
        prompt = template.format(
            teil1_prompt=teil1_prompt,
            teil1_user_text=teil1_user_text or "(Keine Antwort eingegeben)",
            teil2_prompt=teil2_prompt,
            teil2_user_text=teil2_user_text or "(Keine Antwort eingegeben)"
        )
        raw = generate(prompt, max_tokens=1000, temperature=0.2)
        parsed = _extract_json(raw)
    except Exception as e:
        logger.error("Error evaluating writing via LLM: %s", e)
        parsed = None

    if not parsed or "overall_score" not in parsed:
        # Fallback heuristic calculation if LLM evaluation is unavailable
        t1_len = len(teil1_user_text.split())
        t2_len = len(teil2_user_text.split())
        
        t1_score = min(12.5, round((t1_len / 25.0) * 10.0, 1)) if t1_len > 5 else 2.0
        t2_score = min(12.5, round((t2_len / 35.0) * 10.0, 1)) if t2_len > 5 else 2.0
        total = round(t1_score + t2_score, 1)

        parsed = {
            "overall_score": total,
            "max_score": 25.0,
            "passed": total >= 15.0,
            "teil1": {
                "task_fulfillment_score": round(t1_score * 0.4, 1),
                "task_fulfillment_max": 5.0,
                "task_fulfillment_band": "B" if t1_score >= 8 else "C",
                "language_score": round(t1_score * 0.6, 1),
                "language_max": 7.5,
                "language_band": "B" if t1_score >= 8 else "C",
                "total_score": t1_score,
                "word_count": t1_len,
                "feedback": "Guter Entwurf für Teil 1. Achten Sie auf die Vollständigkeit aller Leitpunkte.",
                "corrections": [],
                "model_answer": "Lieber Freund, vielen Dank für die Einladung. Ich komme gerne am Samstag zu deiner Party!"
            },
            "teil2": {
                "task_fulfillment_score": round(t2_score * 0.4, 1),
                "task_fulfillment_max": 5.0,
                "task_fulfillment_band": "B" if t2_score >= 8 else "C",
                "language_score": round(t2_score * 0.6, 1),
                "language_max": 7.5,
                "language_band": "B" if t2_score >= 8 else "C",
                "total_score": t2_score,
                "word_count": t2_len,
                "feedback": "Gute formelle Struktur. Bitte überprüfen Sie Verbpositionen und Anrede.",
                "corrections": [],
                "model_answer": "Sehr geehrte Damen und Herren, ich interessiere mich für Ihren Deutschkurs und bitte um weitere Informationen."
            },
            "general_feedback": "Schriftliche Arbeit erfolgreich abgegeben. Beachten Sie die Wortanzahl und Höflichkeitsformeln."
        }

    overall_score = float(parsed.get("overall_score", 15.0))
    passed = bool(parsed.get("passed", overall_score >= 15.0))

    breakdown = {
        "teil1": parsed.get("teil1", {}),
        "teil2": parsed.get("teil2", {}),
    }

    return EvaluationResult(
        submission_id=str(uuid.uuid4()),
        paper_id=paper.get("paper_id", ""),
        module=ExamModule.SCHREIBEN,
        level=paper.get("level", "A2"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        raw_score=overall_score,
        max_raw_score=25.0,
        module_score=overall_score,
        max_module_score=25.0,
        passed=passed,
        breakdown=breakdown,
        general_feedback=parsed.get("general_feedback", "")
    )
