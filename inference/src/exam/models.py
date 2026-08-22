"""Data models for the Goethe-Zertifikat A2 Exam Suite.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExamModule(str, Enum):
    LESEN = "lesen"
    SCHREIBEN = "schreiben"


# ---------------------------------------------------------------------------
# Lesen (Reading) Models
# ---------------------------------------------------------------------------

class LesenQuestion(BaseModel):
    id: int
    question: str
    options: Dict[str, str]  # "a", "b", "c" -> text
    answer_key: Optional[str] = None  # stripped for frontend
    explanation: Optional[str] = None


class DirectoryFloor(BaseModel):
    floor: str
    departments: str


class ClassifiedAd(BaseModel):
    id: str  # "a" .. "f"
    title: str
    text: str


class AdPerson(BaseModel):
    id: int  # 16 .. 20
    person: str
    answer_key: Optional[str] = None  # "a".."f" or "x", stripped for frontend
    explanation: Optional[str] = None


class LesenTeil1(BaseModel):
    teil: int = 1
    title: str
    text: str
    items: List[LesenQuestion]


class LesenTeil2(BaseModel):
    teil: int = 2
    title: str
    directory: List[DirectoryFloor]
    items: List[LesenQuestion]


class LesenTeil3(BaseModel):
    teil: int = 3
    sender: str
    recipient: str
    subject: str
    text: str
    items: List[LesenQuestion]


class LesenTeil4(BaseModel):
    teil: int = 4
    title: str
    instructions: str
    ads: List[ClassifiedAd]
    items: List[AdPerson]


# ---------------------------------------------------------------------------
# Schreiben (Writing) Models
# ---------------------------------------------------------------------------

class SchreibenTeil1(BaseModel):
    teil: int = 1
    title: str
    scenario_german: str
    instructions_german: str
    bullet_points: List[str]
    target_word_count: str = "20–30 Wörter"
    tips_english: str = ""


class SchreibenTeil2(BaseModel):
    teil: int = 2
    title: str
    scenario_german: str
    instructions_german: str
    bullet_points: List[str]
    target_word_count: str = "30–40 Wörter"
    tips_english: str = ""


# ---------------------------------------------------------------------------
# Complete Exam Paper Models
# ---------------------------------------------------------------------------

class ExamPaper(BaseModel):
    paper_id: str
    label: Optional[str] = None
    module: ExamModule
    level: str = "A2"
    created_at: str
    duration_minutes: int = 30
    total_points: float = 25.0
    # Module specific contents
    teils: Dict[str, Any] = Field(default_factory=dict)
    # Server-side answer key (only present for auto-scorable modules like Lesen)
    answer_key: Optional[Dict[str, Any]] = None


class ExamSubmission(BaseModel):
    paper_id: str
    module: ExamModule
    level: str = "A2"
    answers: Dict[str, Any]  # e.g. {"teil1": {"1": "a", ...}, "teil2": ...} or {"teil1": "...", "teil2": "..."}


class QuestionEvaluation(BaseModel):
    id: int
    user_answer: str
    correct_answer: str
    is_correct: bool
    explanation: str


class EvaluationResult(BaseModel):
    submission_id: str
    paper_id: str
    module: ExamModule
    level: str = "A2"
    timestamp: str
    raw_score: float
    max_raw_score: float
    module_score: float  # scaled /25
    max_module_score: float = 25.0
    passed: bool  # >= 15 / 25
    breakdown: Dict[str, Any] = Field(default_factory=dict)
    general_feedback: Optional[str] = None
