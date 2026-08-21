"""Unit tests for the Goethe A2 Exam suite (generators, storage, evaluators, orchestrator).
"""
from __future__ import annotations

import asyncio
import pytest
from inference.src.config import PROMPTS_DIR
from inference.src.exam.models import ExamModule
from inference.src.exam.storage import ExamStorage
from inference.src.exam.evaluators import evaluate_reading, evaluate_writing
from inference.src.exam.orchestrator import ExamOrchestrator


def test_exam_prompt_files_exist() -> None:
    """Verify all required Goethe A2 exam prompt template files exist."""
    required = [
        "exam_lesen_teil1.txt",
        "exam_lesen_teil2.txt",
        "exam_lesen_teil3.txt",
        "exam_lesen_teil4.txt",
        "exam_schreiben_teil1.txt",
        "exam_schreiben_teil2.txt",
        "exam_eval_schreiben.txt",
    ]
    for prompt_file in required:
        assert (PROMPTS_DIR / prompt_file).is_file(), f"Missing prompt file: {prompt_file}"


def test_reading_exam_orchestration_and_isolation() -> None:
    """Verify generate_paper for 'lesen' produces 4 reading Teile and sanitized output."""
    async def _run():
        storage = ExamStorage()
        orchestrator = ExamOrchestrator(storage=storage)

        paper = await orchestrator.generate_paper(module="lesen", level="A2")

        assert paper["module"] == "lesen"
        assert "paper_id" in paper
        assert paper["duration_minutes"] == 30
        assert paper["total_points"] == 25.0

        teils = paper["teils"]
        assert "teil1" in teils
        assert "teil2" in teils
        assert "teil3" in teils
        assert "teil4" in teils
        assert "teil5" not in teils  # Only 4 reading teils

        # Verify answer keys are NOT exposed to frontend
        assert "answer_key" not in paper
        for t_key, t_val in teils.items():
            for item in t_val.get("items", []):
                assert "answer_key" not in item, f"answer_key leaked in {t_key} item {item}"

        # Verify stored in storage with answer key
        stored = storage.get_paper(paper["paper_id"])
        assert stored is not None
        assert "answer_key" in stored
        assert len(stored["answer_key"]["answer_key"]) == 20

    asyncio.run(_run())


def test_writing_exam_orchestration_and_isolation() -> None:
    """Verify generate_paper for 'schreiben' produces 2 writing Teile and no reading content."""
    async def _run():
        storage = ExamStorage()
        orchestrator = ExamOrchestrator(storage=storage)

        paper = await orchestrator.generate_paper(module="schreiben", level="A2")

        assert paper["module"] == "schreiben"
        assert "paper_id" in paper
        teils = paper["teils"]
        assert "teil1" in teils
        assert "teil2" in teils
        assert "teil3" not in teils  # Only 2 writing teils

        assert "bullet_points" in teils["teil1"]
        assert "bullet_points" in teils["teil2"]

    asyncio.run(_run())


def test_lesen_exact_scoring_and_pass_threshold() -> None:
    """Verify 20 raw questions * 1.25 = 25 module points and pass rule (>=15 raw items)."""
    # Construct mock stored paper
    mock_paper = {
        "paper_id": "test-paper-123",
        "module": "lesen",
        "level": "A2",
        "answer_key": {
            "answer_key": {str(i): "a" for i in range(1, 21)},
            "explanations": {str(i): "Explanation for " + str(i) for i in range(1, 21)}
        }
    }

    # Test perfect score: 20 correct -> 20 * 1.25 = 25.0 points, passed=True
    perfect_answers = {str(i): "a" for i in range(1, 21)}
    result_perfect = evaluate_reading(mock_paper, perfect_answers)
    assert result_perfect.raw_score == 20.0
    assert result_perfect.module_score == 25.0
    assert result_perfect.passed is True
    assert len(result_perfect.breakdown["items"]) == 20

    # Test passing boundary: 15 correct -> 15 * 1.25 = 18.8 points, passed=True
    pass_answers = {str(i): ("a" if i <= 15 else "b") for i in range(1, 21)}
    result_pass = evaluate_reading(mock_paper, pass_answers)
    assert result_pass.raw_score == 15.0
    assert result_pass.module_score == 18.8
    assert result_pass.passed is True

    # Test failing score: 14 correct -> 14 * 1.25 = 17.5 points, passed=False
    fail_answers = {str(i): ("a" if i <= 14 else "b") for i in range(1, 21)}
    result_fail = evaluate_reading(mock_paper, fail_answers)
    assert result_fail.raw_score == 14.0
    assert result_fail.module_score == 17.5
    assert result_fail.passed is False


def test_storage_submission_and_history() -> None:
    """Verify storing submissions updates history and can be queried."""
    storage = ExamStorage()
    sub_id = "test-sub-abc"
    result_data = {
        "submission_id": sub_id,
        "paper_id": "paper-123",
        "module": "lesen",
        "timestamp": "2026-08-21T00:00:00Z",
        "module_score": 22.5,
        "max_module_score": 25.0,
        "passed": True
    }

    assert storage.store_submission(sub_id, result_data) is True
    retrieved = storage.get_submission(sub_id)
    assert retrieved is not None
    assert retrieved["module_score"] == 22.5

    history = storage.get_history(limit=5)
    assert any(h["submission_id"] == sub_id for h in history)


def test_writing_evaluation_scoring() -> None:
    """Verify writing evaluation calculates scores and creates structured result."""
    mock_paper = {
        "paper_id": "test-write-123",
        "module": "schreiben",
        "level": "A2",
        "teils": {
            "teil1": {"title": "SMS an Michael", "bullet_points": ["P1", "P2", "P3"]},
            "teil2": {"title": "Email an Schule", "bullet_points": ["P1", "P2", "P3", "P4"]}
        }
    }
    user_answers = {
        "teil1": "Lieber Michael, ich komme heute leider 20 Minuten später weil mein Zug Verspätung hat. Treffen wir uns vor dem Kino?",
        "teil2": "Sehr geehrte Frau Weber, ich möchte im Sommer einen Deutschkurs an Ihrer Schule besuchen. Bitte senden Sie mir Termine und Preise."
    }

    result = evaluate_writing(mock_paper, user_answers)
    assert result.module == ExamModule.SCHREIBEN
    assert result.max_module_score == 25.0
    assert 0 <= result.module_score <= 25.0
    assert "teil1" in result.breakdown
    assert "teil2" in result.breakdown


def test_service_exam_integration() -> None:
    """Verify LangLearnService exam endpoints dispatch without errors."""
    from unittest.mock import patch

    async def _run():
        with patch("inference.src.service.warmup"), patch("inference.src.service.SemanticCache"):
            from inference.src.service import LangLearnService
            service = LangLearnService()

            # 1. Generate reading paper
            read_paper = await service.exam_generate(module="lesen", level="A2")
            assert "paper_id" in read_paper
            assert read_paper["module"] == "lesen"
            paper_id = read_paper["paper_id"]

            # 2. Evaluate reading paper
            answers = {str(i): "a" for i in range(1, 21)}
            eval_result = service.exam_evaluate(
                paper_id=paper_id,
                module="lesen",
                answers=answers,
                level="A2"
            )
            assert eval_result["paper_id"] == paper_id
            assert eval_result["module"] == "lesen"
            assert "module_score" in eval_result

            # 3. Check history endpoint
            hist = service.exam_history(limit=5)
            assert "history" in hist
            assert isinstance(hist["history"], list)

    asyncio.run(_run())

