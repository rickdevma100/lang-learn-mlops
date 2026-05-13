"""Smoke tests for the lang-learn BentoML service.

The slow tests load the full ~5 GB Gemma MLX model and run real Metal
inference, so they're gated behind the `slow` marker. Run them with:

    pytest -m slow inference/tests/test_service.py

The fast tests cover prompt loading and template formatting and run in
milliseconds with no model required.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from inference.src.config import MODEL_PATH, PROMPTS_DIR
from inference.src.prompts import load_prompt


def test_prompt_files_exist() -> None:
    for name in ("scenario_dialogue.txt", "image_describe.txt"):
        assert (PROMPTS_DIR / name).is_file(), f"missing prompt: {name}"


def test_scenario_dialogue_template_formats() -> None:
    template = load_prompt("scenario_dialogue.txt")
    rendered = template.format(scenario="bargaining on cup price")
    assert "bargaining on cup price" in rendered
    assert "{scenario}" not in rendered


def test_load_prompt_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist.txt")


@pytest.mark.slow
def test_scenario_dialogue_end_to_end() -> None:
    """Loads the real model and runs a short generation. Requires gemma-mlx."""
    if not Path(MODEL_PATH).is_dir():
        pytest.skip(f"Model not present at {MODEL_PATH}")

    from inference.src.service import LangLearnService

    service = LangLearnService()
    result = service.scenario_dialogue(
        scenario="bargaining on cup price",
        max_tokens=64,
    )
    assert isinstance(result, dict)
    assert "response" in result
    assert isinstance(result["response"], str)
    assert len(result["response"].strip()) > 0
