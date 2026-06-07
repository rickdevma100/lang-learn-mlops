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
    for name in ("scenario_dialogue.txt", "image_describe.txt", "explain_word.txt"):
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


def test_parse_dialogue() -> None:
    from inference.src.service import parse_dialogue
    
    # Standard format without translations
    raw_text_1 = "Guten Tag. Ich brauche Hilfe.\nPerson B:\nWas ist los?\nPerson A:\nIch verstehe nichts."
    dialogue_1 = parse_dialogue(raw_text_1)
    
    assert len(dialogue_1) == 3
    assert dialogue_1[0] == {"speaker": "Person A", "german": "Guten Tag. Ich brauche Hilfe.", "english": ""}
    assert dialogue_1[1] == {"speaker": "Person B", "german": "Was ist los?", "english": ""}
    assert dialogue_1[2] == {"speaker": "Person A", "german": "Ich verstehe nichts.", "english": ""}

    # Format starting with explicit Person A
    raw_text_2 = "Person A:\nHallo!\nPerson B: Hallo, wie geht es dir?\nPerson A:\nMir geht es gut."
    dialogue_2 = parse_dialogue(raw_text_2)
    
    assert len(dialogue_2) == 3
    assert dialogue_2[0] == {"speaker": "Person A", "german": "Hallo!", "english": ""}
    assert dialogue_2[1] == {"speaker": "Person B", "german": "Hallo, wie geht es dir?", "english": ""}
    assert dialogue_2[2] == {"speaker": "Person A", "german": "Mir geht es gut.", "english": ""}

    # Format with translations
    raw_text_3 = (
        "Person A:\nHallo, wie geht es dir?\nTranslation: Hello, how are you?\n"
        "Person B:\nMir geht es gut, danke!\nEnglish: I am doing well, thank you!\n"
        "Person A:\nSchön zu hören.\nEnglisch: Nice to hear."
    )
    dialogue_3 = parse_dialogue(raw_text_3)
    assert len(dialogue_3) == 3
    assert dialogue_3[0] == {"speaker": "Person A", "german": "Hallo, wie geht es dir?", "english": "Hello, how are you?"}
    assert dialogue_3[1] == {"speaker": "Person B", "german": "Mir geht es gut, danke!", "english": "I am doing well, thank you!"}
    assert dialogue_3[2] == {"speaker": "Person A", "german": "Schön zu hören.", "english": "Nice to hear."}

