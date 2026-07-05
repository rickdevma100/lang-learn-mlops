"""Text-to-Speech module using edge-tts.

Provides async synthesis for German dialogue lines with
speaker-specific voices and CEFR-level-appropriate speech rates.
"""
from __future__ import annotations

import asyncio
import logging
from io import BytesIO

# pyrefly: ignore [missing-import]
import edge_tts

logger = logging.getLogger("lang_learn.tts")

# ---------------------------------------------------------------------------
# Voice catalog — best native-sounding voices per language
# ---------------------------------------------------------------------------
VOICE_CATALOG: dict[str, dict[str, str]] = {
    "German": {
        "person_a": "de-DE-KatjaNeural",      # female, warm
        "person_b": "de-DE-ConradNeural",     # male, clear
        "default":  "de-DE-KatjaNeural",
    },
    "French": {
        "person_a": "fr-FR-DeniseNeural",
        "person_b": "fr-FR-HenriNeural",
        "default":  "fr-FR-DeniseNeural",
    },
    "Spanish": {
        "person_a": "es-ES-ElviraNeural",
        "person_b": "es-ES-AlvaroNeural",
        "default":  "es-ES-ElviraNeural",
    },
    "Italian": {
        "person_a": "it-IT-ElsaNeural",
        "person_b": "it-IT-DiegoNeural",
        "default":  "it-IT-ElsaNeural",
    },
    "Japanese": {
        "person_a": "ja-JP-NanamiNeural",
        "person_b": "ja-JP-KeitaNeural",
        "default":  "ja-JP-NanamiNeural",
    },
    "English": {
        "person_a": "en-US-JennyNeural",
        "person_b": "en-US-GuyNeural",
        "default":  "en-US-JennyNeural",
    },
}

# CEFR level → speech rate (slower for beginners)
RATE_BY_LEVEL: dict[str, str] = {
    "A1": "-25%",   # very slow
    "A2": "-15%",   # slow
    "B1": "-6%",    # slightly slow
    "B2": "+0%",    # normal
    "C1": "+0%",    # normal
    "C2": "-5%",    # slightly fast
}


async def synthesize(text: str, voice: str, rate: str = "+0%") -> bytes:
    """Generate MP3 audio bytes for a single piece of text using the given voice."""
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    audio_buffer = BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])
    return audio_buffer.getvalue()


async def synthesize_line(
    text: str,
    language: str = "German",
    level: str = "A2",
    speaker: str | None = None,
) -> bytes:
    """Synthesize a single dialogue line to MP3 bytes.

    Args:
        text:     The text to speak.
        language: Language key (e.g. 'German').
        level:    CEFR level (e.g. 'A2') — controls speech rate.
        speaker:  'person_a', 'person_b', or None for default voice.
    """
    voices = VOICE_CATALOG.get(language, VOICE_CATALOG["German"])
    role = speaker if speaker in voices else "default"
    voice = voices[role]
    rate = RATE_BY_LEVEL.get(level, "+0%")

    logger.debug("TTS: voice=%s rate=%s text_len=%d", voice, rate, len(text))
    return await synthesize(text, voice, rate)


def synthesize_line_sync(
    text: str,
    language: str = "German",
    level: str = "A2",
    speaker: str | None = None,
) -> bytes:
    """Synchronous wrapper around synthesize_line for use in non-async contexts."""
    return asyncio.run(synthesize_line(text, language, level, speaker))
