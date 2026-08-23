"""Redis-backed certified text pool for Goethe A2 exams.

Manages a pool of certified texts per Teil with anti-repeat tracking.
Texts are stored in Redis and seeded from seed_texts.json on first boot.
"""
from __future__ import annotations

import collections
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis

from ..config import REDIS_HOST, REDIS_PORT

logger = logging.getLogger("lang_learn.exam.text_pool")

# Redis key patterns
_POOL_KEY = "exam:textpool:{teil}:texts"       # JSON list of texts
_RECENT_KEY = "exam:textpool:recent:{teil}"     # List of recent indices
_ANTI_REPEAT_SIZE = 15  # Track last 15 served indices


class TextPool:
    """Manages certified Goethe A2 texts in Redis with anti-repeat tracking."""

    def __init__(self, redis_client: Optional[redis.Redis] = None) -> None:
        self._redis = redis_client or redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_random_text(self, teil: str) -> Dict[str, Any]:
        """Pick a random text from the pool, avoiding recent repeats.
        
        Args:
            teil: One of 'lesen_teil1', 'lesen_teil2', 'lesen_teil3', 'lesen_teil4'
        
        Returns:
            A dict with the text data (title, text, etc.)
        
        Raises:
            ValueError: If no texts are available for the given teil.
        """
        key = _POOL_KEY.format(teil=teil)
        raw = self._redis.get(key)
        if not raw:
            raise ValueError(f"No texts in pool for '{teil}'. Run seed_pool() first.")

        texts = json.loads(raw)
        if not texts:
            raise ValueError(f"Empty pool for '{teil}'.")

        pool_size = len(texts)
        recent_key = _RECENT_KEY.format(teil=teil)

        # Get recently served indices
        recent_raw = self._redis.lrange(recent_key, 0, -1)
        recent_indices = set(int(x) for x in recent_raw if x.isdigit())

        # Find available indices (not recently served)
        available = [i for i in range(pool_size) if i not in recent_indices]
        if not available:
            # All exhausted → clear history and pick from all
            self._redis.delete(recent_key)
            available = list(range(pool_size))

        choice_idx = random.choice(available)

        # Track this choice
        self._redis.rpush(recent_key, str(choice_idx))
        self._redis.ltrim(recent_key, -_ANTI_REPEAT_SIZE, -1)

        text_data = texts[choice_idx]
        text_data["_pool_index"] = choice_idx
        return text_data

    def add_text(self, teil: str, text_data: Dict[str, Any]) -> int:
        """Add a custom text to the pool. Returns new pool size."""
        key = _POOL_KEY.format(teil=teil)
        raw = self._redis.get(key)
        texts = json.loads(raw) if raw else []
        texts.append(text_data)
        self._redis.set(key, json.dumps(texts, ensure_ascii=False))
        logger.info("Added text to %s pool (now %d texts)", teil, len(texts))
        return len(texts)

    def remove_text(self, teil: str, index: int) -> bool:
        """Remove a text from the pool by index. Returns True if successful."""
        key = _POOL_KEY.format(teil=teil)
        raw = self._redis.get(key)
        if not raw:
            return False
        texts = json.loads(raw)
        if index < 0 or index >= len(texts):
            return False
        texts.pop(index)
        self._redis.set(key, json.dumps(texts, ensure_ascii=False))
        # Clear recent history since indices shifted
        self._redis.delete(_RECENT_KEY.format(teil=teil))
        logger.info("Removed text %d from %s pool (now %d texts)", index, teil, len(texts))
        return True

    def list_texts(self, teil: str) -> List[Dict[str, Any]]:
        """List all texts in the pool for a given teil."""
        key = _POOL_KEY.format(teil=teil)
        raw = self._redis.get(key)
        if not raw:
            return []
        return json.loads(raw)

    def get_pool_size(self, teil: str) -> int:
        """Return number of texts in pool for given teil."""
        key = _POOL_KEY.format(teil=teil)
        raw = self._redis.get(key)
        if not raw:
            return 0
        return len(json.loads(raw))

    def get_pool_stats(self) -> Dict[str, int]:
        """Return pool sizes for all teile."""
        stats = {}
        for teil in ["lesen_teil1", "lesen_teil2", "lesen_teil3", "lesen_teil4"]:
            stats[teil] = self.get_pool_size(teil)
        return stats

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_pool(self, force: bool = False) -> Dict[str, int]:
        """Load seed texts from seed_texts.json into Redis.
        
        Args:
            force: If True, overwrite existing pool data.
        
        Returns:
            Dict of teil → count of texts seeded.
        """
        seed_file = Path(__file__).parent / "seed_texts.json"
        if not seed_file.is_file():
            logger.warning("seed_texts.json not found at %s", seed_file)
            return {}

        with open(seed_file, encoding="utf-8") as f:
            seed_data = json.load(f)

        result = {}
        for teil, texts in seed_data.items():
            key = _POOL_KEY.format(teil=teil)
            existing = self._redis.get(key)

            if existing and not force:
                existing_count = len(json.loads(existing))
                if existing_count >= len(texts):
                    logger.info("Pool %s already has %d texts, skipping seed", teil, existing_count)
                    result[teil] = existing_count
                    continue

            self._redis.set(key, json.dumps(texts, ensure_ascii=False))
            result[teil] = len(texts)
            logger.info("Seeded %s with %d texts", teil, len(texts))

        return result
