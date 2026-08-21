"""Redis storage for Goethe A2 Exam Papers, Answer Keys, and Submissions.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
import redis

from ..config import REDIS_HOST, REDIS_PORT

logger = logging.getLogger("lang_learn.exam.storage")


class ExamStorage:
    """Manages persistence of exam papers, answer keys, and submissions in Redis."""

    def __init__(self) -> None:
        self.redis_client: Optional[redis.Redis] = None
        self._memory_papers: Dict[str, str] = {}
        self._memory_submissions: Dict[str, str] = {}
        self._memory_history: List[str] = []

        try:
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                socket_connect_timeout=2.0,
                socket_keepalive=True,
            )
            client.ping()
            self.redis_client = client
            logger.info("ExamStorage connected to Redis at %s:%d", REDIS_HOST, REDIS_PORT)
        except Exception as e:
            logger.warning("ExamStorage running in in-memory fallback mode (Redis unavailable): %s", e)

    def store_paper(self, paper_id: str, paper_dict: Dict[str, Any], ttl: int = 604800) -> bool:
        """Store exam paper with answer key in Redis (7 days TTL)."""
        key = f"exam:paper:{paper_id}"
        serialized = json.dumps(paper_dict, ensure_ascii=False)

        if self.redis_client is not None:
            try:
                self.redis_client.setex(key, ttl, serialized)
                logger.info("Stored exam paper in Redis under key: %s", key)
            except Exception as e:
                logger.error("Failed to store paper in Redis: %s", e)

        # Fallback / local cache
        self._memory_papers[paper_id] = serialized
        return True

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve exam paper (including answer key) by ID."""
        key = f"exam:paper:{paper_id}"

        if self.redis_client is not None:
            try:
                raw = self.redis_client.get(key)
                if raw and isinstance(raw, (str, bytes, bytearray)):
                    return json.loads(raw)
            except Exception as e:
                logger.error("Failed to fetch paper %s from Redis: %s", paper_id, e)

        # Fallback
        raw = self._memory_papers.get(paper_id)
        if raw and isinstance(raw, (str, bytes, bytearray)):
            return json.loads(raw)
        return None

    def store_submission(self, submission_id: str, result_dict: Dict[str, Any], ttl: int = 2592000) -> bool:
        """Store exam submission & evaluation result in Redis (30 days TTL)."""
        key = f"exam:submission:{submission_id}"
        serialized = json.dumps(result_dict, ensure_ascii=False)

        if self.redis_client is not None:
            try:
                self.redis_client.setex(key, ttl, serialized)
                # Maintain recent history list
                self.redis_client.lpush("exam:history", submission_id)
                self.redis_client.ltrim("exam:history", 0, 99)  # keep last 100
                logger.info("Stored submission in Redis under key: %s", key)
            except Exception as e:
                logger.error("Failed to store submission in Redis: %s", e)

        # Fallback / local cache
        self._memory_submissions[submission_id] = serialized
        self._memory_history.insert(0, submission_id)
        return True

    def get_submission(self, submission_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve submission evaluation result by ID."""
        key = f"exam:submission:{submission_id}"

        if self.redis_client is not None:
            try:
                raw = self.redis_client.get(key)
                if raw and isinstance(raw, (str, bytes, bytearray)):
                    return json.loads(raw)
            except Exception as e:
                logger.error("Failed to fetch submission %s from Redis: %s", submission_id, e)

        raw = self._memory_submissions.get(submission_id)
        if raw and isinstance(raw, (str, bytes, bytearray)):
            return json.loads(raw)
        return None

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve summary of recent test submissions."""
        submissions: List[Dict[str, Any]] = []

        if self.redis_client is not None:
            try:
                sub_ids = self.redis_client.lrange("exam:history", 0, limit - 1)
                for sid in sub_ids:
                    res = self.get_submission(sid)
                    if res:
                        submissions.append({
                            "submission_id": res.get("submission_id"),
                            "paper_id": res.get("paper_id"),
                            "module": res.get("module"),
                            "timestamp": res.get("timestamp"),
                            "module_score": res.get("module_score"),
                            "max_module_score": res.get("max_module_score", 25.0),
                            "passed": res.get("passed")
                        })
                return submissions
            except Exception as e:
                logger.error("Failed to fetch history from Redis: %s", e)

        for sid in self._memory_history[:limit]:
            res = self.get_submission(sid)
            if res:
                submissions.append({
                    "submission_id": res.get("submission_id"),
                    "paper_id": res.get("paper_id"),
                    "module": res.get("module"),
                    "timestamp": res.get("timestamp"),
                    "module_score": res.get("module_score"),
                    "max_module_score": res.get("max_module_score", 25.0),
                    "passed": res.get("passed")
                })
        return submissions
