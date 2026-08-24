"""Redis storage for Goethe A2 Exam Papers, Answer Keys, and Submissions.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import time
from typing import Any, Dict, List, Optional
try:
    import redis
except ImportError:
    redis = None

from ..config import REDIS_HOST, REDIS_PORT

logger = logging.getLogger("lang_learn.exam.storage")


def _compute_paper_fingerprint(paper_dict: Dict[str, Any]) -> str:
    """Compute a deterministic hash of the core content of an exam paper."""
    import hashlib

    raw_mod = paper_dict.get("module", "lesen")
    if isinstance(raw_mod, dict):
        raw_mod = raw_mod.get("value", "lesen")
    mod_str = str(raw_mod).lower().replace("exammodule.", "")

    teils = paper_dict.get("teils") or {}
    signatures = [mod_str]

    if mod_str == "lesen":
        t1 = teils.get("teil1") or {}
        signatures.append(str(t1.get("text", "")).strip()[:100])
        t2 = teils.get("teil2") or {}
        signatures.append(str(t2.get("title", "")).strip())
        t3 = teils.get("teil3") or {}
        signatures.append(str(t3.get("text", "")).strip()[:100])
        t4 = teils.get("teil4") or {}
        signatures.append(str(t4.get("title", "")).strip())
    elif mod_str == "schreiben":
        t1 = teils.get("teil1") or {}
        signatures.append(str(t1.get("scenario_german", "")).strip()[:100])
        t2 = teils.get("teil2") or {}
        signatures.append(str(t2.get("scenario_german", "")).strip()[:100])
    else:
        signatures.append(json.dumps(teils, sort_keys=True))

    combined = "||".join(signatures)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:24]


class ExamStorage:
    """Manages persistence of exam papers, answer keys, and submissions in Redis."""

    def __init__(self) -> None:
        self.redis_client: Optional[redis.Redis] = None
        self._memory_papers: Dict[str, str] = {}
        self._memory_paper_meta: Dict[str, Dict[str, Any]] = {}
        self._memory_counters: Dict[str, int] = {}
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

    def store_paper(self, paper_id: str, paper_dict: Dict[str, Any], ttl: int = 604800) -> str:
        """Store exam paper with answer key in Redis (7 days TTL).

        Guarantees deduplication: if a paper with identical content was already stored,
        it skips creating a duplicate entry and reuses the existing paper.
        """
        raw_mod = paper_dict.get("module", "lesen")
        if isinstance(raw_mod, dict):
            raw_mod = raw_mod.get("value", "lesen")
        mod_str = str(raw_mod).lower().replace("exammodule.", "")

        fp = _compute_paper_fingerprint(paper_dict)
        fp_key = f"exam:paper:fp:{mod_str}:{fp}"

        if self.redis_client is not None:
            try:
                existing_pid = self.redis_client.get(fp_key)
                if existing_pid:
                    existing_meta = self.redis_client.hgetall(f"exam:paper:meta:{existing_pid}")
                    if existing_meta and "label" in existing_meta:
                        existing_label = existing_meta["label"]
                        logger.info("Exam content already exists as [%s] (ID: %s). Deduplicating.", existing_label, existing_pid)
                        paper_dict["paper_id"] = existing_pid
                        paper_dict["label"] = existing_label
                        return existing_label
            except Exception as e:
                logger.warning("Error checking paper fingerprint in Redis: %s", e)

        # Generate new sequential label
        counter_key = f"exam:paper_counter:{mod_str}"
        seq_num = 1
        if self.redis_client is not None:
            try:
                seq_num = self.redis_client.incr(counter_key)
            except Exception as e:
                logger.error("Failed to increment paper counter in Redis: %s", e)
                self._memory_counters[mod_str] = self._memory_counters.get(mod_str, 0) + 1
                seq_num = self._memory_counters[mod_str]
        else:
            self._memory_counters[mod_str] = self._memory_counters.get(mod_str, 0) + 1
            seq_num = self._memory_counters[mod_str]

        label = paper_dict.get("label") or f"{mod_str.capitalize()} Paper {seq_num}"
        paper_dict["label"] = label

        key = f"exam:paper:{paper_id}"
        meta_key = f"exam:paper:meta:{paper_id}"
        alias_key = f"exam:paper:by_label:{label.lower().replace(' ', '_')}"
        serialized = json.dumps(paper_dict, ensure_ascii=False)
        now_ts = time.time()

        meta = {
            "paper_id": paper_id,
            "label": label,
            "module": mod_str,
            "level": str(paper_dict.get("level", "A2")),
            "created_at": str(paper_dict.get("created_at") or datetime.now(timezone.utc).isoformat()),
            "duration_minutes": str(paper_dict.get("duration_minutes", 30)),
            "total_points": str(paper_dict.get("total_points", 25.0)),
            "status": "pending",
            "fingerprint": fp,
        }

        if self.redis_client is not None:
            try:
                # 1. Store full paper content with answers
                self.redis_client.setex(key, ttl, serialized)
                # 2. Store alias key for direct human-readable lookup
                self.redis_client.setex(alias_key, ttl, paper_id)
                # 3. Store fingerprint for deduplication
                self.redis_client.setex(fp_key, ttl, paper_id)
                # 4. Store paper metadata
                self.redis_client.hset(meta_key, mapping=meta)
                self.redis_client.expire(meta_key, ttl)
                # 5. Add to sorted index
                self.redis_client.zadd("exam:papers:index", {paper_id: now_ts})
                logger.info("Stored unique exam paper in Redis under key: %s (label: %s, fp: %s)", key, label, fp)
            except Exception as e:
                logger.error("Failed to store paper in Redis: %s", e)

        # Fallback / local cache
        self._memory_papers[paper_id] = serialized
        self._memory_paper_meta[paper_id] = meta
        return label

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve exam paper (including answer key) by ID or label alias."""
        lookup_id = paper_id

        if self.redis_client is not None:
            try:
                # Check if paper_id is a label alias (e.g. 'lesen_paper_1' or 'Lesen Paper 1')
                if not paper_id.startswith("exam:paper:"):
                    alias_key = f"exam:paper:by_label:{paper_id.lower().replace(' ', '_')}"
                    aliased_id = self.redis_client.get(alias_key)
                    if aliased_id and isinstance(aliased_id, (str, bytes, bytearray)):
                        lookup_id = str(aliased_id)

                key = f"exam:paper:{lookup_id}"
                raw = self.redis_client.get(key)
                if raw and isinstance(raw, (str, bytes, bytearray)):
                    return json.loads(raw)
            except Exception as e:
                logger.error("Failed to fetch paper %s from Redis: %s", lookup_id, e)

        # Fallback
        raw = self._memory_papers.get(lookup_id)
        if raw and isinstance(raw, (str, bytes, bytearray)):
            return json.loads(raw)
        return None

    def list_papers(
        self,
        module: Optional[str] = None,
        status: Optional[str] = "pending",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Retrieve saved exam question papers from Redis."""
        papers: List[Dict[str, Any]] = []
        mod_filter = module.strip().lower() if module else None

        if self.redis_client is not None:
            try:
                paper_ids = self.redis_client.zrevrange("exam:papers:index", 0, -1)
                for pid in paper_ids:
                    meta_key = f"exam:paper:meta:{pid}"
                    meta = self.redis_client.hgetall(meta_key)
                    if not meta:
                        # Paper expired or deleted; cleanup index
                        self.redis_client.zrem("exam:papers:index", pid)
                        continue

                    if mod_filter and meta.get("module") != mod_filter:
                        continue
                    if status and meta.get("status") != status:
                        continue

                    papers.append({
                        "paper_id": meta.get("paper_id", pid),
                        "label": meta.get("label", f"Paper {pid[:8]}"),
                        "module": meta.get("module", "lesen"),
                        "level": meta.get("level", "A2"),
                        "created_at": meta.get("created_at", ""),
                        "duration_minutes": int(meta.get("duration_minutes", 30)),
                        "total_points": float(meta.get("total_points", 25.0)),
                        "status": meta.get("status", "pending")
                    })
                    if len(papers) >= limit:
                        break
                return papers
            except Exception as e:
                logger.error("Failed to list papers from Redis: %s", e)

        # Fallback in-memory
        for pid, meta in sorted(
            self._memory_paper_meta.items(),
            key=lambda x: x[1].get("created_at", ""),
            reverse=True
        ):
            if mod_filter and meta.get("module") != mod_filter:
                continue
            if status and meta.get("status") != status:
                continue
            papers.append({
                "paper_id": meta.get("paper_id", pid),
                "label": meta.get("label", f"Paper {pid[:8]}"),
                "module": meta.get("module", "lesen"),
                "level": meta.get("level", "A2"),
                "created_at": meta.get("created_at", ""),
                "duration_minutes": int(meta.get("duration_minutes", 30)),
                "total_points": float(meta.get("total_points", 25.0)),
                "status": meta.get("status", "pending")
            })
            if len(papers) >= limit:
                break
        return papers

    def get_paper_meta(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata dictionary for a paper."""
        meta_key = f"exam:paper:meta:{paper_id}"
        if self.redis_client is not None:
            try:
                meta = self.redis_client.hgetall(meta_key)
                if meta:
                    return meta
            except Exception as e:
                logger.error("Failed to fetch paper meta from Redis: %s", e)
        return self._memory_paper_meta.get(paper_id)

    def mark_paper_completed(self, paper_id: str) -> None:
        """Mark a saved paper as completed."""
        meta_key = f"exam:paper:meta:{paper_id}"
        if self.redis_client is not None:
            try:
                self.redis_client.hset(meta_key, "status", "completed")
                logger.info("Marked paper %s as completed in Redis", paper_id)
            except Exception as e:
                logger.error("Failed to mark paper %s completed in Redis: %s", paper_id, e)
        if paper_id in self._memory_paper_meta:
            self._memory_paper_meta[paper_id]["status"] = "completed"

    def delete_paper(self, paper_id: str) -> bool:
        """Delete a saved paper from Redis and clean up indexes."""
        key = f"exam:paper:{paper_id}"
        meta_key = f"exam:paper:meta:{paper_id}"
        if self.redis_client is not None:
            try:
                meta = self.redis_client.hgetall(meta_key)
                if meta and "label" in meta:
                    alias_key = f"exam:paper:by_label:{meta['label'].lower().replace(' ', '_')}"
                    self.redis_client.delete(alias_key)
                if meta and "fingerprint" in meta:
                    fp_key = f"exam:paper:fp:{meta.get('module', 'lesen')}:{meta['fingerprint']}"
                    self.redis_client.delete(fp_key)
                self.redis_client.delete(key)
                self.redis_client.delete(meta_key)
                self.redis_client.zrem("exam:papers:index", paper_id)
                logger.info("Deleted paper %s from Redis", paper_id)
            except Exception as e:
                logger.error("Failed to delete paper %s from Redis: %s", paper_id, e)

        self._memory_papers.pop(paper_id, None)
        self._memory_paper_meta.pop(paper_id, None)
        return True

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
