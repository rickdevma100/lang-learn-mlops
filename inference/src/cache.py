"""Semantic Cache implementation using Redis Stack and multilingual-e5-small embeddings.
"""
from __future__ import annotations

import logging
import uuid
import numpy as np
import redis

from .config import EMBEDDING_MODEL_PATH, REDIS_HOST, REDIS_PORT

logger = logging.getLogger("lang_learn.cache")


class SemanticCache:
    """Semantic Cache for language learning scenarios using Redis Vector Search."""

    def __init__(self) -> None:
        self.enabled = False
        self.model = None
        self.redis_client = None

        try:
            logger.info("Initializing SentenceTransformer embedding model from %s...", EMBEDDING_MODEL_PATH)
            # Offline loading (HF_HUB_OFFLINE=1 should be set in env)
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(EMBEDDING_MODEL_PATH)
            logger.info("Embedding model loaded successfully.")

            logger.info("Connecting to Redis at %s:%d...", REDIS_HOST, REDIS_PORT)
            # Use decode_responses=False so we get binary vectors correctly
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=False,
                socket_connect_timeout=2.0,
                socket_keepalive=True
            )
            self.redis_client.ping()
            logger.info("Connected to Redis.")

            self._create_index_if_not_exists()
            self.enabled = True
            logger.info("Semantic cache successfully enabled.")
        except Exception as e:
            logger.exception("Failed to initialize semantic cache, running without cache: %s", e)

    def _create_index_if_not_exists(self) -> None:
        """Create the FT vector search index if it doesn't already exist."""
        try:
            self.redis_client.execute_command("FT.INFO", "lang_learn_idx")
            logger.info("Redis vector search index 'lang_learn_idx' already exists.")
        except Exception:
            logger.info("Creating Redis vector search index 'lang_learn_idx'...")
            # Schema details:
            # - language, level: Tags for exact filtering
            # - scenario: Text for metadata
            # - embedding: FLAT vector index, COSINE distance, 384 dimensions
            # - response: Text of the generated dialogue
            # - cefr_score: Numeric metadata
            # - hit_count: Numeric tracker
            cmd = [
                "FT.CREATE", "lang_learn_idx",
                "ON", "HASH",
                "PREFIX", "1", "dialog:",
                "SCHEMA",
                "language", "TAG",
                "level", "TAG",
                "scenario", "TEXT",
                "embedding", "VECTOR", "FLAT", "6",
                "TYPE", "FLOAT32",
                "DIM", "384",
                "DISTANCE_METRIC", "COSINE",
                "response", "TEXT",
                "cefr_score", "NUMERIC",
                "hit_count", "NUMERIC"
            ]
            self.redis_client.execute_command(*cmd)
            logger.info("Redis vector search index 'lang_learn_idx' created.")

    def lookup(self, scenario: str, language: str, level: str) -> tuple[bool, str | None, float]:
        """Perform semantic lookup in Redis.

        Returns:
            (is_hit, response_text, similarity)
        """
        if not self.enabled or self.model is None or self.redis_client is None:
            return False, None, 0.0

        try:
            # Generate query embedding
            query_text = f"query: {scenario} | language: {language} | level: {level}"
            query_vector = self.model.encode(query_text).astype(np.float32).tobytes()

            # Escape tags (spaces -> "\ ")
            esc_lang = language.replace(" ", "\\ ")
            esc_level = level.replace(" ", "\\ ")

            # Build KNN search query
            # We filter by language and level tags, then find 1 nearest neighbor
            query_str = f"(@language:{{{esc_lang}}} @level:{{{esc_level}}})=>[KNN 1 @embedding $query_vector AS vector_score]"

            cmd = [
                "FT.SEARCH", "lang_learn_idx",
                query_str,
                "PARAMS", "2", "query_vector", query_vector,
                "SORTBY", "vector_score",
                "DIALECT", "2"
            ]

            res = self.redis_client.execute_command(*cmd)
            if not res or len(res) <= 1:
                return False, None, 0.0

            total_hits = res[0]
            if total_hits == 0:
                return False, None, 0.0

            # Document ID and Fields
            doc_id = res[1].decode("utf-8")
            fields_list = res[2]

            fields = {}
            for i in range(0, len(fields_list), 2):
                fields[fields_list[i].decode("utf-8")] = fields_list[i+1]

            # Parse score and calculate similarity
            vector_score = float(fields.get("vector_score", b"1.0").decode("utf-8"))
            similarity = 1.0 - vector_score

            # Check threshold
            if similarity > 0.93:
                response = fields.get("response", b"").decode("utf-8")
                # Increment hit count in Redis
                try:
                    self.redis_client.hincrby(doc_id, "hit_count", 1)
                except Exception as ex:
                    logger.warning("Failed to increment hit count for %s: %s", doc_id, ex)

                logger.info("Cache HIT: similarity=%f for scenario=%s", similarity, scenario)
                return True, response, similarity

            logger.info("Cache MISS: closest similarity=%f for scenario=%s", similarity, scenario)
            return False, None, similarity

        except Exception as e:
            logger.error("Error during cache lookup: %s", e)
            return False, None, 0.0

    def store(self, response: str, scenario: str, language: str, level: str, cefr_score: float) -> bool:
        """Store generated response in the semantic cache."""
        if not self.enabled or self.model is None or self.redis_client is None:
            return False

        try:
            # Generate passage embedding
            passage_text = f"passage: {scenario} | language: {language} | level: {level}"
            embedding = self.model.encode(passage_text).astype(np.float32).tobytes()

            doc_id = str(uuid.uuid4())
            key = f"dialog:{doc_id}"

            self.redis_client.hset(key, mapping={
                "language": language,
                "level": level,
                "scenario": scenario,
                "embedding": embedding,
                "response": response,
                "cefr_score": str(cefr_score),
                "hit_count": "0"
            })
            logger.info("Cached response stored successfully under key: %s", key)
            return True
        except Exception as e:
            logger.error("Error storing response in cache: %s", e)
            return False
