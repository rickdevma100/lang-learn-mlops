"""Unit tests for the SemanticCache module.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

# Mock third-party dependencies before importing the tested module
sys.modules["redis"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()

import numpy as np
import pytest

from inference.src.cache import SemanticCache


@patch("redis.Redis")
@patch("sentence_transformers.SentenceTransformer")
def test_cache_init(mock_transformer, mock_redis_class) -> None:
    mock_redis = MagicMock()
    mock_redis_class.return_value = mock_redis

    # Initialize cache (FT.INFO succeeds)
    cache = SemanticCache()

    # Assertions
    assert cache.enabled is True
    mock_redis.execute_command.assert_called_with("FT.INFO", "lang_learn_idx")


@patch("redis.Redis")
@patch("sentence_transformers.SentenceTransformer")
def test_cache_init_create_index(mock_transformer, mock_redis_class) -> None:
    mock_redis = MagicMock()
    mock_redis_class.return_value = mock_redis

    # FT.INFO raises exception (index missing), so FT.CREATE is called
    mock_redis.execute_command.side_effect = [Exception("index not found"), None]

    cache = SemanticCache()

    assert cache.enabled is True
    assert mock_redis.execute_command.call_count == 2
    # Verify the second call is FT.CREATE
    create_args = mock_redis.execute_command.call_args_list[1][0]
    assert create_args[0] == "FT.CREATE"
    assert "lang_learn_idx" in create_args


@patch("redis.Redis")
@patch("sentence_transformers.SentenceTransformer")
def test_cache_lookup_hit(mock_transformer, mock_redis_class) -> None:
    mock_redis = MagicMock()
    mock_redis_class.return_value = mock_redis

    # Mock FT.SEARCH to return a hit (similarity = 0.95 -> score = 0.05)
    mock_redis.execute_command.side_effect = [
        None,  # FT.INFO
        [1, b"dialog:doc_123", [b"response", b"Guten Tag", b"vector_score", b"0.05", b"hit_count", b"0"]]  # FT.SEARCH
    ]

    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros(384)
    mock_transformer.return_value = mock_model

    cache = SemanticCache()
    is_hit, response, similarity = cache.lookup("coffee", "German", "A1")

    assert is_hit is True
    assert response == "Guten Tag"
    assert similarity == pytest.approx(0.95)
    mock_redis.hincrby.assert_called_once_with("dialog:doc_123", "hit_count", 1)


@patch("redis.Redis")
@patch("sentence_transformers.SentenceTransformer")
def test_cache_lookup_miss_below_threshold(mock_transformer, mock_redis_class) -> None:
    mock_redis = MagicMock()
    mock_redis_class.return_value = mock_redis

    # Mock FT.SEARCH to return similarity = 0.90 -> score = 0.10
    mock_redis.execute_command.side_effect = [
        None,  # FT.INFO
        [1, b"dialog:doc_123", [b"response", b"Guten Tag", b"vector_score", b"0.10", b"hit_count", b"0"]]  # FT.SEARCH
    ]

    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros(384)
    mock_transformer.return_value = mock_model

    cache = SemanticCache()
    is_hit, response, similarity = cache.lookup("coffee", "German", "A1")

    assert is_hit is False
    assert response is None
    assert similarity == pytest.approx(0.90)
    mock_redis.hincrby.assert_not_called()


@patch("redis.Redis")
@patch("sentence_transformers.SentenceTransformer")
def test_cache_lookup_miss_no_results(mock_transformer, mock_redis_class) -> None:
    mock_redis = MagicMock()
    mock_redis_class.return_value = mock_redis

    mock_redis.execute_command.side_effect = [
        None,  # FT.INFO
        [0]  # FT.SEARCH returns 0 results
    ]

    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros(384)
    mock_transformer.return_value = mock_model

    cache = SemanticCache()
    is_hit, response, similarity = cache.lookup("coffee", "German", "A1")

    assert is_hit is False
    assert response is None
    assert similarity == 0.0


@patch("redis.Redis")
@patch("sentence_transformers.SentenceTransformer")
def test_cache_store(mock_transformer, mock_redis_class) -> None:
    mock_redis = MagicMock()
    mock_redis_class.return_value = mock_redis
    mock_redis.execute_command.side_effect = [None]  # FT.INFO

    mock_model = MagicMock()
    mock_model.encode.return_value = np.ones(384)
    mock_transformer.return_value = mock_model

    cache = SemanticCache()
    success = cache.store("Guten Tag", "coffee", "German", "A1", 0.85)

    assert success is True
    mock_redis.hset.assert_called_once()
    _, kwargs = mock_redis.hset.call_args
    mapping = kwargs.get("mapping")
    assert mapping["language"] == "German"
    assert mapping["level"] == "A1"
    assert mapping["scenario"] == "coffee"
    assert mapping["response"] == "Guten Tag"
    assert mapping["cefr_score"] == "0.85"
    # Assert TTL was NOT set
    mock_redis.expire.assert_not_called()



@patch("redis.Redis")
@patch("sentence_transformers.SentenceTransformer")
def test_cache_clear(mock_transformer, mock_redis_class) -> None:
    mock_redis = MagicMock()
    mock_redis_class.return_value = mock_redis
    mock_redis.execute_command.side_effect = [None]  # FT.INFO

    # Mock scan_iter to return some keys
    mock_redis.scan_iter.return_value = [b"dialog:key1", b"dialog:key2"]

    cache = SemanticCache()
    success = cache.clear()

    assert success is True
    mock_redis.scan_iter.assert_called_once_with("dialog:*")
    mock_redis.delete.assert_called_once_with(b"dialog:key1", b"dialog:key2")
