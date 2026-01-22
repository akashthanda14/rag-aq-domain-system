"""
Unit Tests for Embedding Generation Module

Tests the functionality of embedding generation and related utilities.

Author: CSE435 Project Team
"""

import pytest
import numpy as np
from src.embeddings import EmbeddingGenerator, EmbeddingCache, normalize_embeddings


class TestEmbeddingGenerator:
    """Test cases for EmbeddingGenerator class."""
    
    def test_initialization(self):
        """Test that EmbeddingGenerator initializes correctly."""
        embedder = EmbeddingGenerator()
        assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert embedder.device == "cpu"
        assert embedder.batch_size == 32
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        embedder = EmbeddingGenerator(
            model_name="custom-model",
            device="cuda",
            batch_size=16
        )
        assert embedder.model_name == "custom-model"
        assert embedder.device == "cuda"
        assert embedder.batch_size == 16
    
    # TODO: Add more tests once implementation is complete
    # def test_load_model(self):
    #     """Test model loading."""
    #     pass
    # 
    # def test_generate_embeddings(self):
    #     """Test embedding generation."""
    #     pass


class TestEmbeddingCache:
    """Test cases for EmbeddingCache class."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = EmbeddingCache(max_size=100)
        assert cache.max_size == 100
    
    # TODO: Add more tests
    # def test_cache_put_get(self):
    #     """Test cache storage and retrieval."""
    #     pass


class TestNormalizeEmbeddings:
    """Test cases for embedding normalization."""
    
    def test_normalize(self):
        """Test embedding normalization."""
        # TODO: Implement once function is complete
        embeddings = np.array([[1.0, 2.0, 3.0]])
        normalized = normalize_embeddings(embeddings)
        # Should verify L2 norm is 1
        # assert np.allclose(np.linalg.norm(normalized), 1.0)


if __name__ == "__main__":
    pytest.main([__file__])
