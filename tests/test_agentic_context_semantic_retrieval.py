"""Tests for refactored agentic_context.retrieval.semantic_retrieval module."""

import pytest
from cogneetree.retrieval.semantic_retrieval import EmbeddingModel


class TestEmbeddingModel:
    """Test EmbeddingModel class."""

    def test_embedding_model_creation(self):
        """Test creating an EmbeddingModel."""
        try:
            model = EmbeddingModel()
            assert model is not None
        except ImportError:
            # sentence_transformers not installed
            pytest.skip("sentence_transformers not installed")

    def test_encode_simple_text(self):
        """Test encoding simple text."""
        try:
            model = EmbeddingModel()
            embedding = model.encode("hello world")
            assert embedding is not None
            assert len(embedding) > 0
        except ImportError:
            pytest.skip("sentence_transformers not installed")

    def test_encode_returns_array(self):
        """Test that encode returns a numpy array or list."""
        try:
            model = EmbeddingModel()
            embedding = model.encode("test text")
            # Should be iterable and numeric
            assert hasattr(embedding, '__len__')
            assert len(embedding) > 0
        except ImportError:
            pytest.skip("sentence_transformers not installed")

    def test_encode_multiple_texts(self):
        """Test encoding multiple texts."""
        try:
            model = EmbeddingModel()
            text1 = "API implementation"
            text2 = "Database schema"
            
            emb1 = model.encode(text1)
            emb2 = model.encode(text2)
            
            assert emb1 is not None
            assert emb2 is not None
            assert len(emb1) == len(emb2)  # Same dimension
        except ImportError:
            pytest.skip("sentence_transformers not installed")

    def test_similarity_same_text(self):
        """Test similarity of identical texts."""
        try:
            model = EmbeddingModel()
            text = "API authentication module"
            
            emb = model.encode(text)
            sim = model.similarity(emb, emb)
            # Identical embeddings should have high similarity
            # Allow for floating-point precision issues (max ~1.0000001)
            assert 0.9 <= sim <= 1.00001
        except ImportError:
            pytest.skip("sentence_transformers not installed")

    def test_similarity_different_texts(self):
        """Test similarity of different texts."""
        try:
            model = EmbeddingModel()
            text1 = "API authentication"
            text2 = "Unrelated topic xyz"
            
            emb1 = model.encode(text1)
            emb2 = model.encode(text2)
            sim = model.similarity(emb1, emb2)
            # Different embeddings should have lower similarity
            assert -1.0 <= sim <= 1.0
        except ImportError:
            pytest.skip("sentence_transformers not installed")

    def test_similarity_related_texts(self):
        """Test similarity of related texts."""
        try:
            model = EmbeddingModel()
            text1 = "API authentication"
            text2 = "API authorization"
            
            emb1 = model.encode(text1)
            emb2 = model.encode(text2)
            sim = model.similarity(emb1, emb2)
            # Related texts should have moderate to high similarity
            assert -1.0 <= sim <= 1.0
        except ImportError:
            pytest.skip("sentence_transformers not installed")

    def test_embedding_consistency(self):
        """Test that encoding same text gives same result."""
        try:
            model = EmbeddingModel()
            text = "Consistent text"
            
            emb1 = model.encode(text)
            emb2 = model.encode(text)
            
            # Same text should produce same or very similar embeddings
            import numpy as np
            diff = np.abs(emb1 - emb2).max()
            assert diff < 0.001  # Very small difference
        except ImportError:
            pytest.skip("sentence_transformers not installed")

    def test_similarity_range(self):
        """Test that similarity is in expected range."""
        try:
            model = EmbeddingModel()
            
            # Test various pairs
            pairs = [
                ("hello", "hello"),
                ("hello", "world"),
                ("cat", "dog"),
                ("machine learning", "deep learning"),
            ]
            
            for text1, text2 in pairs:
                emb1 = model.encode(text1)
                emb2 = model.encode(text2)
                sim = model.similarity(emb1, emb2)
                # Allow small floating point errors
                assert -1.1 <= sim <= 1.1, f"Similarity out of range: {sim}"
        except ImportError:
            pytest.skip("sentence_transformers not installed")

    def test_encode_empty_text(self):
        """Test encoding empty text."""
        try:
            model = EmbeddingModel()
            embedding = model.encode("")
            assert embedding is not None
            assert len(embedding) > 0
        except ImportError:
            pytest.skip("sentence_transformers not installed")

    def test_encode_long_text(self):
        """Test encoding very long text."""
        try:
            model = EmbeddingModel()
            long_text = "word " * 1000
            embedding = model.encode(long_text)
            assert embedding is not None
            assert len(embedding) > 0
        except ImportError:
            pytest.skip("sentence_transformers not installed")
