"""Test fixtures and configuration for topicmodels tests."""

import numpy as np
import pytest
import scipy.sparse as sparse


@pytest.fixture
def small_document_term_matrix():
    """Small DTM for quick tests."""
    np.random.seed(42)
    counts = sparse.random(20, 50, density=0.3, format="csr", dtype=np.float32)
    vocab = np.array([f"word_{i}" for i in range(50)])
    return counts, vocab


@pytest.fixture
def medium_document_term_matrix():
    """Medium DTM for integration tests."""
    np.random.seed(123)
    counts = sparse.random(100, 200, density=0.15, format="csr", dtype=np.float32)
    vocab = np.array([f"word_{i}" for i in range(200)])
    return counts, vocab


@pytest.fixture
def keywords_dict():
    """Standard keywords dictionary for seeded models."""
    return {
        "topic_a": ["word_0", "word_1", "word_2", "word_3"],
        "topic_b": ["word_10", "word_11", "word_12"],
        "topic_c": ["word_20", "word_21", "word_22", "word_23", "word_24"],
    }


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run all tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
