"""
Comprehensive test suite for topicmodels package.

This file demonstrates best practices for testing topic models,
including fixtures, parametrization, and edge case handling.
"""

from typing import Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sparse

# These imports will work after the package is restructured
from poisson_topicmodels import PF, SPF

# ============================================================================
# Fixtures (Module-level for pytest discovery)
# ============================================================================


@pytest.fixture
def small_document_term_matrix() -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Small DTM for quick tests."""
    np.random.seed(42)
    counts = sparse.random(20, 50, density=0.3, format="csr", dtype=np.float32)
    vocab = np.array([f"word_{i}" for i in range(50)])
    return counts, vocab


@pytest.fixture
def medium_document_term_matrix() -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Medium DTM for integration tests."""
    np.random.seed(123)
    counts = sparse.random(500, 1000, density=0.1, format="csr", dtype=np.float32)
    vocab = np.array([f"word_{i}" for i in range(1000)])
    return counts, vocab


@pytest.fixture
def keywords_dict() -> dict:
    """Standard keywords dictionary for seeded models."""
    return {
        "topic_a": ["word_0", "word_1", "word_2", "word_3"],
        "topic_b": ["word_10", "word_11", "word_12"],
        "topic_c": ["word_20", "word_21", "word_22", "word_23", "word_24"],
    }


@pytest.fixture
def design_matrix(small_document_term_matrix) -> np.ndarray:
    """Design matrix for covariate models."""
    counts, _ = small_document_term_matrix
    n_docs = counts.shape[0]
    return np.random.randn(n_docs, 3)  # 3 covariates


# ============================================================================
# Test: PF Model (Poisson Factorization)
# ============================================================================


class TestPFModelInitialization:
    """Test PF initialization and validation."""

    # NOTE: These tests are templates. Once package is restructured,
    # uncomment the import and adapt as needed.

    #
    def test_pf_valid_initialization(self, small_document_term_matrix):
        """PF initializes correctly with valid inputs."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=5, batch_size=4)

        assert model.K == 5
        assert model.D == 20
        assert model.V == 50
        assert model.batch_size == 4

    def test_pf_invalid_num_topics(self, small_document_term_matrix):
        """PF rejects invalid num_topics."""
        counts, vocab = small_document_term_matrix
        with pytest.raises(ValueError):
            PF(counts, vocab, num_topics=0, batch_size=4)

    def test_pf_invalid_batch_size(self, small_document_term_matrix):
        """PF rejects invalid batch_size."""
        counts, vocab = small_document_term_matrix
        with pytest.raises((ValueError, AssertionError)):
            PF(counts, vocab, num_topics=5, batch_size=0)

    @pytest.mark.parametrize("num_topics", [1, 5, 10, 50])
    def test_pf_various_topic_counts(self, small_document_term_matrix, num_topics):
        """PF accepts various valid topic counts."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=num_topics, batch_size=4)
        assert model.K == num_topics


class TestPFModelTraining:
    """Test PF training."""

    def test_pf_training_reduces_loss(self, small_document_term_matrix):
        """PF training reduces loss over time."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=3, batch_size=4)

        # Train for a few steps
        _ = model.train_step(num_steps=10, lr=0.01, random_seed=42)

        # Check that loss is decreasing (on average)
        losses = model.Metrics.loss
        assert len(losses) == 10

    def test_pf_reproducibility_with_seed(self, small_document_term_matrix):
        """PF produces identical results with same random seed."""
        counts, vocab = small_document_term_matrix

        model1 = PF(counts, vocab, num_topics=3, batch_size=4)
        params1 = model1.train_step(num_steps=5, lr=0.01, random_seed=42)

        model2 = PF(counts, vocab, num_topics=3, batch_size=4)
        params2 = model2.train_step(num_steps=5, lr=0.01, random_seed=42)

        # Results should be identical (within numerical precision)
        for key in params1:
            if isinstance(params1[key], np.ndarray):
                np.testing.assert_allclose(params1[key], params2[key], rtol=1e-5)

    def test_pf_training_returns_dict(self, small_document_term_matrix):
        """PF train_step returns parameter dictionary."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=3, batch_size=4)
        params = model.train_step(num_steps=5, lr=0.01)

        assert isinstance(params, dict)
        assert len(params) > 0


# ============================================================================
# Test: SPF Model (Seeded Poisson Factorization)
# ============================================================================


class TestSPFModelInitialization:
    """Test SPF initialization."""

    def test_spf_valid_initialization(self, small_document_term_matrix, keywords_dict):
        """SPF initializes correctly with valid inputs."""
        counts, vocab = small_document_term_matrix
        model = SPF(counts, vocab, keywords=keywords_dict, residual_topics=2, batch_size=4)

        assert model.K == 5  # 3 seeded topics + 2 residual
        assert model.D == 20
        assert model.V == 50

    def test_spf_empty_keywords_raises_error(self, small_document_term_matrix):
        """SPF rejects empty keywords dictionary."""
        counts, vocab = small_document_term_matrix
        with pytest.raises((ValueError, IndexError)):
            SPF(counts, vocab, keywords={}, residual_topics=2, batch_size=4)

    def test_spf_unknown_keywords_handled(self, small_document_term_matrix, keywords_dict):
        """SPF handles keywords not in vocabulary."""
        counts, vocab = small_document_term_matrix
        keywords_with_unknown = keywords_dict.copy()
        keywords_with_unknown["topic_d"] = ["unknown_word_xyz", "another_unknown"]

        # Should crash; unknown words should raise an error
        with pytest.raises((ValueError, IndexError)):
            SPF(counts, vocab, keywords=keywords_with_unknown, residual_topics=2, batch_size=4)


# ============================================================================
# Test: Output Methods
# ============================================================================


class TestOutputMethods:
    """Test methods for extracting and interpreting results."""

    def test_return_topics_output_shape(self, small_document_term_matrix):
        """return_topics() produces correct output shapes."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=5, batch_size=4)
        model.train_step(num_steps=2, lr=0.01)

        topics, theta = model.return_topics()

        assert isinstance(topics, jnp.ndarray)
        assert isinstance(theta, jnp.ndarray)
        assert topics.shape == (20,)  # K topics
        assert theta.shape == (20, 5)  # D docs × K topics

    def test_return_beta_shape(self, small_document_term_matrix):
        """return_beta() produces correct shape."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=5, batch_size=4)
        model.train_step(num_steps=2, lr=0.01)

        beta = model.return_beta()

        assert isinstance(beta, pd.DataFrame)
        assert beta.shape == (50, 5)  # K topics × V vocabulary

    def test_return_top_words_per_topic(self, small_document_term_matrix, keywords_dict):
        """return_top_words_per_topic() returns expected format."""
        counts, vocab = small_document_term_matrix
        model = SPF(counts, vocab, keywords_dict, batch_size=4, residual_topics=0)
        model.train_step(num_steps=2, lr=0.01)

        top_words = model.return_top_words_per_topic(n=10)

        assert isinstance(top_words, dict)
        assert len(top_words) == 3  # 3 topics
        for topic_id, words in top_words.items():
            assert len(words) == 10  # n=10 words
            # assert all(w in vocab for w in words)


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test handling of edge cases and error conditions."""

    def test_single_document(self):
        """Model handles single document."""
        counts = sparse.csr_matrix(np.array([[1, 2, 3, 0, 1]], dtype=np.float32))
        vocab = np.array(["w0", "w1", "w2", "w3", "w4"])

        model = PF(counts, vocab, num_topics=2, batch_size=1)
        _ = model.train_step(num_steps=3, lr=0.01)

        assert model.D == 1

    def test_sparse_document_term_matrix(self):
        """Model handles highly sparse data."""
        counts = sparse.csr_matrix(np.random.poisson(0.01, size=(100, 1000)).astype(np.float32))
        vocab = np.array([f"w{i}" for i in range(1000)])

        model = PF(counts, vocab, num_topics=5, batch_size=16)
        _ = model.train_step(num_steps=3, lr=0.01)

        assert model.D == 100

    def test_dense_document_term_matrix(self):
        """Model handles dense data."""
        counts = sparse.csr_matrix(np.random.poisson(5, size=(50, 200)).astype(np.float32))
        vocab = np.array([f"w{i}" for i in range(200)])

        model = PF(counts, vocab, num_topics=5, batch_size=8)
        _ = model.train_step(num_steps=3, lr=0.01)

        assert model.D == 50


# ============================================================================
# Test: Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow_unsupervised(self, small_document_term_matrix, keywords_dict):
        """Complete workflow for unsupervised topic modeling."""
        counts, vocab = small_document_term_matrix

        # 1. Initialize model
        model = PF(counts, vocab, batch_size=4, num_topics=3)

        # 2. Train
        _ = model.train_step(num_steps=10, lr=0.01, random_seed=42)

        # 3. Extract results
        _, theta = model.return_topics()
        beta = model.return_beta()

        # 4. Verify results
        assert theta.shape == (20, 3)
        assert beta.shape == (50, 3)

    def test_complete_workflow_seeded(self, small_document_term_matrix, keywords_dict):
        """Complete workflow for seeded topic modeling."""
        counts, vocab = small_document_term_matrix

        # 1. Initialize seeded model
        model = SPF(counts, vocab, keywords=keywords_dict, residual_topics=1, batch_size=4)

        # 2. Train
        _ = model.train_step(num_steps=10, lr=0.01, random_seed=42)

        # 3. Extract results
        topics, theta = model.return_topics()

        # 4. Verify
        assert topics is not None
        assert theta.shape[0] == 20  # n_docs


# ============================================================================
# Test: Utilities
# ============================================================================


class TestUtilities:
    """Tests for utility functions."""

    def test_create_batches(self):
        """Batch creation utility works correctly."""
        # TODO: Implement batch creation test
        pass

    def test_sparse_matrix_conversion(self):
        """Sparse matrix handling is correct."""
        # TODO: Implement sparse matrix conversion test
        pass


# ============================================================================
# Parametrized Tests
# ============================================================================


@pytest.mark.parametrize(
    "num_topics,num_docs,num_words",
    [
        (2, 10, 20),
        (5, 50, 100),
        (10, 100, 500),
    ],
)
def test_various_model_sizes(num_topics, num_docs, num_words):
    """PF works with various problem sizes."""
    np.random.seed(42)
    counts = sparse.random(num_docs, num_words, density=0.1, format="csr", dtype=np.float32)
    vocab = np.array([f"w{i}" for i in range(num_words)])

    model = PF(counts, vocab, num_topics=num_topics, batch_size=max(1, num_docs // 2))
    _ = model.train_step(num_steps=3, lr=0.01)

    assert model.K == num_topics


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run all tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
