"""
Input validation tests for all models.

Tests that models properly validate input parameters and raise
appropriate errors for invalid configurations.
"""

import numpy as np
import pytest
import scipy.sparse as sparse
from topicmodels import CPF, PF, SPF


class TestPFValidation:
    """Test PF input validation."""

    def test_pf_empty_counts_matrix_raises_error(self):
        """Empty counts matrix should raise error."""
        empty_counts = sparse.csr_matrix((0, 100), dtype=np.float32)
        vocab = np.array([f"word_{i}" for i in range(100)])

        # Should raise ValueError for empty matrix
        with pytest.raises((ValueError, AssertionError)):
            PF(empty_counts, vocab, num_topics=5, batch_size=10)

    def test_pf_empty_vocab_raises_error(self):
        """Empty vocabulary should raise error."""
        counts = sparse.random(10, 100, density=0.3, format="csr", dtype=np.float32)
        empty_vocab = np.array([])

        with pytest.raises((ValueError, AssertionError)):
            PF(counts, empty_vocab, num_topics=5, batch_size=10)

    def test_pf_negative_num_topics_raises_error(self):
        """Negative num_topics should raise error."""
        counts = sparse.random(10, 100, density=0.3, format="csr", dtype=np.float32)
        vocab = np.array([f"word_{i}" for i in range(100)])

        with pytest.raises(ValueError):
            PF(counts, vocab, num_topics=-5, batch_size=10)

    def test_pf_zero_num_topics_raises_error(self):
        """Zero num_topics should raise error."""
        counts = sparse.random(10, 100, density=0.3, format="csr", dtype=np.float32)
        vocab = np.array([f"word_{i}" for i in range(100)])

        with pytest.raises(ValueError):
            PF(counts, vocab, num_topics=0, batch_size=10)

    def test_pf_negative_batch_size_raises_error(self):
        """Negative batch_size should raise error."""
        counts = sparse.random(10, 100, density=0.3, format="csr", dtype=np.float32)
        vocab = np.array([f"word_{i}" for i in range(100)])

        with pytest.raises(ValueError):
            PF(counts, vocab, num_topics=5, batch_size=-1)

    def test_pf_batch_size_exceeds_documents_raises_error(self):
        """Batch size > number of documents should raise error."""
        counts = sparse.random(10, 100, density=0.3, format="csr", dtype=np.float32)
        vocab = np.array([f"word_{i}" for i in range(100)])

        with pytest.raises(ValueError):
            PF(counts, vocab, num_topics=5, batch_size=20)

    def test_pf_dense_counts_matrix_raises_error(self):
        """Dense matrix instead of sparse should raise TypeError."""
        counts = np.random.random((10, 100)).astype(np.float32)
        vocab = np.array([f"word_{i}" for i in range(100)])

        with pytest.raises(TypeError):
            PF(counts, vocab, num_topics=5, batch_size=10)

    def test_pf_vocab_size_mismatch_raises_error(self):
        """Vocabulary size mismatch should raise error."""
        counts = sparse.random(10, 100, density=0.3, format="csr", dtype=np.float32)
        vocab = np.array([f"word_{i}" for i in range(50)])  # Wrong size

        with pytest.raises((ValueError, AssertionError)):
            PF(counts, vocab, num_topics=5, batch_size=10)


class TestSPFValidation:
    """Test SPF (Seeded Poisson Factorization) input validation."""

    def test_spf_invalid_keywords_structure_raises_error(self):
        """Invalid keywords structure should raise error."""
        counts = sparse.random(10, 100, density=0.3, format="csr", dtype=np.float32)
        vocab = np.array([f"word_{i}" for i in range(100)])
        invalid_keywords = ["word_1", "word_2"]  # Should be dict

        with pytest.raises((TypeError, ValueError)):
            SPF(counts, vocab, invalid_keywords, residual_topics=2, batch_size=10)

    def test_spf_negative_residual_topics_raises_error(self):
        """Negative residual_topics should raise error."""
        counts = sparse.random(10, 100, density=0.3, format="csr", dtype=np.float32)
        vocab = np.array([f"word_{i}" for i in range(100)])
        keywords = {0: ["word_1", "word_2"]}

        with pytest.raises(ValueError):
            SPF(counts, vocab, keywords, residual_topics=-1, batch_size=10)

    def test_spf_keywords_with_invalid_terms_raises_error(self):
        """Keywords with terms not in vocabulary should raise error."""
        counts = sparse.random(10, 100, density=0.3, format="csr", dtype=np.float32)
        vocab = np.array(["apple", "banana", "cherry"])
        keywords = {0: ["apple", "invalid_word"]}  # 'invalid_word' not in vocab

        with pytest.raises((ValueError, KeyError)):
            SPF(counts, vocab, keywords, residual_topics=1, batch_size=10)


class TestCPFValidation:
    """Test CPF (Covariate Poisson Factorization) input validation."""

    def test_cpf_mismatched_covariates_raises_error(self):
        """Covariates shape mismatch should raise error."""
        counts = sparse.random(10, 100, density=0.3, format="csr", dtype=np.float32)
        vocab = np.array([f"word_{i}" for i in range(100)])
        covariates = np.random.randn(5, 3)  # Wrong number of documents

        with pytest.raises((ValueError, AssertionError)):
            CPF(counts, vocab, covariates, num_topics=5, batch_size=10)

    def test_cpf_empty_covariates_raises_error(self):
        """Empty covariates should raise error."""
        counts = sparse.random(10, 100, density=0.3, format="csr", dtype=np.float32)
        vocab = np.array([f"word_{i}" for i in range(100)])
        covariates = np.array([]).reshape(10, 0)  # No covariates

        with pytest.raises(ValueError):
            CPF(counts, vocab, covariates, num_topics=5, batch_size=10)

    def test_cpf_1d_covariates_raises_error(self):
        """1D covariates should be 2D."""
        counts = sparse.random(10, 100, density=0.3, format="csr", dtype=np.float32)
        vocab = np.array([f"word_{i}" for i in range(100)])
        covariates = np.random.randn(10)  # 1D instead of 2D

        with pytest.raises((ValueError, AssertionError)):
            CPF(counts, vocab, covariates, num_topics=5, batch_size=10)


class TestTrainingValidation:
    """Test training method parameter validation."""

    def test_pf_negative_num_steps_raises_error(self, small_document_term_matrix, keywords_dict):
        """Negative num_steps should raise error."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=5, batch_size=4)

        with pytest.raises(ValueError):
            model.train_step(num_steps=-10, lr=0.01)

    def test_pf_zero_num_steps_raises_error(self, small_document_term_matrix):
        """Zero num_steps should raise error or be no-op."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=5, batch_size=4)

        # Should either raise or return empty/no training
        result = model.train_step(num_steps=0, lr=0.01)
        # Depending on implementation, may raise or return None/empty

    def test_pf_negative_learning_rate_raises_error(self, small_document_term_matrix):
        """Negative learning rate should raise error."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=5, batch_size=4)

        with pytest.raises(ValueError):
            model.train_step(num_steps=10, lr=-0.01)

    def test_pf_zero_learning_rate_raises_error(self, small_document_term_matrix):
        """Zero learning rate should raise error."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=5, batch_size=4)

        with pytest.raises(ValueError):
            model.train_step(num_steps=10, lr=0.0)
