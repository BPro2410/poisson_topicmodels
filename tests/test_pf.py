"""Basic tests for PF model initialization and functionality."""

import numpy as np
import pytest
import scipy.sparse as sparse

from poisson_topicmodels import PF


class TestPFInitialization:
    """Test PF model initialization."""

    def test_pf_initializes_with_valid_inputs(self, small_document_term_matrix):
        """Test PF initializes correctly with valid inputs."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=5, batch_size=4)

        assert model.K == 5
        assert model.D == 20
        assert model.V == 50
        assert model.batch_size == 4

    def test_pf_stores_counts_and_vocab(self, small_document_term_matrix):
        """Test that counts and vocab are stored correctly."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=3, batch_size=4)

        assert model.counts is counts
        assert np.array_equal(model.vocab, vocab)

    def test_pf_with_various_topic_counts(self, small_document_term_matrix):
        """Test PF initialization with different topic counts."""
        counts, vocab = small_document_term_matrix

        for num_topics in [1, 2, 5, 10]:
            model = PF(counts, vocab, num_topics=num_topics, batch_size=4)
            assert model.K == num_topics

    def test_pf_with_various_batch_sizes(self, small_document_term_matrix):
        """Test PF initialization with different batch sizes."""
        counts, vocab = small_document_term_matrix

        for batch_size in [1, 4, 10, 20]:
            model = PF(counts, vocab, num_topics=5, batch_size=batch_size)
            assert model.batch_size == batch_size


class TestPFTraining:
    """Test PF model training."""

    def test_pf_has_train_step_method(self, small_document_term_matrix):
        """Test that PF has a train_step method."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=3, batch_size=4)

        assert hasattr(model, "train_step")
        assert callable(model.train_step)

    def test_pf_has_return_methods(self, small_document_term_matrix):
        """Test that PF has output return methods."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=3, batch_size=4)

        assert hasattr(model, "return_topics")
        assert hasattr(model, "return_beta")
        assert hasattr(model, "return_top_words_per_topic")


class TestPFMetrics:
    """Test PF metrics tracking."""

    def test_pf_has_metrics_attribute(self, small_document_term_matrix):
        """Test that PF has Metrics attribute."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=3, batch_size=4)

        assert hasattr(model, "Metrics")
        assert hasattr(model.Metrics, "loss")
