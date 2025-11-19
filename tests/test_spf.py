"""Basic tests for SPF model initialization."""

import pytest
import numpy as np
import scipy.sparse as sparse
from packages.models import SPF


class TestSPFInitialization:
    """Test SPF model initialization."""

    def test_spf_initializes_with_valid_inputs(self, small_document_term_matrix, keywords_dict):
        """Test SPF initializes correctly with valid inputs."""
        counts, vocab = small_document_term_matrix
        model = SPF(counts, vocab, keywords=keywords_dict, residual_topics=2, batch_size=4)
        
        # SPF should have K = number of seeded topics + residual topics
        assert model.K == len(keywords_dict) + 2  # 3 seeded + 2 residual
        assert model.D == 20
        assert model.V == 50
        assert model.batch_size == 4

    def test_spf_stores_keywords(self, small_document_term_matrix, keywords_dict):
        """Test that keywords are stored correctly."""
        counts, vocab = small_document_term_matrix
        model = SPF(counts, vocab, keywords=keywords_dict, residual_topics=1, batch_size=4)
        
        assert model.keywords == keywords_dict
        assert model.residual_topics == 1

    def test_spf_with_various_residual_topics(self, small_document_term_matrix, keywords_dict):
        """Test SPF with different numbers of residual topics."""
        counts, vocab = small_document_term_matrix
        
        for residual in [0, 1, 2, 5]:
            model = SPF(counts, vocab, keywords=keywords_dict, residual_topics=residual, batch_size=4)
            assert model.K == len(keywords_dict) + residual


class TestSPFTraining:
    """Test SPF model training."""

    def test_spf_has_train_step_method(self, small_document_term_matrix, keywords_dict):
        """Test that SPF has a train_step method."""
        counts, vocab = small_document_term_matrix
        model = SPF(counts, vocab, keywords=keywords_dict, residual_topics=2, batch_size=4)
        
        assert hasattr(model, 'train_step')
        assert callable(model.train_step)

    def test_spf_has_return_methods(self, small_document_term_matrix, keywords_dict):
        """Test that SPF has output return methods."""
        counts, vocab = small_document_term_matrix
        model = SPF(counts, vocab, keywords=keywords_dict, residual_topics=2, batch_size=4)
        
        assert hasattr(model, 'return_topics')
        assert hasattr(model, 'return_beta')
        assert hasattr(model, 'return_top_words_per_topic')
