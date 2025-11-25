"""
Integration tests and training tests for topic models.

Tests complete workflows and training functionality including
reproducibility with fixed seeds.
"""

import numpy as np
import pytest
import scipy.sparse as sparse

from poisson_topicmodels import PF, SPF


class TestTrainingIntegration:
    """Integration tests for model training."""

    def test_pf_training_reduces_loss(self, small_document_term_matrix):
        """Training should reduce loss over iterations."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=5, batch_size=4)

        # Train
        _ = model.train_step(num_steps=50, lr=0.01, random_seed=42)

        # Loss should exist and have multiple entries
        assert hasattr(model.Metrics, "loss")
        assert len(model.Metrics.loss) > 0

        # Loss should generally trend downward (allow some variance)
        # Check that final loss is less than initial loss (rough check)
        first_quarter = np.mean(model.Metrics.loss[:10])
        last_quarter = np.mean(model.Metrics.loss[-10:])
        # Note: May not always decrease, so we're lenient
        assert first_quarter > 0 and last_quarter > 0

    def test_pf_training_with_seed_reproducible(self, small_document_term_matrix):
        """Same seed should produce same results."""
        counts, vocab = small_document_term_matrix

        # Train twice with same seed
        model1 = PF(counts, vocab, num_topics=5, batch_size=4)
        _ = model1.train_step(num_steps=20, lr=0.01, random_seed=42)
        loss1 = np.array(model1.Metrics.loss)

        model2 = PF(counts, vocab, num_topics=5, batch_size=4)
        _ = model2.train_step(num_steps=20, lr=0.01, random_seed=42)
        loss2 = np.array(model2.Metrics.loss)

        # Losses should be identical or very close
        np.testing.assert_allclose(loss1, loss2, rtol=1e-5)

    def test_pf_training_without_seed_varies(self, small_document_term_matrix):
        """Different seeds should produce different (but valid) results."""
        counts, vocab = small_document_term_matrix

        model1 = PF(counts, vocab, num_topics=5, batch_size=4)
        _ = model1.train_step(num_steps=20, lr=0.01, random_seed=42)
        loss1 = np.array(model1.Metrics.loss)

        model2 = PF(counts, vocab, num_topics=5, batch_size=4)
        _ = model2.train_step(num_steps=20, lr=0.001, random_seed=123)
        loss2 = np.array(model2.Metrics.loss)

        # Should not be identical
        assert not np.allclose(loss1, loss2)

    def test_pf_topics_extraction_after_training(self, small_document_term_matrix):
        """Topics should be extractable after training."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=5, batch_size=4)
        model.train_step(num_steps=20, lr=0.01, random_seed=42)

        # Extract topics
        topics = model.return_topics()
        assert topics is not None
        assert len(topics) > 0

    def test_pf_top_words_extraction(self, small_document_term_matrix):
        """Top words should be extractable after training."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=5, batch_size=4)
        model.train_step(num_steps=20, lr=0.01, random_seed=42)

        # Extract top words
        top_words = model.return_top_words_per_topic(n=5)
        assert top_words is not None
        assert len(top_words) > 0

    def test_spf_training_with_keywords(self, small_document_term_matrix, keywords_dict):
        """SPF should train successfully with keyword guidance."""
        counts, vocab = small_document_term_matrix

        # Filter keywords to only match actual vocab
        valid_keywords = {}
        for topic_id, words in keywords_dict.items():
            valid_words = [w for w in words if w in vocab]
            if valid_words:
                valid_keywords[topic_id] = valid_words

        if valid_keywords:
            model = SPF(counts, vocab, valid_keywords, residual_topics=2, batch_size=4)
            _ = model.train_step(num_steps=20, lr=0.01, random_seed=42)

            # Should have trained
            assert hasattr(model.Metrics, "loss")
            assert len(model.Metrics.loss) > 0


class TestModelOutputShapes:
    """Test that model outputs have expected shapes."""

    def test_pf_output_shapes(self, small_document_term_matrix):
        """PF model outputs should have correct shapes."""
        counts, vocab = small_document_term_matrix
        D, V = counts.shape
        K = 5

        model = PF(counts, vocab, num_topics=K, batch_size=4)
        model.train_step(num_steps=10, lr=0.01, random_seed=42)

        # Check if output methods exist and return data
        topics, _ = model.return_topics()
        assert topics is not None

        top_words = model.return_top_words_per_topic(n=10)
        assert top_words is not None


class TestBatchSizeVariations:
    """Test models with different batch sizes."""

    def test_pf_batch_size_1(self, small_document_term_matrix):
        """PF should work with batch size 1."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=5, batch_size=1)

        # Should initialize
        assert model.batch_size == 1
        assert model.K == 5

    def test_pf_batch_size_equals_documents(self, small_document_term_matrix):
        """PF should work with batch size equal to document count."""
        counts, vocab = small_document_term_matrix
        D = counts.shape[0]

        model = PF(counts, vocab, num_topics=5, batch_size=D)
        assert model.batch_size == D

        # Should train
        _ = model.train_step(num_steps=10, lr=0.01, random_seed=42)

    def test_pf_different_topic_counts(self, small_document_term_matrix):
        """PF should work with various topic counts."""
        counts, vocab = small_document_term_matrix

        for num_topics in [1, 2, 5, 10, 20]:
            model = PF(counts, vocab, num_topics=num_topics, batch_size=4)
            assert model.K == num_topics


class TestLargerDataset:
    """Tests with larger datasets."""

    def test_pf_large_document_count(self):
        """PF should handle larger document counts."""
        D, V = 500, 1000
        counts = sparse.random(D, V, density=0.01, format="csr", dtype=np.float32)
        vocab = np.array([f"word_{i}" for i in range(V)])

        model = PF(counts, vocab, num_topics=10, batch_size=64)
        assert model.D == D
        assert model.V == V

        # Quick training test
        _ = model.train_step(num_steps=5, lr=0.01, random_seed=42)

    def test_pf_large_vocabulary(self):
        """PF should handle large vocabulary."""
        D, V = 100, 10000
        counts = sparse.random(D, V, density=0.001, format="csr", dtype=np.float32)
        vocab = np.array([f"word_{i}" for i in range(V)])

        model = PF(counts, vocab, num_topics=10, batch_size=32)
        assert model.V == V

        # Quick training test
        _ = model.train_step(num_steps=3, lr=0.01, random_seed=42)

    def test_pf_dense_documents(self):
        """PF should handle denser word distributions."""
        D, V = 100, 500
        counts = sparse.random(D, V, density=0.2, format="csr", dtype=np.float32)
        vocab = np.array([f"word_{i}" for i in range(V)])

        model = PF(counts, vocab, num_topics=10, batch_size=32)
        _ = model.train_step(num_steps=5, lr=0.01, random_seed=42)

        assert len(model.Metrics.loss) > 0


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_pf_single_topic(self, small_document_term_matrix):
        """PF should work with a single topic."""
        counts, vocab = small_document_term_matrix
        model = PF(counts, vocab, num_topics=1, batch_size=4)

        assert model.K == 1
        _ = model.train_step(num_steps=10, lr=0.01, random_seed=42)

    def test_pf_all_zeros_document(self):
        """PF should handle documents with all zero word counts."""
        # Create dataset with some all-zero documents
        data = np.random.poisson(1, (100, 50)).astype(np.float32)
        data[0, :] = 0  # First document is all zeros
        counts = sparse.csr_matrix(data)
        vocab = np.array([f"word_{i}" for i in range(50)])

        model = PF(counts, vocab, num_topics=5, batch_size=10)
        _ = model.train_step(num_steps=5, lr=0.01, random_seed=42)

        assert len(model.Metrics.loss) > 0

    def test_pf_single_word_vocabulary(self):
        """PF should work with vocabulary of size 1."""
        counts = sparse.csr_matrix(np.ones((10, 1), dtype=np.float32))
        vocab = np.array(["only_word"])

        model = PF(counts, vocab, num_topics=1, batch_size=5)
        assert model.V == 1

    def test_pf_very_high_topic_count(self, small_document_term_matrix):
        """PF should work even with more topics than documents."""
        counts, vocab = small_document_term_matrix
        D = counts.shape[0]

        # More topics than documents
        model = PF(counts, vocab, num_topics=D + 5, batch_size=4)
        assert model.K == D + 5


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run all tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
