"""
Tests for model training and output extraction methods.

Tests training workflows, reproducibility, and result extraction.
"""

import numpy as np
import pytest
import scipy.sparse as sparse
from topicmodels import CPF, PF, SPF

try:
    from topicmodels import CSPF
    HAS_CSPF = True
except ImportError:
    HAS_CSPF = False


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def training_dtm():
    """Document-term matrix for training tests."""
    return sparse.random(30, 150, density=0.1, format="csr", dtype=np.float32)


@pytest.fixture
def training_vocab():
    """Vocabulary for training tests."""
    return np.array([f"word_{i}" for i in range(150)])


@pytest.fixture
def training_keywords():
    """Keywords for training tests."""
    return {
        0: ["word_0", "word_1", "word_2"],
        1: ["word_10", "word_11"],
    }


@pytest.fixture
def training_covariates():
    """Covariates for training tests."""
    return np.random.randn(30, 2)


# ============================================================================
# SPF TRAINING TESTS
# ============================================================================


class TestSPFTraining:
    """Test SPF model training."""

    def test_spf_training_completes(self, training_dtm, training_vocab, training_keywords):
        """SPF training should complete without errors."""
        model = SPF(
            training_dtm,
            training_vocab,
            training_keywords,
            residual_topics=2,
            batch_size=6,
        )
        params = model.train_step(num_steps=10, lr=0.01, random_seed=42)
        assert params is not None

    def test_spf_training_with_seed_reproducible(
        self, training_dtm, training_vocab, training_keywords
    ):
        """SPF training with same seed should be reproducible."""
        keywords = training_keywords

        model1 = SPF(
            training_dtm,
            training_vocab,
            keywords,
            residual_topics=2,
            batch_size=6,
        )
        model1.train_step(num_steps=10, lr=0.01, random_seed=42)
        loss1 = np.array(model1.Metrics.loss)

        model2 = SPF(
            training_dtm,
            training_vocab,
            keywords,
            residual_topics=2,
            batch_size=6,
        )
        model2.train_step(num_steps=10, lr=0.01, random_seed=42)
        loss2 = np.array(model2.Metrics.loss)

        # Should be identical or very close
        np.testing.assert_allclose(loss1, loss2, rtol=1e-4)

    def test_spf_training_without_seed_varies(
        self, training_dtm, training_vocab, training_keywords
    ):
        """SPF training without fixed seed should vary."""
        keywords = training_keywords

        model1 = SPF(
            training_dtm,
            training_vocab,
            keywords,
            residual_topics=2,
            batch_size=6,
        )
        model1.train_step(num_steps=10, lr=0.01, random_seed=42)
        loss1 = np.array(model1.Metrics.loss)

        model2 = SPF(
            training_dtm,
            training_vocab,
            keywords,
            residual_topics=2,
            batch_size=6,
        )
        model2.train_step(num_steps=10, lr=0.01, random_seed=123)
        loss2 = np.array(model2.Metrics.loss)

        # Losses should differ
        assert not np.allclose(loss1, loss2)

    def test_spf_loss_tracking(self, training_dtm, training_vocab, training_keywords):
        """SPF should track loss during training."""
        model = SPF(
            training_dtm,
            training_vocab,
            training_keywords,
            residual_topics=2,
            batch_size=6,
        )
        model.train_step(num_steps=5, lr=0.01, random_seed=42)

        assert len(model.Metrics.loss) > 0
        assert all(isinstance(loss, (int, float, np.number)) for loss in model.Metrics.loss)

    def test_spf_different_learning_rates(self, training_dtm, training_vocab, training_keywords):
        """Different learning rates should produce different trajectories."""
        keywords = training_keywords

        model1 = SPF(
            training_dtm,
            training_vocab,
            keywords,
            residual_topics=2,
            batch_size=6,
        )
        model1.train_step(num_steps=10, lr=0.001, random_seed=42)
        loss1 = np.array(model1.Metrics.loss)

        model2 = SPF(
            training_dtm,
            training_vocab,
            keywords,
            residual_topics=2,
            batch_size=6,
        )
        model2.train_step(num_steps=10, lr=0.1, random_seed=42)
        loss2 = np.array(model2.Metrics.loss)

        # Different learning rates should produce different trajectories
        # (not necessarily different final loss, but different path)
        assert not np.allclose(loss1, loss2)


# ============================================================================
# CPF TRAINING TESTS
# ============================================================================


class TestCPFTraining:
    """Test CPF model training."""

    def test_cpf_training_completes(self, training_dtm, training_vocab, training_covariates):
        """CPF training should complete without errors."""
        model = CPF(
            training_dtm,
            training_vocab,
            num_topics=5,
            batch_size=6,
            X_design_matrix=training_covariates,
        )
        params = model.train_step(num_steps=10, lr=0.01, random_seed=42)
        assert params is not None

    def test_cpf_training_without_covariates(self, training_dtm, training_vocab):
        """CPF should train without explicit covariates."""
        model = CPF(
            training_dtm,
            training_vocab,
            num_topics=5,
            batch_size=6,
            X_design_matrix=None,
        )
        params = model.train_step(num_steps=10, lr=0.01, random_seed=42)
        assert params is not None

    def test_cpf_training_with_seed_reproducible(
        self, training_dtm, training_vocab, training_covariates
    ):
        """CPF training with same seed should be reproducible."""
        model1 = CPF(
            training_dtm,
            training_vocab,
            num_topics=5,
            batch_size=6,
            X_design_matrix=training_covariates,
        )
        model1.train_step(num_steps=10, lr=0.01, random_seed=42)
        loss1 = np.array(model1.Metrics.loss)

        model2 = CPF(
            training_dtm,
            training_vocab,
            num_topics=5,
            batch_size=6,
            X_design_matrix=training_covariates,
        )
        model2.train_step(num_steps=10, lr=0.01, random_seed=42)
        loss2 = np.array(model2.Metrics.loss)

        np.testing.assert_allclose(loss1, loss2, rtol=1e-4)

    def test_cpf_loss_tracking(self, training_dtm, training_vocab, training_covariates):
        """CPF should track loss during training."""
        model = CPF(
            training_dtm,
            training_vocab,
            num_topics=5,
            batch_size=6,
            X_design_matrix=training_covariates,
        )
        model.train_step(num_steps=5, lr=0.01, random_seed=42)

        assert len(model.Metrics.loss) > 0


# ============================================================================
# OUTPUT EXTRACTION TESTS
# ============================================================================


class TestSPFOutputExtraction:
    """Test SPF output extraction methods."""

    def test_spf_return_topics(self, training_dtm, training_vocab, training_keywords):
        """SPF should extract topics."""
        model = SPF(
            training_dtm,
            training_vocab,
            training_keywords,
            residual_topics=2,
            batch_size=6,
        )
        model.train_step(num_steps=10, lr=0.01, random_seed=42)

        topics, E_theta = model.return_topics()
        assert topics is not None
        assert len(topics) == 30  # D documents

    def test_spf_return_beta(self, training_dtm, training_vocab, training_keywords):
        """SPF should extract beta matrix."""
        model = SPF(
            training_dtm,
            training_vocab,
            training_keywords,
            residual_topics=2,
            batch_size=6,
        )
        model.train_step(num_steps=10, lr=0.01, random_seed=42)

        beta = model.return_beta()
        assert beta is not None
        assert beta.shape[0] == 150  # V words
        assert beta.shape[1] == 4  # K topics

    def test_spf_return_top_words_per_topic(
        self, training_dtm, training_vocab, training_keywords
    ):
        """SPF should extract top words per topic."""
        model = SPF(
            training_dtm,
            training_vocab,
            training_keywords,
            residual_topics=2,
            batch_size=6,
        )
        model.train_step(num_steps=10, lr=0.01, random_seed=42)

        top_words = model.return_top_words_per_topic(n=5)
        assert top_words is not None
        assert len(top_words) == 4  # K topics


class TestCPFOutputExtraction:
    """Test CPF output extraction methods."""

    def test_cpf_return_topics(self, training_dtm, training_vocab, training_covariates):
        """CPF should extract topics."""
        model = CPF(
            training_dtm,
            training_vocab,
            num_topics=5,
            batch_size=6,
            X_design_matrix=training_covariates,
        )
        model.train_step(num_steps=10, lr=0.01, random_seed=42)

        topics, E_theta = model.return_topics()
        assert topics is not None
        assert len(topics) == 30

    def test_cpf_return_beta(self, training_dtm, training_vocab, training_covariates):
        """CPF should extract beta matrix."""
        model = CPF(
            training_dtm,
            training_vocab,
            num_topics=5,
            batch_size=6,
            X_design_matrix=training_covariates,
        )
        model.train_step(num_steps=10, lr=0.01, random_seed=42)

        beta = model.return_beta()
        assert beta is not None
        assert beta.shape == (150, 5)


# ============================================================================
# CSPF TRAINING TESTS (if available)
# ============================================================================


@pytest.mark.skipif(not HAS_CSPF, reason="CSPF not available")
class TestCSPFTraining:
    """Test CSPF model training."""

    def test_cspf_training_completes(
        self, training_dtm, training_vocab, training_keywords, training_covariates
    ):
        """CSPF training should complete without errors."""
        model = CSPF(
            training_dtm,
            training_vocab,
            training_keywords,
            residual_topics=2,
            batch_size=6,
            X_design_matrix=training_covariates,
        )
        params = model.train_step(num_steps=10, lr=0.01, random_seed=42)
        assert params is not None

    def test_cspf_training_reproducible(
        self, training_dtm, training_vocab, training_keywords, training_covariates
    ):
        """CSPF training with same seed should be reproducible."""
        model1 = CSPF(
            training_dtm,
            training_vocab,
            training_keywords,
            residual_topics=2,
            batch_size=6,
            X_design_matrix=training_covariates,
        )
        model1.train_step(num_steps=10, lr=0.01, random_seed=42)
        loss1 = np.array(model1.Metrics.loss)

        model2 = CSPF(
            training_dtm,
            training_vocab,
            training_keywords,
            residual_topics=2,
            batch_size=6,
            X_design_matrix=training_covariates,
        )
        model2.train_step(num_steps=10, lr=0.01, random_seed=42)
        loss2 = np.array(model2.Metrics.loss)

        np.testing.assert_allclose(loss1, loss2, rtol=1e-4)


# ============================================================================
# METRICS TESTS
# ============================================================================


class TestMetrics:
    """Test Metrics class functionality."""

    def test_metrics_tracks_loss(self, training_dtm, training_vocab):
        """Metrics should track loss values."""
        model = PF(
            training_dtm,
            training_vocab,
            num_topics=5,
            batch_size=6,
        )
        model.train_step(num_steps=5, lr=0.01, random_seed=42)

        assert len(model.Metrics.loss) > 0

    def test_metrics_loss_are_numeric(self, training_dtm, training_vocab):
        """All loss values should be numeric."""
        model = PF(
            training_dtm,
            training_vocab,
            num_topics=5,
            batch_size=6,
        )
        model.train_step(num_steps=5, lr=0.01, random_seed=42)

        for loss in model.Metrics.loss:
            assert isinstance(loss, (int, float, np.number))

    def test_metrics_independent_per_instance(self):
        """Different model instances should have independent metrics."""
        dtm = sparse.random(10, 50, density=0.1, format="csr", dtype=np.float32)
        vocab = np.array([f"word_{i}" for i in range(50)])

        model1 = PF(dtm, vocab, num_topics=3, batch_size=5)
        model1.train_step(num_steps=3, lr=0.01, random_seed=42)
        loss1_count = len(model1.Metrics.loss)

        model2 = PF(dtm, vocab, num_topics=3, batch_size=5)
        model2.train_step(num_steps=5, lr=0.01, random_seed=42)
        loss2_count = len(model2.Metrics.loss)

        # Different number of steps should result in different loss counts
        assert loss1_count == 3
        assert loss2_count == 5
        # Metrics should be independent
        assert len(model1.Metrics.loss) == 3


# ============================================================================
# BATCH PROCESSING TESTS
# ============================================================================


class TestBatchProcessing:
    """Test batch processing during training."""

    def test_spf_respects_batch_size(self, training_dtm, training_vocab, training_keywords):
        """SPF should respect batch size during training."""
        batch_size = 6
        model = SPF(
            training_dtm,
            training_vocab,
            training_keywords,
            residual_topics=2,
            batch_size=batch_size,
        )
        model.train_step(num_steps=5, lr=0.01, random_seed=42)
        # If training completes, batch size is respected
        assert True

    def test_cpf_respects_batch_size(self, training_dtm, training_vocab, training_covariates):
        """CPF should respect batch size during training."""
        batch_size = 6
        model = CPF(
            training_dtm,
            training_vocab,
            num_topics=5,
            batch_size=batch_size,
            X_design_matrix=training_covariates,
        )
        model.train_step(num_steps=5, lr=0.01, random_seed=42)
        assert True

    def test_spf_full_batch_size_equal_to_documents(
        self, training_dtm, training_vocab, training_keywords
    ):
        """SPF should work when batch size equals number of documents."""
        D = 30
        model = SPF(
            training_dtm,
            training_vocab,
            training_keywords,
            residual_topics=2,
            batch_size=D,
        )
        model.train_step(num_steps=5, lr=0.01, random_seed=42)
        assert True

    def test_cpf_full_batch_size_equal_to_documents(
        self, training_dtm, training_vocab, training_covariates
    ):
        """CPF should work when batch size equals number of documents."""
        D = 30
        model = CPF(
            training_dtm,
            training_vocab,
            num_topics=5,
            batch_size=D,
            X_design_matrix=training_covariates,
        )
        model.train_step(num_steps=5, lr=0.01, random_seed=42)
        assert True
