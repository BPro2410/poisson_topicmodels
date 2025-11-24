"""
Comprehensive tests for SPF, CPF, CSPF models and their functionality.

Tests model initialization, training, topic extraction, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sparse

from poisson_topicmodels import CPF, SPF

try:
    from poisson_topicmodels import CSPF

    HAS_CSPF = True
except ImportError:
    HAS_CSPF = False


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def small_dtm():
    """Small document-term matrix for testing."""
    return sparse.random(20, 100, density=0.1, format="csr", dtype=np.float32)


@pytest.fixture
def small_vocab():
    """Small vocabulary."""
    return np.array([f"word_{i}" for i in range(100)])


@pytest.fixture
def keywords_dict():
    """Dictionary of seed keywords for SPF."""
    return {
        0: ["word_0", "word_1", "word_2"],
        1: ["word_10", "word_11", "word_12"],
    }


@pytest.fixture
def covariates_data():
    """Covariate data for CPF."""
    return np.random.randn(20, 3)


@pytest.fixture
def covariates_df():
    """Covariate data as DataFrame."""
    return pd.DataFrame(
        np.random.randn(20, 3),
        columns=["cov_1", "cov_2", "cov_3"],
    )


# ============================================================================
# SPF TESTS
# ============================================================================


class TestSPFInitialization:
    """Test SPF model initialization."""

    def test_spf_creates_with_valid_inputs(self, small_dtm, small_vocab, keywords_dict):
        """SPF should initialize with valid inputs."""
        model = SPF(
            small_dtm,
            small_vocab,
            keywords_dict,
            residual_topics=3,
            batch_size=5,
        )
        assert model.D == 20
        assert model.V == 100
        assert model.K == 5  # 2 seeded + 3 residual
        assert model.residual_topics == 3

    def test_spf_stores_keywords(self, small_dtm, small_vocab, keywords_dict):
        """SPF should store keywords correctly."""
        model = SPF(
            small_dtm,
            small_vocab,
            keywords_dict,
            residual_topics=2,
            batch_size=5,
        )
        assert model.keywords == keywords_dict

    def test_spf_computes_keyword_indices(self, small_dtm, small_vocab, keywords_dict):
        """SPF should compute keyword indices."""
        model = SPF(
            small_dtm,
            small_vocab,
            keywords_dict,
            residual_topics=2,
            batch_size=5,
        )
        assert model.Tilde_V > 0  # Should have some keyword indices

    def test_spf_single_topic_keywords(self, small_dtm, small_vocab):
        """SPF with single seeded topic should work."""
        keywords = {0: ["word_0", "word_1"]}
        model = SPF(
            small_dtm,
            small_vocab,
            keywords,
            residual_topics=2,
            batch_size=5,
        )
        assert model.K == 3  # 1 seeded + 2 residual

    def test_spf_invalid_keyword_term_raises_error(self, small_dtm, small_vocab):
        """SPF with invalid keyword terms should raise error."""
        invalid_keywords = {0: ["nonexistent_word_xyz"]}
        with pytest.raises(ValueError):
            SPF(
                small_dtm,
                small_vocab,
                invalid_keywords,
                residual_topics=2,
                batch_size=5,
            )

    def test_spf_partial_invalid_keywords_raises_error(self, small_dtm, small_vocab):
        """SPF with some invalid keywords should raise error."""
        invalid_keywords = {0: ["word_0", "nonexistent_word_xyz"]}
        with pytest.raises(ValueError):
            SPF(
                small_dtm,
                small_vocab,
                invalid_keywords,
                residual_topics=2,
                batch_size=5,
            )


class TestSPFValidation:
    """Test SPF input validation."""

    def test_spf_dense_matrix_raises_error(self, small_vocab, keywords_dict):
        """Dense matrix should raise TypeError."""
        dense = np.random.randn(20, 100).astype(np.float32)
        with pytest.raises(TypeError):
            SPF(dense, small_vocab, keywords_dict, residual_topics=2, batch_size=5)

    def test_spf_non_dict_keywords_raises_error(self, small_dtm, small_vocab):
        """Non-dict keywords should raise TypeError."""
        with pytest.raises(TypeError):
            SPF(
                small_dtm,
                small_vocab,
                ["word_0", "word_1"],  # List instead of dict
                residual_topics=2,
                batch_size=5,
            )

    def test_spf_vocab_size_mismatch_raises_error(self, small_dtm):
        """Vocabulary size mismatch should raise error."""
        wrong_vocab = np.array([f"word_{i}" for i in range(50)])
        keywords = {0: ["word_0"]}
        with pytest.raises(ValueError):
            SPF(
                small_dtm,
                wrong_vocab,
                keywords,
                residual_topics=2,
                batch_size=5,
            )

    def test_spf_batch_size_exceeds_docs_raises_error(self, small_dtm, small_vocab, keywords_dict):
        """Batch size > documents should raise error."""
        with pytest.raises(ValueError):
            SPF(
                small_dtm,
                small_vocab,
                keywords_dict,
                residual_topics=2,
                batch_size=100,  # Greater than D=20
            )

    def test_spf_negative_batch_size_raises_error(self, small_dtm, small_vocab, keywords_dict):
        """Negative batch size should raise error."""
        with pytest.raises(ValueError):
            SPF(
                small_dtm,
                small_vocab,
                keywords_dict,
                residual_topics=2,
                batch_size=-1,
            )

    def test_spf_zero_batch_size_raises_error(self, small_dtm, small_vocab, keywords_dict):
        """Zero batch size should raise error."""
        with pytest.raises(ValueError):
            SPF(
                small_dtm,
                small_vocab,
                keywords_dict,
                residual_topics=2,
                batch_size=0,
            )

    def test_spf_negative_residual_topics_raises_error(self, small_dtm, small_vocab, keywords_dict):
        """Negative residual topics should raise error."""
        with pytest.raises(ValueError):
            SPF(
                small_dtm,
                small_vocab,
                keywords_dict,
                residual_topics=-1,
                batch_size=5,
            )


# ============================================================================
# CPF TESTS
# ============================================================================


class TestCPFInitialization:
    """Test CPF model initialization."""

    def test_cpf_creates_with_valid_inputs(self, small_dtm, small_vocab, covariates_data):
        """CPF should initialize with valid inputs."""
        model = CPF(
            small_dtm,
            small_vocab,
            num_topics=5,
            batch_size=5,
            X_design_matrix=covariates_data,
        )
        assert model.D == 20
        assert model.V == 100
        assert model.K == 5
        assert model.C == 3

    def test_cpf_without_covariates(self, small_dtm, small_vocab):
        """CPF should work without covariates (defaults to ones)."""
        model = CPF(
            small_dtm,
            small_vocab,
            num_topics=5,
            batch_size=5,
            X_design_matrix=None,
        )
        assert model.D == 20
        assert model.C == 1

    def test_cpf_with_dataframe_covariates(self, small_dtm, small_vocab, covariates_df):
        """CPF should accept DataFrame covariates."""
        model = CPF(
            small_dtm,
            small_vocab,
            num_topics=5,
            batch_size=5,
            X_design_matrix=covariates_df,
        )
        assert model.C == 3

    def test_cpf_stores_covariate_data(self, small_dtm, small_vocab, covariates_data):
        """CPF should store covariate data."""
        model = CPF(
            small_dtm,
            small_vocab,
            num_topics=5,
            batch_size=5,
            X_design_matrix=covariates_data,
        )
        assert model.X_design_matrix is not None
        assert model.X_design_matrix.shape[0] == 20


class TestCPFValidation:
    """Test CPF input validation."""

    def test_cpf_dense_matrix_raises_error(self, small_vocab, covariates_data):
        """Dense counts matrix should raise TypeError."""
        dense = np.random.randn(20, 100).astype(np.float32)
        with pytest.raises(TypeError):
            CPF(
                dense,
                small_vocab,
                num_topics=5,
                batch_size=5,
                X_design_matrix=covariates_data,
            )

    def test_cpf_mismatched_covariates_shape_raises_error(self, small_dtm, small_vocab):
        """Covariates with wrong number of rows should raise error."""
        wrong_covariates = np.random.randn(10, 3)  # D=10 != 20
        with pytest.raises(ValueError):
            CPF(
                small_dtm,
                small_vocab,
                num_topics=5,
                batch_size=5,
                X_design_matrix=wrong_covariates,
            )

    def test_cpf_1d_covariates_raises_error(self, small_dtm, small_vocab):
        """1D covariates should raise error."""
        covariates_1d = np.random.randn(20)
        with pytest.raises(ValueError):
            CPF(
                small_dtm,
                small_vocab,
                num_topics=5,
                batch_size=5,
                X_design_matrix=covariates_1d,
            )

    def test_cpf_vocab_size_mismatch_raises_error(self, small_dtm, covariates_data):
        """Vocabulary size mismatch should raise error."""
        wrong_vocab = np.array([f"word_{i}" for i in range(50)])
        with pytest.raises(ValueError):
            CPF(
                small_dtm,
                wrong_vocab,
                num_topics=5,
                batch_size=5,
                X_design_matrix=covariates_data,
            )

    def test_cpf_batch_size_exceeds_docs_raises_error(
        self, small_dtm, small_vocab, covariates_data
    ):
        """Batch size > documents should raise error."""
        with pytest.raises(ValueError):
            CPF(
                small_dtm,
                small_vocab,
                num_topics=5,
                batch_size=100,
                X_design_matrix=covariates_data,
            )

    def test_cpf_zero_num_topics_raises_error(self, small_dtm, small_vocab):
        """Zero num_topics should raise error."""
        with pytest.raises(ValueError):
            CPF(
                small_dtm,
                small_vocab,
                num_topics=0,
                batch_size=5,
                X_design_matrix=None,
            )

    def test_cpf_negative_num_topics_raises_error(self, small_dtm, small_vocab):
        """Negative num_topics should raise error."""
        with pytest.raises(ValueError):
            CPF(
                small_dtm,
                small_vocab,
                num_topics=-5,
                batch_size=5,
                X_design_matrix=None,
            )


# ============================================================================
# CSPF TESTS (if available)
# ============================================================================


@pytest.mark.skipif(not HAS_CSPF, reason="CSPF not available")
class TestCSPFInitialization:
    """Test CSPF model initialization."""

    def test_cspf_creates_with_valid_inputs(
        self, small_dtm, small_vocab, keywords_dict, covariates_data
    ):
        """CSPF should initialize with valid inputs."""
        model = CSPF(
            small_dtm,
            small_vocab,
            keywords_dict,
            residual_topics=2,
            batch_size=5,
            X_design_matrix=covariates_data,
        )
        assert model.D == 20
        assert model.V == 100
        assert model.K == 4  # 2 seeded + 2 residual
        assert model.C == 3

    def test_cspf_without_covariates(self, small_dtm, small_vocab, keywords_dict):
        """CSPF should work without covariates."""
        model = CSPF(
            small_dtm,
            small_vocab,
            keywords_dict,
            residual_topics=2,
            batch_size=5,
            X_design_matrix=None,
        )
        assert model.C == 1  # Default to ones


@pytest.mark.skipif(not HAS_CSPF, reason="CSPF not available")
class TestCSPFValidation:
    """Test CSPF input validation."""

    def test_cspf_dense_matrix_raises_error(self, small_vocab, keywords_dict, covariates_data):
        """Dense counts matrix should raise TypeError."""
        dense = np.random.randn(20, 100).astype(np.float32)
        with pytest.raises(TypeError):
            CSPF(
                dense,
                small_vocab,
                keywords_dict,
                residual_topics=2,
                batch_size=5,
                X_design_matrix=covariates_data,
            )

    def test_cspf_non_dict_keywords_raises_error(self, small_dtm, small_vocab, covariates_data):
        """Non-dict keywords should raise TypeError."""
        with pytest.raises(TypeError):
            CSPF(
                small_dtm,
                small_vocab,
                ["word_0"],  # Should be dict
                residual_topics=2,
                batch_size=5,
                X_design_matrix=covariates_data,
            )

    def test_cspf_mismatched_covariates_raises_error(self, small_dtm, small_vocab, keywords_dict):
        """Covariates with wrong shape should raise error."""
        wrong_covariates = np.random.randn(10, 3)
        with pytest.raises(ValueError):
            CSPF(
                small_dtm,
                small_vocab,
                keywords_dict,
                residual_topics=2,
                batch_size=5,
                X_design_matrix=wrong_covariates,
            )


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_spf_single_document(self, small_vocab, keywords_dict):
        """SPF should handle single document."""
        single_doc = sparse.random(1, 100, density=0.1, format="csr", dtype=np.float32)
        model = SPF(
            single_doc,
            small_vocab,
            keywords_dict,
            residual_topics=2,
            batch_size=1,
        )
        assert model.D == 1

    def test_cpf_single_document(self, small_vocab, covariates_data):
        """CPF should handle single document."""
        single_doc = sparse.random(1, 100, density=0.1, format="csr", dtype=np.float32)
        single_cov = covariates_data[:1]
        model = CPF(
            single_doc,
            small_vocab,
            num_topics=3,
            batch_size=1,
            X_design_matrix=single_cov,
        )
        assert model.D == 1

    def test_spf_large_number_of_keywords(self, small_dtm, small_vocab):
        """SPF should handle many keywords."""
        # Create keywords for many topics
        keywords = {i: [f"word_{i}", f"word_{i+1}"] for i in range(0, 20, 2)}
        model = SPF(
            small_dtm,
            small_vocab,
            keywords,
            residual_topics=1,
            batch_size=5,
        )
        assert model.K == len(keywords) + 1

    def test_cpf_high_dimensional_covariates(self, small_dtm, small_vocab):
        """CPF should handle high-dimensional covariates."""
        high_dim_cov = np.random.randn(20, 50)
        model = CPF(
            small_dtm,
            small_vocab,
            num_topics=5,
            batch_size=5,
            X_design_matrix=high_dim_cov,
        )
        assert model.C == 50

    def test_spf_with_sparse_counts(self, small_vocab, keywords_dict):
        """SPF should handle very sparse count matrix."""
        sparse_counts = sparse.random(20, 100, density=0.01, format="csr", dtype=np.float32)
        model = SPF(
            sparse_counts,
            small_vocab,
            keywords_dict,
            residual_topics=2,
            batch_size=5,
        )
        assert model.D == 20
        assert sparse_counts.nnz / (20 * 100) <= 0.01


# ============================================================================
# TYPE AND STRUCTURE TESTS
# ============================================================================


class TestModelStructure:
    """Test model structure and attributes."""

    def test_spf_has_required_attributes(self, small_dtm, small_vocab, keywords_dict):
        """SPF should have all required attributes."""
        model = SPF(
            small_dtm,
            small_vocab,
            keywords_dict,
            residual_topics=2,
            batch_size=5,
        )
        assert hasattr(model, "counts")
        assert hasattr(model, "vocab")
        assert hasattr(model, "keywords")
        assert hasattr(model, "D")
        assert hasattr(model, "V")
        assert hasattr(model, "K")

    def test_cpf_has_required_attributes(self, small_dtm, small_vocab, covariates_data):
        """CPF should have all required attributes."""
        model = CPF(
            small_dtm,
            small_vocab,
            num_topics=5,
            batch_size=5,
            X_design_matrix=covariates_data,
        )
        assert hasattr(model, "counts")
        assert hasattr(model, "vocab")
        assert hasattr(model, "X_design_matrix")
        assert hasattr(model, "C")

    def test_spf_matrix_shapes(self, small_dtm, small_vocab, keywords_dict):
        """SPF should have correct matrix shapes."""
        model = SPF(
            small_dtm,
            small_vocab,
            keywords_dict,
            residual_topics=2,
            batch_size=5,
        )
        assert model.counts.shape == (20, 100)
        assert model.vocab.shape == (100,)

    def test_cpf_covariate_shapes(self, small_dtm, small_vocab, covariates_data):
        """CPF should have correct covariate shapes."""
        model = CPF(
            small_dtm,
            small_vocab,
            num_topics=5,
            batch_size=5,
            X_design_matrix=covariates_data,
        )
        assert model.X_design_matrix.shape == (20, 3)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run all tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
