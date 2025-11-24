"""Tests for poisson_topicmodels package initialization and imports."""

import pytest


def test_package_imports():
    """Test that package can be imported."""
    import poisson_topicmodels

    assert hasattr(poisson_topicmodels, "__version__")
    assert poisson_topicmodels.__version__ == "0.1.0"


def test_package_metadata():
    """Test that package has required metadata."""
    import poisson_topicmodels

    # Metadata test could be adjusted based on actual metadata
    assert hasattr(poisson_topicmodels, "PF")
    assert hasattr(poisson_topicmodels, "SPF")


def test_models_can_be_imported():
    """Test that we can import models from poisson_topicmodels."""
    from poisson_topicmodels import PF, SPF, TBIP, Metrics

    assert PF is not None
    assert SPF is not None
    assert TBIP is not None
    assert Metrics is not None


def test_models_factory_exists():
    """Test that topicmodels factory function exists."""
    from poisson_topicmodels import topicmodels as tm_factory

    assert callable(tm_factory)


def test_utils_can_be_imported():
    """Test that utility functions can be imported."""
    from poisson_topicmodels.utils.utils import create_word2vec_embedding_from_dataset

    assert callable(create_word2vec_embedding_from_dataset)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run all tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
