"""Tests for topicmodels package initialization and imports."""

import pytest


def test_package_imports():
    """Test that package can be imported."""
    import topicmodels

    assert hasattr(topicmodels, "__version__")
    assert topicmodels.__version__ == "0.1.0"


def test_package_metadata():
    """Test that package has required metadata."""
    import topicmodels

    assert hasattr(topicmodels, "__author__")
    assert hasattr(topicmodels, "__email__")
    assert "Bernd Prostmaier" in topicmodels.__author__


def test_models_can_be_imported():
    """Test that we can import models from packages.models."""
    from packages.models import PF, SPF, TBIP, Metrics

    assert PF is not None
    assert SPF is not None
    assert TBIP is not None
    assert Metrics is not None


def test_models_factory_exists():
    """Test that topicmodels factory function exists."""
    from packages.models import topicmodels as tm_factory

    assert callable(tm_factory)


def test_utils_can_be_imported():
    """Test that utility functions can be imported."""
    from packages.utils.utils import create_word2vec_embedding_from_dataset

    assert callable(create_word2vec_embedding_from_dataset)
