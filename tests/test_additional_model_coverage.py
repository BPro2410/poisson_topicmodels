"""Additional high-value model tests for previously under-covered paths.

These tests focus on:
- dynamic model factory behavior,
- ETM initialization and probabilistic graph construction,
- TBIP time-varying setup and plotting code paths.
"""

import numpy as np
import pytest
import scipy.sparse as sparse
from jax import random
from numpyro import handlers

from poisson_topicmodels import ETM, TBIP
from poisson_topicmodels.models.topicmodels import get_base_class, topicmodels


def _small_counts_and_vocab():
    """Create a tiny deterministic corpus used across model coverage tests."""
    counts = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 2, 1, 0],
                [0, 1, 1, 0, 1],
                [2, 0, 0, 1, 1],
                [0, 2, 1, 0, 0],
            ],
            dtype=np.float32,
        )
    )
    vocab = np.array(["a", "b", "c", "d", "e"])
    return counts, vocab


def test_topicmodels_factory_mapping_and_dynamic_instance():
    """Verify factory routing for all supported model names and dynamic wrapper behavior."""
    assert get_base_class("PF").__name__ == "PF"
    assert get_base_class("SPF").__name__ == "SPF"
    assert get_base_class("CSPF").__name__ == "CSPF"
    assert get_base_class("TBIP").__name__ == "TBIP"
    assert get_base_class("TVTBIP").__name__ == "TBIP"
    assert get_base_class("CPF").__name__ == "CPF"
    assert get_base_class("ETM").__name__ == "ETM"

    with pytest.raises(ValueError):
        get_base_class("UNKNOWN")

    counts, vocab = _small_counts_and_vocab()
    model = topicmodels("PF", counts, vocab, num_topics=2, batch_size=2)
    assert model.__class__.__mro__[1].__name__ == "PF"
    assert "PF initialized with 2 topics" in repr(model)


def test_etm_init_model_guide_and_notimplemented_methods():
    """Cover ETM init/model/guide paths and explicit unimplemented API methods."""
    counts, vocab = _small_counts_and_vocab()
    embeddings_mapping = {
        "a": np.ones(4),
        "b": np.array([0.5, 0.1, -0.2, 0.3]),
    }

    model = ETM(
        counts=counts,
        vocab=vocab,
        num_topics=2,
        batch_size=2,
        embeddings_mapping=embeddings_mapping,
        embed_size=4,
    )
    assert model.rho.shape == (5, 4)
    assert np.isfinite(model.rho).all()

    y_batch = np.array([[1, 0, 2, 1, 0], [0, 1, 1, 0, 1]], dtype=np.float32)
    d_batch = np.array([0, 1])

    model_trace = handlers.trace(handlers.seed(model._model, random.PRNGKey(0))).get_trace(
        y_batch, d_batch
    )
    assert "theta" in model_trace
    assert "Y_batch" in model_trace

    guide_trace = handlers.trace(handlers.seed(model._guide, random.PRNGKey(1))).get_trace(
        y_batch, d_batch
    )
    assert "theta" in guide_trace

    with pytest.raises(ValueError, match="Model must be trained"):
        model.return_topics()
    with pytest.raises(ValueError, match="Model must be trained"):
        model.return_beta()

    with pytest.raises(ValueError, match="embeddings_mapping cannot be empty"):
        ETM(
            counts=counts,
            vocab=vocab,
            num_topics=2,
            batch_size=2,
            embeddings_mapping={},
            embed_size=4,
        )


def test_tbip_time_varying_get_batch_model_guide_and_plot():
    """Exercise TBIP time-varying branches, traceable latent variables, and plotting."""
    counts, vocab = _small_counts_and_vocab()
    authors = np.array(["alice", "bob", "alice", "carol"])
    beta_shape_init = np.ones((2, counts.shape[1]), dtype=np.float32)
    beta_rate_init = np.ones((2, counts.shape[1]), dtype=np.float32)

    with pytest.warns(UserWarning):
        model = TBIP(
            counts=counts,
            vocab=vocab,
            num_topics=2,
            authors=authors,
            batch_size=2,
            time_varying=True,
            beta_shape_init=beta_shape_init,
            beta_rate_init=beta_rate_init,
        )

    y_batch, d_batch, i_batch = model._get_batch(random.PRNGKey(3), counts)
    assert y_batch.shape == (2, counts.shape[1])
    assert d_batch.shape == (2,)
    assert i_batch.shape == (2,)

    model_trace = handlers.trace(handlers.seed(model._model, random.PRNGKey(0))).get_trace(
        y_batch, d_batch, i_batch
    )
    assert "x" in model_trace
    assert "beta" in model_trace
    assert "Y_batch" in model_trace

    guide_trace = handlers.trace(handlers.seed(model._guide, random.PRNGKey(1))).get_trace(
        y_batch, d_batch, i_batch
    )
    assert "x" in guide_trace
    assert "theta" in guide_trace

    with pytest.raises(ValueError):
        model.train_step(num_steps=0, lr=0.1)
    with pytest.raises(ValueError):
        model.train_step(num_steps=2, lr=0)

    model.estimated_params = {
        "mu_x": np.array([0.1, -0.2, 0.3], dtype=np.float32),
        "sigma_x": np.array([0.05, 0.05, 0.05], dtype=np.float32),
    }
    author_map = model._TBIP__create_author_ideal_map()
    assert set(author_map.keys()) == {"alice", "bob", "carol"}

    model.plot_ideal_points(selected_authors=["alice", "carol"])

    with pytest.raises(ValueError, match="must have shape"):
        TBIP(
            counts=counts,
            vocab=vocab,
            num_topics=2,
            authors=authors,
            batch_size=2,
            time_varying=True,
            beta_shape_init=np.ones((2, 2), dtype=np.float32),
            beta_rate_init=beta_rate_init,
        )
