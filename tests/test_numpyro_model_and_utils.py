"""Coverage-oriented tests for NumpyroModel helper paths and utils module.

These tests validate:
- base-class error handling and plotting utilities,
- cache preparation branches (CPU/Metal guard),
- metrics helper methods,
- embedding utility I/O and input validation behavior.
"""

import numpy as np
import pytest
import scipy.sparse as sparse
from jax import random

from poisson_topicmodels import PF
from poisson_topicmodels.models.Metrics import Metrics
from poisson_topicmodels.utils.utils import (
    MemoryFriendlyFileIterator,
    create_word2vec_embedding_from_dataset,
    load_embeds,
    save_embeds,
)


def _pf_model():
    """Build a minimal PF model used to exercise NumpyroModel inherited methods."""
    counts = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 2, 1],
                [0, 1, 1, 0],
                [2, 0, 0, 1],
            ],
            dtype=np.float32,
        )
    )
    vocab = np.array(["word_a", "word_b", "word_c", "word_d"])
    return PF(counts=counts, vocab=vocab, num_topics=2, batch_size=2)


def test_numpyro_model_error_paths_and_plotting(tmp_path, monkeypatch):
    """Cover trained/untrained branches and plotting/cache helper behavior."""
    model = _pf_model()

    with pytest.raises(ValueError, match="must be trained"):
        model.return_topics()
    with pytest.raises(ValueError, match="must be trained"):
        model.return_beta()
    with pytest.raises(ValueError, match="No training loss data available"):
        model.plot_model_loss()
    with pytest.raises(ValueError, match="must be trained"):
        model.plot_topic_wordclouds()

    model.estimated_params = {
        "theta_shape": np.array([[2.0, 1.0], [1.0, 2.0], [1.5, 1.5]], dtype=np.float32),
        "theta_rate": np.ones((3, 2), dtype=np.float32),
        "beta_shape": np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=np.float32),
        "beta_rate": np.ones((2, 4), dtype=np.float32),
    }
    model.Metrics.loss = [10.0, 8.0, 6.0, 5.0]

    categories, e_theta = model.return_topics()
    assert categories.shape == (3,)
    assert e_theta.shape == (3, 2)
    beta_df = model.return_beta()
    assert beta_df.shape == (4, 2)

    top_words = model.return_top_words_per_topic(n=2)
    assert len(top_words) == 2
    assert all(len(words) == 2 for words in top_words.values())

    loss_plot_path = tmp_path / "loss.png"
    model.plot_model_loss(window=2, save_path=str(loss_plot_path))
    assert loss_plot_path.exists()

    cloud_plot_path = tmp_path / "clouds.png"
    model.plot_topic_wordclouds(n_words=3, save_path=str(cloud_plot_path))
    assert cloud_plot_path.exists()

    # Exercise Metal guard path in dense cache preparation.
    monkeypatch.setattr(
        "poisson_topicmodels.models.numpyro_model.jax.default_backend", lambda: "metal"
    )
    model._prepare_dense_cache(cache_dense_counts=True, dense_cache_max_gb=1.0)
    assert model._dense_counts_cache is None

    # Exercise non-metal path.
    monkeypatch.setattr(
        "poisson_topicmodels.models.numpyro_model.jax.default_backend", lambda: "cpu"
    )
    model._prepare_dense_cache(cache_dense_counts=True, dense_cache_max_gb=1.0)
    assert model._dense_counts_cache is not None
    y_batch, d_batch = model._get_batch(random.PRNGKey(0), model.counts)
    assert y_batch.shape == (2, 4)
    assert d_batch.shape == (2,)

    with pytest.raises(ValueError, match="dense_cache_max_gb must be > 0"):
        model.train_step(num_steps=1, lr=0.1, dense_cache_max_gb=0)


def test_metrics_reset_and_last_loss():
    """Verify Metrics convenience methods for reset and last value lookup."""
    metrics = Metrics(loss=[1.0, 2.0])
    assert metrics.last_loss() == 2.0
    metrics.reset()
    assert metrics.loss == []
    assert metrics.last_loss() is None


def test_utils_embeddings_and_iterators(tmp_path):
    """Test embedding creation in list/file modes and persistence round-trips."""
    data_file = tmp_path / "sentences.txt"
    data_file.write_text("alpha beta gamma\nbeta delta alpha\n", encoding="utf-8")

    iterator_tokens = list(MemoryFriendlyFileIterator(str(data_file)))
    assert iterator_tokens == [["alpha", "beta", "gamma"], ["beta", "delta", "alpha"]]

    # In-memory dataset branch.
    embeddings = create_word2vec_embedding_from_dataset(
        dataset=["alpha beta gamma", "beta delta alpha"],
        dim_rho=8,
        min_count=1,
        workers=1,
        negative_samples=1,
        window_size=2,
        iters=2,
    )
    assert embeddings.vector_size == 8
    assert "alpha" in embeddings.key_to_index

    # File-backed dataset branch + persistence paths.
    kv_path = tmp_path / "embeds.kv"
    embeddings_from_file = create_word2vec_embedding_from_dataset(
        dataset=str(data_file),
        dim_rho=8,
        min_count=1,
        workers=1,
        negative_samples=1,
        window_size=2,
        iters=2,
        embedding_file_path=str(kv_path),
        save_c_format_w2vec=True,
    )
    assert embeddings_from_file.vector_size == 8
    assert kv_path.exists()
    assert (tmp_path / "embeds.kv.bin").exists()
    assert (tmp_path / "embeds.kv.txt").exists()

    final_bin = tmp_path / "saved.bin"
    save_embeds(embeddings_from_file, str(final_bin))
    loaded = load_embeds(str(final_bin))
    assert loaded.vector_size == 8
    assert "beta" in loaded.key_to_index


def test_utils_input_validation_assertions(tmp_path):
    """Assert the documented validation guards in embedding utility helpers."""
    with pytest.raises(AssertionError, match="dataset must be file path or list of sentences"):
        create_word2vec_embedding_from_dataset(dataset=123)

    file_path = tmp_path / "dataset.txt"
    file_path.write_text("a b c\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="output embeddings file path must be given"):
        create_word2vec_embedding_from_dataset(dataset=str(file_path))

    with pytest.raises(AssertionError, match="output embeddings file path must be given"):
        create_word2vec_embedding_from_dataset(
            dataset=["a b c"],
            save_c_format_w2vec=True,
        )
