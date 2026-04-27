"""Tests to boost coverage for new post-fitting inspection methods.

Covers:
- NumpyroModel: summary(), compute_topic_coherence(), compute_topic_diversity(),
  plot_topic_prevalence(), plot_topic_correlation(), plot_document_topic_heatmap(),
  plot_wordclouds(save_path=...), _prepare_dense_cache() branches
- PF: summary() (base _summary_extra)
- SPF: summary(), plot_seed_effectiveness(), validation branches
- CPF: summary(), return_covariate_effects(), return_covariate_effects_ci(),
  plot_cov_effects(), validation branches
- CSPF: summary(), return_topics(), return_beta(), return_covariate_effects(),
  return_covariate_effects_ci(), plot_cov_effects(), plot_cov_effects(include_shrinkage),
  _build_group_index separators, validation branches
- TBIP: summary(), return_topics(), return_beta(), return_ideal_points(),
  return_ideological_words(), plot_ideal_points(show_ci), validation branches
- ETM: summary(), return_topics(), return_beta(), validation branches
- utils: train_word2vec debug_mode branches
"""

import os
import tempfile
from unittest.mock import patch

import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sparse

from poisson_topicmodels import CPF, ETM, PF, SPF, TBIP
from poisson_topicmodels.models.CSPF import CSPF
from poisson_topicmodels.models.Metrics import Metrics as TopicModelMetrics

matplotlib.use("Agg")  # non-interactive backend for CI

# ============================================================================
# Shared helpers
# ============================================================================


def _counts_vocab(D=20, V=50):
    """Create a small sparse DTM and vocabulary."""
    np.random.seed(42)
    counts = sparse.random(D, V, density=0.3, format="csr", dtype=np.float32)
    vocab = np.array([f"word_{i}" for i in range(V)])
    return counts, vocab


def _keywords(vocab_prefix="word_"):
    return {
        "topic_a": [f"{vocab_prefix}0", f"{vocab_prefix}1", f"{vocab_prefix}2"],
        "topic_b": [f"{vocab_prefix}10", f"{vocab_prefix}11", f"{vocab_prefix}12"],
    }


def _make_pf_with_params(D=20, V=50, K=3):
    counts, vocab = _counts_vocab(D, V)
    model = PF(counts, vocab, num_topics=K, batch_size=4)
    model.estimated_params = {
        "theta_shape": np.random.rand(D, K).astype(np.float32) + 0.1,
        "theta_rate": np.ones((D, K), dtype=np.float32),
        "beta_shape": np.random.rand(K, V).astype(np.float32) + 0.1,
        "beta_rate": np.ones((K, V), dtype=np.float32),
    }
    model.Metrics = TopicModelMetrics(loss=[100.0, 80.0, 60.0])
    return model


def _make_spf_with_params(D=20, V=50):
    counts, vocab = _counts_vocab(D, V)
    keywords = _keywords()
    K = len(keywords) + 1  # 1 residual
    model = SPF(counts, vocab, keywords=keywords, residual_topics=1, batch_size=4)

    Tilde_V = model.Tilde_V
    model.estimated_params = {
        "theta_shape": jnp.array(np.random.rand(D, K).astype(np.float32) + 0.1),
        "theta_rate": jnp.ones((D, K), dtype=np.float32),
        "beta_shape": jnp.array(np.random.rand(K, V).astype(np.float32) + 0.1),
        "beta_rate": jnp.ones((K, V), dtype=np.float32),
        "beta_tilde_shape": jnp.ones(Tilde_V, dtype=np.float32) * 2,
        "beta_tilde_rate": jnp.ones(Tilde_V, dtype=np.float32),
    }
    model.Metrics = TopicModelMetrics(loss=[100.0, 70.0, 50.0])
    return model


def _make_cpf_with_params(D=20, V=50, K=3, C=2):
    counts, vocab = _counts_vocab(D, V)
    covariates = np.random.randn(D, C).astype(np.float32)
    model = CPF(counts, vocab, num_topics=K, batch_size=4, X_design_matrix=covariates)
    model.estimated_params = {
        "theta_shape": np.random.rand(D, K).astype(np.float32) + 0.1,
        "theta_rate": np.ones((D, K), dtype=np.float32),
        "beta_shape": np.random.rand(K, V).astype(np.float32) + 0.1,
        "beta_rate": np.ones((K, V), dtype=np.float32),
        "lambda_location": np.random.randn(C, K).astype(np.float32),
        "lambda_scale": np.abs(np.random.randn(C, K).astype(np.float32)) + 0.01,
    }
    model.Metrics = TopicModelMetrics(loss=[100.0, 60.0])
    return model


def _make_cspf_with_params(D=20, V=50):
    counts, vocab = _counts_vocab(D, V)
    keywords = _keywords()
    K = len(keywords) + 1  # 1 residual topic
    C = 2
    covariates = np.random.randn(D, C).astype(np.float32)
    model = CSPF(
        counts,
        vocab,
        keywords=keywords,
        residual_topics=1,
        batch_size=4,
        X_design_matrix=covariates,
    )
    G = model.G
    Tilde_V = model.Tilde_V

    model.estimated_params = {
        "theta_shape": jnp.array(np.random.rand(D, K).astype(np.float32) + 0.1),
        "theta_rate": jnp.ones((D, K), dtype=np.float32),
        "beta_shape": jnp.array(np.random.rand(K, V).astype(np.float32) + 0.1),
        "beta_rate": jnp.ones((K, V), dtype=np.float32),
        "beta_tilde_shape": jnp.ones(Tilde_V, dtype=np.float32) * 2,
        "beta_tilde_rate": jnp.ones(Tilde_V, dtype=np.float32),
        "lambda_location": np.random.randn(C, K).astype(np.float32),
        "lambda_scale": np.abs(np.random.randn(C, K).astype(np.float32)) + 0.01,
        "lambda_intercept_location": np.random.randn(K).astype(np.float32),
        "lambda_intercept_scale": np.abs(np.random.randn(K).astype(np.float32)) + 0.01,
        "tau2_shape": np.ones(K, dtype=np.float32) * 2,
        "tau2_rate": np.ones(K, dtype=np.float32),
        "delta2_shape": np.ones((G, K), dtype=np.float32) * 2,
        "delta2_rate": np.ones((G, K), dtype=np.float32),
    }
    model.Metrics = TopicModelMetrics(loss=[200.0, 150.0, 100.0])
    return model


def _make_tbip_with_params(D=20, V=50, K=2):
    counts, vocab = _counts_vocab(D, V)
    authors = np.array([f"author_{i % 5}" for i in range(D)])
    model = TBIP(counts, vocab, num_topics=K, authors=authors, batch_size=4)
    N = len(model.authors_unique)

    model.estimated_params = {
        "mu_theta": np.random.randn(D, K).astype(np.float32),
        "sigma_theta": np.abs(np.random.randn(D, K).astype(np.float32)) + 0.01,
        "mu_beta": np.random.randn(K, V).astype(np.float32),
        "sigma_beta": np.abs(np.random.randn(K, V).astype(np.float32)) + 0.01,
        "mu_x": np.random.randn(N).astype(np.float32),
        "sigma_x": np.abs(np.random.randn(N).astype(np.float32)) + 0.01,
        "mu_eta": np.random.randn(K, V).astype(np.float32),
        "sigma_eta": np.abs(np.random.randn(K, V).astype(np.float32)) + 0.01,
    }
    model.Metrics = TopicModelMetrics(loss=[300.0, 200.0, 100.0])
    return model


def _make_etm_with_params(D=20, V=50, K=2, embed_size=4):
    counts, vocab = _counts_vocab(D, V)
    embeddings_mapping = {
        f"word_{i}": np.random.randn(embed_size).astype(np.float32) for i in range(V)
    }
    model = ETM(
        counts,
        vocab,
        num_topics=K,
        batch_size=4,
        embeddings_mapping=embeddings_mapping,
        embed_size=embed_size,
    )

    # Build real encoder params by running the encoder once
    import jax

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, V))
    enc_params = model.encoder.init(rng, dummy_input)["params"]

    # Flatten params to estimated_params format
    flat_params = {}
    flat_params["alpha"] = np.random.randn(embed_size, K).astype(np.float32)

    def _flatten(prefix, d):
        for k, v in d.items():
            key = f"{prefix}${k}"
            if isinstance(v, dict):
                _flatten(key, v)
            else:
                flat_params[key] = np.asarray(v)

    _flatten("encoder$params", enc_params)

    model.estimated_params = flat_params
    model.Metrics = TopicModelMetrics(loss=[500.0, 300.0, 100.0])
    return model


# ============================================================================
# NumpyroModel base class tests (via PF)
# ============================================================================


class TestSummary:
    """Tests for model.summary() and _summary_extra()."""

    def test_pf_summary_with_params(self):
        model = _make_pf_with_params()
        result = model.summary()
        assert "PF" in result
        assert "Topics (K)" in result
        assert "Initial ELBO loss" in result
        assert "Top words per topic" in result

    def test_pf_summary_without_training(self):
        counts, vocab = _counts_vocab()
        model = PF(counts, vocab, num_topics=3, batch_size=4)
        result = model.summary()
        assert "not been trained" in result

    def test_spf_summary_extra(self):
        model = _make_spf_with_params()
        result = model.summary()
        assert "Seeded topics" in result
        assert "Residual topics" in result
        assert "Keyword groups" in result

    def test_cpf_summary_extra(self):
        model = _make_cpf_with_params()
        result = model.summary()
        assert "Covariates (C)" in result
        assert "Covariate names" in result

    def test_cspf_summary_extra(self):
        model = _make_cspf_with_params()
        result = model.summary()
        assert "Keywords" in result
        assert "Residual topics" in result
        assert "Covariates (C)" in result

    def test_tbip_summary_extra(self):
        model = _make_tbip_with_params()
        result = model.summary()
        assert "Authors (N)" in result
        assert "Ideal-point range" in result

    def test_etm_summary_extra(self):
        model = _make_etm_with_params()
        result = model.summary()
        assert "Embedding dimension" in result


class TestTopicQualityMetrics:
    """Tests for compute_topic_coherence() and compute_topic_diversity()."""

    def test_coherence_npmi(self):
        model = _make_pf_with_params()
        df = model.compute_topic_coherence(metric="c_npmi", top_n=5)
        assert isinstance(df, pd.DataFrame)
        assert "topic" in df.columns
        assert "coherence" in df.columns
        assert len(df) == 3
        assert model.Metrics.coherence_scores is not None

    def test_coherence_umass(self):
        model = _make_pf_with_params()
        df = model.compute_topic_coherence(metric="u_mass", top_n=5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_coherence_with_texts(self):
        model = _make_pf_with_params()
        texts = [["word_0", "word_1", "word_2"]] * 20
        df = model.compute_topic_coherence(texts=texts, top_n=5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_diversity(self):
        model = _make_pf_with_params()
        score = model.compute_topic_diversity(top_n=10)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert model.Metrics.diversity == score

    def test_spf_coherence_and_diversity(self):
        model = _make_spf_with_params()
        df = model.compute_topic_coherence(top_n=5)
        assert len(df) == 3
        score = model.compute_topic_diversity(top_n=10)
        assert 0.0 <= score <= 1.0


class TestBasePlots:
    """Tests for NumpyroModel plot methods."""

    def test_plot_topic_prevalence(self):
        model = _make_pf_with_params()
        fig, ax = model.plot_topic_prevalence()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_topic_prevalence_save(self):
        model = _make_pf_with_params()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig, ax = model.plot_topic_prevalence(save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_topic_prevalence_not_trained(self):
        counts, vocab = _counts_vocab()
        model = PF(counts, vocab, num_topics=3, batch_size=4)
        with pytest.raises(ValueError, match="trained"):
            model.plot_topic_prevalence()

    def test_plot_topic_correlation(self):
        model = _make_pf_with_params()
        fig, ax = model.plot_topic_correlation()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_topic_correlation_save(self):
        model = _make_pf_with_params()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig, ax = model.plot_topic_correlation(save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_document_topic_heatmap(self):
        model = _make_pf_with_params()
        fig, ax = model.plot_document_topic_heatmap(n_docs=10)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_document_topic_heatmap_sorted(self):
        model = _make_pf_with_params()
        fig, ax = model.plot_document_topic_heatmap(n_docs=10, sort_by_topic=True)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_document_topic_heatmap_save(self):
        model = _make_pf_with_params()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig, ax = model.plot_document_topic_heatmap(n_docs=5, save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_wordclouds_save(self):
        model = _make_pf_with_params()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig, axes = model.plot_topic_wordclouds(save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestDenseCacheBranches:
    """Tests for _prepare_dense_cache branches."""

    def test_cache_false(self):
        model = _make_pf_with_params()
        model._prepare_dense_cache(cache_dense_counts=False, dense_cache_max_gb=1.0)
        assert model._dense_counts_cache is None

    def test_cache_true(self):
        model = _make_pf_with_params()
        with patch("jax.default_backend", return_value="cpu"):
            model._prepare_dense_cache(cache_dense_counts=True, dense_cache_max_gb=1.0)
            assert model._dense_counts_cache is not None

    def test_cache_auto_small(self):
        model = _make_pf_with_params()
        # Small matrix should auto-cache
        with patch("jax.default_backend", return_value="cpu"):
            model._prepare_dense_cache(cache_dense_counts=None, dense_cache_max_gb=10.0)
            assert model._dense_counts_cache is not None

    def test_cache_auto_too_large(self):
        model = _make_pf_with_params()
        # Pretend very small max to force no-cache
        with patch("jax.default_backend", return_value="cpu"):
            model._prepare_dense_cache(cache_dense_counts=None, dense_cache_max_gb=0.0)
            assert model._dense_counts_cache is None

    def test_cache_metal_backend(self):
        model = _make_pf_with_params()
        with patch("jax.default_backend", return_value="metal"):
            model._prepare_dense_cache(cache_dense_counts=True, dense_cache_max_gb=10.0)
            assert model._dense_counts_cache is None


# ============================================================================
# SPF-specific tests
# ============================================================================


class TestSPFCoverage:
    """Cover SPF-specific methods and validation."""

    def test_plot_seed_effectiveness(self):
        model = _make_spf_with_params()
        fig, axes = model.plot_seed_effectiveness()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_seed_effectiveness_not_trained(self):
        counts, vocab = _counts_vocab()
        keywords = _keywords()
        model = SPF(counts, vocab, keywords=keywords, residual_topics=1, batch_size=4)
        with pytest.raises(ValueError, match="trained"):
            model.plot_seed_effectiveness()

    def test_plot_seed_effectiveness_save(self):
        model = _make_spf_with_params()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig, axes = model.plot_seed_effectiveness(save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_recode_topics(self):
        model = _make_spf_with_params()
        topics, E_theta = model.return_topics()
        assert len(topics) == 20
        assert E_theta.shape == (20, 3)

    def test_spf_not_sparse_raises(self):
        dense = np.random.rand(20, 50).astype(np.float32)
        vocab = np.array([f"word_{i}" for i in range(50)])
        keywords = _keywords()
        with pytest.raises(TypeError, match="sparse"):
            SPF(dense, vocab, keywords=keywords, residual_topics=1, batch_size=4)

    def test_spf_empty_keywords_raises(self):
        counts, vocab = _counts_vocab()
        with pytest.raises(ValueError, match="empty"):
            SPF(counts, vocab, keywords={}, residual_topics=1, batch_size=4)

    def test_spf_empty_keyword_list_raises(self):
        counts, vocab = _counts_vocab()
        with pytest.raises(ValueError, match="non-empty list"):
            SPF(counts, vocab, keywords={"topic": []}, residual_topics=1, batch_size=4)


# ============================================================================
# CPF-specific tests
# ============================================================================


class TestCPFCoverage:
    """Cover CPF covariate methods and validation."""

    def test_return_covariate_effects(self):
        model = _make_cpf_with_params()
        df = model.return_covariate_effects()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)  # C=2, K=3

    def test_return_covariate_effects_not_trained(self):
        counts, vocab = _counts_vocab()
        covs = np.random.randn(20, 2).astype(np.float32)
        model = CPF(counts, vocab, num_topics=3, batch_size=4, X_design_matrix=covs)
        with pytest.raises(ValueError, match="trained"):
            model.return_covariate_effects()

    def test_return_covariate_effects_ci(self):
        model = _make_cpf_with_params()
        df = model.return_covariate_effects_ci(ci=0.90)
        assert isinstance(df, pd.DataFrame)
        assert "mean" in df.columns
        assert "lower" in df.columns
        assert "upper" in df.columns
        n_rows = 2 * 3  # C=2, K=3
        assert len(df) == n_rows

    def test_return_covariate_effects_ci_not_trained(self):
        counts, vocab = _counts_vocab()
        covs = np.random.randn(20, 2).astype(np.float32)
        model = CPF(counts, vocab, num_topics=3, batch_size=4, X_design_matrix=covs)
        with pytest.raises(ValueError, match="trained"):
            model.return_covariate_effects_ci()

    def test_plot_cov_effects(self):
        model = _make_cpf_with_params()
        fig, axes = model.plot_cov_effects()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_cov_effects_subset(self):
        model = _make_cpf_with_params()
        fig, axes = model.plot_cov_effects(topics=["topic_1"])
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_cov_effects_invalid_topic(self):
        model = _make_cpf_with_params()
        with pytest.raises(ValueError, match="None of"):
            model.plot_cov_effects(topics=["nonexistent"])

    def test_plot_cov_effects_not_trained(self):
        counts, vocab = _counts_vocab()
        covs = np.random.randn(20, 2).astype(np.float32)
        model = CPF(counts, vocab, num_topics=3, batch_size=4, X_design_matrix=covs)
        with pytest.raises(ValueError, match="trained"):
            model.plot_cov_effects()

    def test_plot_cov_effects_save(self):
        model = _make_cpf_with_params()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig, axes = model.plot_cov_effects(save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_cpf_not_sparse_raises(self):
        dense = np.random.rand(20, 50).astype(np.float32)
        vocab = np.array([f"word_{i}" for i in range(50)])
        with pytest.raises(TypeError, match="sparse"):
            CPF(dense, vocab, num_topics=3, batch_size=4)


# ============================================================================
# CSPF-specific tests
# ============================================================================


class TestCSPFCoverage:
    """Cover CSPF return, CI, and plot methods."""

    def test_return_topics(self):
        model = _make_cspf_with_params()
        topics, E_theta = model.return_topics()
        assert len(topics) == 20
        assert E_theta.shape == (20, 3)

    def test_return_beta(self):
        model = _make_cspf_with_params()
        beta = model.return_beta()
        assert isinstance(beta, pd.DataFrame)
        assert beta.shape[0] == 50
        assert beta.shape[1] == 3

    def test_return_covariate_effects(self):
        model = _make_cspf_with_params()
        df = model.return_covariate_effects()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)

    def test_return_covariate_effects_ci(self):
        model = _make_cspf_with_params()
        df = model.return_covariate_effects_ci(ci=0.95)
        assert isinstance(df, pd.DataFrame)
        assert "mean" in df.columns
        assert "lower" in df.columns
        assert "upper" in df.columns

    def test_return_covariate_effects_ci_not_trained(self):
        counts, vocab = _counts_vocab()
        keywords = _keywords()
        covs = np.random.randn(20, 2).astype(np.float32)
        model = CSPF(
            counts, vocab, keywords=keywords, residual_topics=1, batch_size=4, X_design_matrix=covs
        )
        with pytest.raises(ValueError, match="trained"):
            model.return_covariate_effects_ci()

    def test_plot_cov_effects_lambda_only(self):
        model = _make_cspf_with_params()
        results = model.plot_cov_effects()
        assert "lambda" in results
        assert results["lambda"][0] is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_cov_effects_with_shrinkage(self):
        model = _make_cspf_with_params()
        results = model.plot_cov_effects(include_shrinkage=True)
        assert "lambda" in results
        assert "lambda_intercept" in results
        assert "tau2" in results
        assert "delta2" in results
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_cov_effects_subset(self):
        model = _make_cspf_with_params()
        results = model.plot_cov_effects(topics=["topic_a"])
        assert "lambda" in results
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_cov_effects_invalid_topic(self):
        model = _make_cspf_with_params()
        with pytest.raises(ValueError, match="None of"):
            model.plot_cov_effects(topics=["nonexistent"])

    def test_plot_cov_effects_not_trained(self):
        counts, vocab = _counts_vocab()
        keywords = _keywords()
        covs = np.random.randn(20, 2).astype(np.float32)
        model = CSPF(
            counts, vocab, keywords=keywords, residual_topics=1, batch_size=4, X_design_matrix=covs
        )
        with pytest.raises(RuntimeError, match="No estimated parameters"):
            model.plot_cov_effects()

    def test_plot_cov_effects_save_dir(self):
        model = _make_cspf_with_params()
        with tempfile.TemporaryDirectory() as tmpdir:
            model.plot_cov_effects(save_path=tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "forest_lambda.png"))
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_cov_effects_save_dir_with_shrinkage(self):
        model = _make_cspf_with_params()
        with tempfile.TemporaryDirectory() as tmpdir:
            model.plot_cov_effects(include_shrinkage=True, save_path=tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "forest_lambda.png"))
            assert os.path.exists(os.path.join(tmpdir, "forest_lambda_intercept.png"))
            assert os.path.exists(os.path.join(tmpdir, "forest_tau2.png"))
            assert os.path.exists(os.path.join(tmpdir, "forest_delta2.png"))
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_gamma_ci_static(self):
        shape = np.array([2.0, 3.0])
        rate = np.array([1.0, 1.0])
        mean, lo, hi = CSPF._gamma_ci(shape, rate, ci=0.95)
        assert mean.shape == (2,)
        assert np.all(lo < mean)
        assert np.all(hi > mean)

    def test_topic_names(self):
        model = _make_cspf_with_params()
        names = model._topic_names()
        assert "topic_a" in names
        assert "topic_b" in names
        assert "residual_topic_1" in names

    def test_group_names(self):
        model = _make_cspf_with_params()
        groups = model._group_names()
        assert len(groups) == 2  # cov_0, cov_1


class TestCSPFValidation:
    """Cover CSPF __init__ validation branches."""

    def test_not_sparse_raises(self):
        dense = np.random.rand(20, 50).astype(np.float32)
        vocab = np.array([f"word_{i}" for i in range(50)])
        kw = _keywords()
        with pytest.raises(TypeError, match="sparse"):
            CSPF(dense, vocab, keywords=kw, residual_topics=1, batch_size=4)

    def test_empty_counts_raises(self):
        counts = sparse.csr_matrix((0, 50), dtype=np.float32)
        vocab = np.array([f"word_{i}" for i in range(50)])
        kw = _keywords()
        with pytest.raises(ValueError, match="empty"):
            CSPF(counts, vocab, keywords=kw, residual_topics=1, batch_size=4)

    def test_vocab_mismatch_raises(self):
        counts, _ = _counts_vocab()
        vocab = np.array(["a", "b", "c"])
        kw = {"t": ["a"]}
        with pytest.raises(ValueError, match="vocab size"):
            CSPF(counts, vocab, keywords=kw, residual_topics=1, batch_size=4)

    def test_negative_residual_raises(self):
        counts, vocab = _counts_vocab()
        kw = _keywords()
        with pytest.raises(ValueError, match="residual_topics"):
            CSPF(counts, vocab, keywords=kw, residual_topics=-1, batch_size=4)

    def test_invalid_batch_size_raises(self):
        counts, vocab = _counts_vocab()
        kw = _keywords()
        with pytest.raises(ValueError, match="batch_size"):
            CSPF(counts, vocab, keywords=kw, residual_topics=1, batch_size=0)

    def test_1d_covariates_raises(self):
        counts, vocab = _counts_vocab()
        kw = _keywords()
        covs_1d = np.random.randn(20).astype(np.float32)
        with pytest.raises(ValueError, match="2D"):
            CSPF(
                counts, vocab, keywords=kw, residual_topics=1, batch_size=4, X_design_matrix=covs_1d
            )

    def test_zero_column_covariates_raises(self):
        counts, vocab = _counts_vocab()
        kw = _keywords()
        covs_0col = np.random.randn(20, 0).astype(np.float32)
        with pytest.raises(ValueError, match="empty.*0 columns"):
            CSPF(
                counts,
                vocab,
                keywords=kw,
                residual_topics=1,
                batch_size=4,
                X_design_matrix=covs_0col,
            )

    def test_dataframe_covariates(self):
        counts, vocab = _counts_vocab()
        kw = _keywords()
        df_cov = pd.DataFrame(
            {
                "cov_alpha": np.random.randn(20),
                "cov_beta": np.random.randn(20),
            }
        )
        model = CSPF(
            counts, vocab, keywords=kw, residual_topics=1, batch_size=4, X_design_matrix=df_cov
        )
        assert "cov_alpha" in model.covariates
        assert "cov_beta" in model.covariates


class TestCSPFGroupIndexSeparators:
    """Cover _build_group_index with different separator styles."""

    def test_double_colon_separator(self):
        names = ["grp1::a", "grp1::b", "grp2::c"]
        idx = CSPF._build_group_index(names)
        assert idx[0] == idx[1]  # grp1
        assert idx[2] != idx[0]  # grp2

    def test_equals_separator(self):
        names = ["color=red", "color=blue", "shape=circle"]
        idx = CSPF._build_group_index(names)
        assert idx[0] == idx[1]
        assert idx[2] != idx[0]

    def test_bracket_separator(self):
        names = ["age[young]", "age[old]", "region[north]"]
        idx = CSPF._build_group_index(names)
        assert idx[0] == idx[1]
        assert idx[2] != idx[0]

    def test_no_separator(self):
        names = ["age", "income", "education"]
        idx = CSPF._build_group_index(names)
        assert len(set(idx.tolist())) == 3  # all different groups


# ============================================================================
# TBIP-specific tests
# ============================================================================


class TestTBIPCoverage:
    """Cover TBIP return and plot methods."""

    def test_return_topics(self):
        model = _make_tbip_with_params()
        cats, E_theta = model.return_topics()
        assert cats.shape == (20,)
        assert E_theta.shape == (20, 2)

    def test_return_topics_not_trained(self):
        counts, vocab = _counts_vocab()
        authors = np.array([f"author_{i % 5}" for i in range(20)])
        model = TBIP(counts, vocab, num_topics=2, authors=authors, batch_size=4)
        with pytest.raises(ValueError, match="trained"):
            model.return_topics()

    def test_return_beta(self):
        model = _make_tbip_with_params()
        beta = model.return_beta()
        assert isinstance(beta, pd.DataFrame)
        assert beta.shape == (50, 2)

    def test_return_beta_not_trained(self):
        counts, vocab = _counts_vocab()
        authors = np.array([f"author_{i % 5}" for i in range(20)])
        model = TBIP(counts, vocab, num_topics=2, authors=authors, batch_size=4)
        with pytest.raises(ValueError, match="trained"):
            model.return_beta()

    def test_return_ideal_points(self):
        model = _make_tbip_with_params()
        df = model.return_ideal_points()
        assert isinstance(df, pd.DataFrame)
        assert "author" in df.columns
        assert "ideal_point" in df.columns
        assert "std" in df.columns
        assert len(df) == len(model.authors_unique)

    def test_return_ideal_points_not_trained(self):
        counts, vocab = _counts_vocab()
        authors = np.array([f"author_{i % 5}" for i in range(20)])
        model = TBIP(counts, vocab, num_topics=2, authors=authors, batch_size=4)
        with pytest.raises(ValueError, match="trained"):
            model.return_ideal_points()

    def test_return_ideological_words(self):
        model = _make_tbip_with_params()
        df = model.return_ideological_words(topic=0, n=5)
        assert isinstance(df, pd.DataFrame)
        assert "word" in df.columns
        assert "eta" in df.columns
        assert "direction" in df.columns
        assert len(df) == 10  # 5 positive + 5 negative

    def test_return_ideological_words_not_trained(self):
        counts, vocab = _counts_vocab()
        authors = np.array([f"author_{i % 5}" for i in range(20)])
        model = TBIP(counts, vocab, num_topics=2, authors=authors, batch_size=4)
        with pytest.raises(ValueError, match="trained"):
            model.return_ideological_words(topic=0)

    def test_return_ideological_words_invalid_topic(self):
        model = _make_tbip_with_params()
        with pytest.raises(ValueError, match="topic must be"):
            model.return_ideological_words(topic=99)

    def test_plot_ideal_points_show_ci(self):
        model = _make_tbip_with_params()
        fig, ax = model.plot_ideal_points(show_ci=True)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_ideal_points_no_ci(self):
        model = _make_tbip_with_params()
        fig, ax = model.plot_ideal_points(show_ci=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_ideal_points_save(self):
        model = _make_tbip_with_params()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig, ax = model.plot_ideal_points(save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestTBIPValidation:
    """Cover TBIP __init__ validation branches."""

    def test_not_sparse_raises(self):
        dense = np.random.rand(20, 50).astype(np.float32)
        vocab = np.array([f"word_{i}" for i in range(50)])
        authors = np.array([f"a_{i}" for i in range(20)])
        with pytest.raises(TypeError, match="sparse"):
            TBIP(dense, vocab, num_topics=2, authors=authors, batch_size=4)

    def test_empty_counts_raises(self):
        counts = sparse.csr_matrix((0, 50), dtype=np.float32)
        vocab = np.array([f"word_{i}" for i in range(50)])
        authors = np.array([])
        with pytest.raises(ValueError, match="empty"):
            TBIP(counts, vocab, num_topics=2, authors=authors, batch_size=4)

    def test_zero_topics_raises(self):
        counts, vocab = _counts_vocab()
        authors = np.array([f"a_{i}" for i in range(20)])
        with pytest.raises(ValueError, match="num_topics"):
            TBIP(counts, vocab, num_topics=0, authors=authors, batch_size=4)

    def test_invalid_batch_raises(self):
        counts, vocab = _counts_vocab()
        authors = np.array([f"a_{i}" for i in range(20)])
        with pytest.raises(ValueError, match="batch_size"):
            TBIP(counts, vocab, num_topics=2, authors=authors, batch_size=0)

    def test_vocab_mismatch_raises(self):
        counts, _ = _counts_vocab()
        vocab = np.array(["a", "b"])
        authors = np.array([f"a_{i}" for i in range(20)])
        with pytest.raises(ValueError, match="vocab size"):
            TBIP(counts, vocab, num_topics=2, authors=authors, batch_size=4)

    def test_authors_length_mismatch_raises(self):
        counts, vocab = _counts_vocab()
        authors = np.array(["a", "b"])  # Too short
        with pytest.raises(ValueError, match="authors"):
            TBIP(counts, vocab, num_topics=2, authors=authors, batch_size=4)

    def test_beta_init_not_ndarray_raises(self):
        counts, vocab = _counts_vocab()
        authors = np.array([f"a_{i}" for i in range(20)])
        with pytest.raises(ValueError, match="numpy"):
            TBIP(
                counts,
                vocab,
                num_topics=2,
                authors=authors,
                batch_size=4,
                time_varying=True,
                beta_shape_init=[[1, 2]],
                beta_rate_init=np.ones((2, 50)),
            )

    def test_time_varying_no_init_warns(self):
        counts, vocab = _counts_vocab()
        authors = np.array([f"a_{i}" for i in range(20)])
        with pytest.warns(UserWarning, match="No initial values"):
            TBIP(counts, vocab, num_topics=2, authors=authors, batch_size=4, time_varying=True)


# ============================================================================
# ETM-specific tests
# ============================================================================


class TestETMCoverage:
    """Cover ETM return methods and validation."""

    def test_return_topics(self):
        model = _make_etm_with_params()
        cats, E_theta = model.return_topics()
        assert cats.shape == (20,)
        assert E_theta.shape == (20, 2)

    def test_return_beta(self):
        model = _make_etm_with_params()
        beta = model.return_beta()
        assert isinstance(beta, pd.DataFrame)
        assert beta.shape == (50, 2)

    def test_not_sparse_raises(self):
        dense = np.random.rand(20, 50).astype(np.float32)
        vocab = np.array([f"word_{i}" for i in range(50)])
        emb = {f"word_{i}": np.random.randn(4) for i in range(50)}
        with pytest.raises(TypeError, match="sparse"):
            ETM(dense, vocab, num_topics=2, batch_size=4, embeddings_mapping=emb, embed_size=4)

    def test_empty_counts_raises(self):
        counts = sparse.csr_matrix((0, 50), dtype=np.float32)
        vocab = np.array([f"word_{i}" for i in range(50)])
        emb = {f"word_{i}": np.random.randn(4) for i in range(50)}
        with pytest.raises(ValueError, match="empty"):
            ETM(counts, vocab, num_topics=2, batch_size=4, embeddings_mapping=emb, embed_size=4)

    def test_zero_topics_raises(self):
        counts, vocab = _counts_vocab()
        emb = {f"word_{i}": np.random.randn(4) for i in range(50)}
        with pytest.raises(ValueError, match="num_topics"):
            ETM(counts, vocab, num_topics=0, batch_size=4, embeddings_mapping=emb, embed_size=4)

    def test_invalid_batch_raises(self):
        counts, vocab = _counts_vocab()
        emb = {f"word_{i}": np.random.randn(4) for i in range(50)}
        with pytest.raises(ValueError, match="batch_size"):
            ETM(counts, vocab, num_topics=2, batch_size=0, embeddings_mapping=emb, embed_size=4)

    def test_zero_embed_size_raises(self):
        counts, vocab = _counts_vocab()
        emb = {f"word_{i}": np.random.randn(4) for i in range(50)}
        with pytest.raises(ValueError, match="embed_size"):
            ETM(counts, vocab, num_topics=2, batch_size=4, embeddings_mapping=emb, embed_size=0)


# ============================================================================
# Utils coverage
# ============================================================================


class TestUtilsCoverage:
    """Cover create_word2vec_embedding_from_dataset debug_mode branches."""

    def test_word2vec_debug_mode(self):
        from poisson_topicmodels.utils.utils import (
            create_word2vec_embedding_from_dataset,
        )

        sentences = ["hello world foo bar", "another sentence here", "more words to process"]
        with tempfile.NamedTemporaryFile(suffix=".kv", delete=False) as f:
            tmp_path = f.name

        try:
            embeddings = create_word2vec_embedding_from_dataset(
                sentences,
                dim_rho=10,
                iters=1,
                min_count=1,
                debug_mode=True,
                embedding_file_path=tmp_path,
            )
            assert embeddings is not None
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_word2vec_c_format(self):
        from poisson_topicmodels.utils.utils import (
            create_word2vec_embedding_from_dataset,
        )

        sentences = ["hello world foo bar", "another sentence here"]
        with tempfile.NamedTemporaryFile(suffix=".kv", delete=False) as f:
            tmp_path = f.name

        try:
            create_word2vec_embedding_from_dataset(
                sentences,
                dim_rho=10,
                iters=1,
                min_count=1,
                debug_mode=True,
                save_c_format_w2vec=True,
                embedding_file_path=tmp_path,
            )
            assert os.path.exists(f"{tmp_path}.bin")
            assert os.path.exists(f"{tmp_path}.txt")
        finally:
            for ext in ["", ".bin", ".txt"]:
                p = f"{tmp_path}{ext}"
                if os.path.exists(p):
                    os.unlink(p)


# ============================================================================
# Cross-model coherence/diversity via SPF and TBIP
# ============================================================================


class TestCrossModelMetrics:
    """Verify coherence/diversity works on non-PF models too."""

    def test_tbip_coherence(self):
        model = _make_tbip_with_params()
        df = model.compute_topic_coherence(top_n=5)
        assert len(df) == 2

    def test_tbip_diversity(self):
        model = _make_tbip_with_params()
        score = model.compute_topic_diversity(top_n=10)
        assert 0.0 <= score <= 1.0

    def test_cpf_coherence(self):
        model = _make_cpf_with_params()
        df = model.compute_topic_coherence(top_n=5)
        assert len(df) == 3

    def test_cspf_coherence(self):
        model = _make_cspf_with_params()
        df = model.compute_topic_coherence(top_n=5)
        assert len(df) == 3

    def test_etm_coherence(self):
        model = _make_etm_with_params()
        df = model.compute_topic_coherence(top_n=5)
        assert len(df) == 2


# ============================================================================
# Cross-model plot tests
# ============================================================================


class TestCrossModelPlots:
    """Verify base-class plots work across different model types."""

    def test_spf_topic_prevalence(self):
        model = _make_spf_with_params()
        fig, ax = model.plot_topic_prevalence()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_tbip_topic_correlation(self):
        model = _make_tbip_with_params()
        fig, ax = model.plot_topic_correlation()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_cspf_document_topic_heatmap(self):
        model = _make_cspf_with_params()
        fig, ax = model.plot_document_topic_heatmap(n_docs=5)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_etm_topic_prevalence(self):
        model = _make_etm_with_params()
        fig, ax = model.plot_topic_prevalence()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)
