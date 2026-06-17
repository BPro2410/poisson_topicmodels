"""Focused STBS tests using small synthetic data.

The tests avoid expensive training by tracing model/guide code directly and by
injecting deterministic fitted parameters for post-fit inspection methods.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sparse
from jax import random
from numpyro import handlers

from poisson_topicmodels import STBS

matplotlib.use("Agg")


def _stbs_inputs():
    counts = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 2, 0, 1],
                [0, 1, 0, 2, 1],
                [2, 1, 0, 0, 1],
                [0, 2, 1, 1, 0],
            ],
            dtype=np.float32,
        )
    )
    vocab = np.array(["alpha", "beta", "gamma", "delta", "epsilon"])
    authors = np.array(["alice", "bob", "alice", "carol"])
    covariates = pd.DataFrame(
        {
            "party": [-1.0, 1.0, 0.0],
            "seniority": [0.2, 0.8, 0.5],
        },
        index=["alice", "bob", "carol"],
    )
    return counts, vocab, authors, covariates


def _stbs_model():
    counts, vocab, authors, covariates = _stbs_inputs()
    return STBS(
        counts=counts,
        vocab=vocab,
        num_topics=2,
        authors=authors,
        batch_size=2,
        X_design_matrix=covariates,
    )


def _fit_stbs_with_params():
    model = _stbs_model()
    model.estimated_params = {
        "theta_shape": np.array([[2.0, 1.0], [1.0, 3.0], [3.0, 1.0], [1.5, 2.5]], dtype=np.float32),
        "theta_rate": np.ones((4, 2), dtype=np.float32),
        "beta_shape": np.array(
            [[3.0, 1.0, 2.0, 0.5, 1.5], [1.0, 3.0, 0.5, 2.0, 1.5]], dtype=np.float32
        ),
        "beta_rate": np.ones((2, 5), dtype=np.float32),
        "mu_eta": np.array(
            [[0.4, -0.2, 0.1, -0.3, 0.2], [-0.1, 0.3, -0.4, 0.2, 0.1]],
            dtype=np.float32,
        ),
        "sigma_eta": np.ones((2, 5), dtype=np.float32) * 0.2,
        "mu_i": np.array([[-0.7, 0.1], [0.8, -0.2], [0.0, 0.6]], dtype=np.float32),
        "sigma_i": np.ones((3, 2), dtype=np.float32) * 0.15,
        "mu_iota": np.array([[-0.5, 0.3], [0.2, -0.1]], dtype=np.float32),
        "sigma_iota": np.ones((2, 2), dtype=np.float32) * 0.1,
    }
    model.Metrics.loss = [12.0, 8.0, 6.0]
    return model


def test_stbs_validation_and_train_step_guards():
    """Cover STBS initialization validation and cheap train_step guards."""
    counts, vocab, authors, covariates = _stbs_inputs()

    with pytest.raises(TypeError, match="counts must be a scipy sparse matrix"):
        STBS(counts.toarray(), vocab, 2, authors, batch_size=2, X_design_matrix=covariates)

    with pytest.raises(ValueError, match="counts matrix is empty"):
        STBS(sparse.csr_matrix((0, 5)), vocab, 2, authors[:0], batch_size=1)

    with pytest.raises(ValueError, match="num_topics must be > 0"):
        STBS(counts, vocab, 0, authors, batch_size=2, X_design_matrix=covariates)

    with pytest.raises(ValueError, match="batch_size must satisfy"):
        STBS(counts, vocab, 2, authors, batch_size=10, X_design_matrix=covariates)

    with pytest.raises(ValueError, match="vocab size"):
        STBS(counts, vocab[:-1], 2, authors, batch_size=2, X_design_matrix=covariates)

    with pytest.raises(ValueError, match="authors length"):
        STBS(counts, vocab, 2, authors[:-1], batch_size=2, X_design_matrix=covariates)

    with pytest.raises(ValueError, match="covariates must be 2D"):
        STBS(counts, vocab, 2, authors, batch_size=2, X_design_matrix=np.ones(3))

    with pytest.raises(ValueError, match="covariates has 2 rows"):
        STBS(counts, vocab, 2, authors, batch_size=2, X_design_matrix=np.ones((2, 2)))

    with pytest.raises(ValueError, match="covariates matrix is empty"):
        STBS(counts, vocab, 2, authors, batch_size=2, X_design_matrix=np.ones((3, 0)))

    model = _stbs_model()
    with pytest.raises(ValueError, match="num_steps must be > 0"):
        model.train_step(num_steps=0, lr=0.1)
    with pytest.raises(ValueError, match="lr must be > 0"):
        model.train_step(num_steps=1, lr=0.0)


def test_stbs_batch_model_and_guide_trace():
    """Trace STBS probabilistic programs without running SVI."""
    model = _stbs_model()
    y_batch, d_batch, i_batch = model._get_batch(random.PRNGKey(0), model.counts)

    assert y_batch.shape == (2, model.V)
    assert d_batch.shape == (2,)
    assert i_batch.shape == (2,)

    model_trace = handlers.trace(handlers.seed(model._model, random.PRNGKey(1))).get_trace(
        y_batch, d_batch, i_batch
    )
    assert {"beta", "eta", "iota", "i", "theta", "Y_batch"}.issubset(model_trace)

    guide_trace = handlers.trace(handlers.seed(model._guide, random.PRNGKey(2))).get_trace(
        y_batch, d_batch, i_batch
    )
    assert {"beta", "eta", "iota", "i", "theta"}.issubset(guide_trace)

    registered = model.input_params()
    assert "a_beta" in registered["hyperparameters"]
    assert "beta_shape" in registered["initialized_variables"]


def test_stbs_post_fit_methods_and_plots(tmp_path):
    """Cover STBS extraction and plotting methods with deterministic parameters."""
    model = _fit_stbs_with_params()

    ideal_points = model.return_ideal_points()
    assert list(ideal_points.columns) == ["author", "topic", "ideal_point", "std"]
    assert ideal_points.shape == (model.N * model.K, 4)

    ideal_covariates = model.return_ideal_covariates()
    assert list(ideal_covariates.columns) == ["covariate", "topic", "iota", "std"]
    assert ideal_covariates.shape == (model.L * model.K, 4)

    summary = model.summary(n_top_words=2)
    assert "Authors (N):" in summary
    assert "Iota range" in summary

    with pytest.raises(ValueError, match="topics list is empty"):
        model.plot_topic_wordclouds(topics=[])
    with pytest.raises(ValueError, match="topics must be indices"):
        model.plot_topic_wordclouds(topics=[model.K])

    model.plot_topic_wordclouds(n_words=2, ideology_values=None)
    model.plot_topic_wordclouds(n_words=2, ideology_values=(-1, 1), topics=[0], log_corrected=False)
    model.plot_topic_prevalence(topic_labels={0: "Economy"}, selected_topics=[0, 1], sort=False)
    model.plot_author_topic_heatmap(
        topic_labels={0: "Economy", 1: "Health"},
        author_labels={0: "Alice", 1: "Bob", 2: "Carol"},
        selected_topics=[0],
    )
    model.plot_ideol_points(group=False)
    model.plot_ideol_points(
        group_var=np.array([-1, 1, 1]),
        group_labels={-1: "left", 1: "right"},
        topic_labels={0: "Economy", 1: "Health"},
    )

    with pytest.raises(ValueError, match="No group_var provided"):
        model.plot_ideol_points(group=True)
    with pytest.raises(ValueError, match="group_var must have length"):
        model.plot_ideol_points(group_var=np.array([1, 2]))

    model.plot_iota_credible_intervals(
        selected_topics=[0],
        selected_covariates=["party"],
        topic_labels={0: "Economy"},
        covariate_labels={0: "Party"},
        save_path=str(tmp_path / "iota.png"),
    )
    assert (tmp_path / "iota.png").exists()

    constant_model = _fit_stbs_with_params()
    constant_model._constantparams.update(
        {
            "theta": constant_model.estimated_params["theta_shape"],
            "beta": constant_model.estimated_params["beta_shape"],
            "eta": constant_model.estimated_params["mu_eta"],
            "i": constant_model.estimated_params["mu_i"],
            "iota": constant_model.estimated_params["mu_iota"],
        }
    )
    with pytest.warns(UserWarning, match="beta is constant"):
        constant_model.plot_topic_wordclouds(n_words=2, ideology_values=(0,), log_corrected=True)
    assert constant_model.return_ideal_points()["std"].eq(0).all()
    assert constant_model.return_ideal_covariates()["std"].eq(0).all()
    constant_model.plot_topic_prevalence()
    constant_model.plot_author_topic_heatmap()
    constant_model.plot_iota_credible_intervals(selected_covariates=[0], selected_topics=[0])

    unfit_model = _stbs_model()
    with pytest.raises(ValueError, match="return_ideal_points"):
        unfit_model.return_ideal_points()
    with pytest.raises(ValueError, match="return_ideal_covariates"):
        unfit_model.return_ideal_covariates()
    with pytest.raises(ValueError, match="plot_topic_prevalence"):
        unfit_model.plot_topic_prevalence()
    with pytest.raises(ValueError, match="plot_author_topic_heatmap"):
        unfit_model.plot_author_topic_heatmap()
    with pytest.raises(ValueError, match="plot_ideology_points"):
        unfit_model.plot_ideol_points()
    with pytest.raises(ValueError, match="plot_iota_credible_intervals"):
        unfit_model.plot_iota_credible_intervals()

    plt.close("all")
