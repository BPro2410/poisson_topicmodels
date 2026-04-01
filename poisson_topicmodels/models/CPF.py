from typing import Dict, List, Optional, Tuple

import jax.nn as jnn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import scipy.sparse as sparse
from scipy import stats as sp_stats
from numpyro import param, plate, sample
from numpyro.distributions import constraints

# Abstract class - defining the minimum requirements for the probabilistic model
from .numpyro_model import NumpyroModel


# Create numpyro model
class CPF(NumpyroModel):
    """
    Covariate Poisson Factorization (CPF) topic model.

    Topic model that incorporates document-level covariates to capture how topics
    vary with external variables (e.g., author attributes, temporal features).

    Parameters
    ----------
    counts : scipy.sparse.csr_matrix
        Document-term matrix of shape (D, V) with word counts.
    vocab : np.ndarray
        Vocabulary array of shape (V,) containing word terms.
    covariates : np.ndarray or pd.DataFrame
        Document-level covariates of shape (D, C) where C is number of features.
    num_topics : int
        Number of topics K. Must be > 0.
    batch_size : int
        Mini-batch size for stochastic variational inference.
        Must satisfy 0 < batch_size <= D.

    Attributes
    ----------
    D : int
        Number of documents.
    V : int
        Vocabulary size.
    K : int
        Number of topics.
    C : int
        Number of covariate features.
    counts : scipy.sparse.csr_matrix
        Document-term matrix.
    vocab : np.ndarray
        Vocabulary array.
    X_design_matrix : jnp.ndarray
        Design matrix of covariates.

    Examples
    --------
    >>> from scipy.sparse import random
    >>> import numpy as np
    >>> from topicmodels import CPF
    >>> counts = random(100, 500, density=0.01, format='csr')
    >>> vocab = np.array([f'word_{i}' for i in range(500)])
    >>> covariates = np.random.randn(100, 3)  # 3 covariate features
    >>> model = CPF(counts, vocab, covariates, num_topics=10, batch_size=32)
    >>> params = model.train_step(num_steps=100, lr=0.01, random_seed=42)
    """

    def __init__(
        self,
        counts: sparse.csr_matrix,
        vocab: np.ndarray,
        num_topics: int,
        batch_size: int,
        X_design_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the CPF model with input validation.

        Parameters
        ----------
        counts : scipy.sparse.csr_matrix
            Document-term matrix.
        vocab : np.ndarray
            Vocabulary array.
        num_topics : int
            Number of topics.
        batch_size : int
            Mini-batch size.
        X_design_matrix : np.ndarray or pd.DataFrame, optional
            Document-level covariates.

        Raises
        ------
        TypeError
            If counts is not sparse or covariates have wrong type.
        ValueError
            If dimensions are invalid or inconsistent.
        """
        super().__init__()

        # Input validation
        if not sparse.issparse(counts):
            raise TypeError(f"counts must be a scipy sparse matrix, got {type(counts).__name__}")

        D, V = counts.shape
        if D == 0 or V == 0:
            raise ValueError(f"counts matrix is empty: shape ({D}, {V})")

        if vocab.shape[0] != V:
            raise ValueError(f"vocab size {vocab.shape[0]} != counts columns {V}")

        if num_topics <= 0:
            raise ValueError(f"num_topics must be > 0, got {num_topics}")

        if batch_size <= 0 or batch_size > D:
            raise ValueError(f"batch_size must satisfy 0 < batch_size <= {D}, got {batch_size}")

        if X_design_matrix is not None:
            _have_names = False
            if isinstance(X_design_matrix, pd.DataFrame):
                covariate_names = [str(col) for col in X_design_matrix.columns]
                _have_names = True
                X_design_matrix = X_design_matrix.values

            X_design_matrix = np.asarray(X_design_matrix)
            if X_design_matrix.ndim != 2:
                raise ValueError(f"covariates must be 2D, got shape {X_design_matrix.shape}")
            if X_design_matrix.shape[0] != D:
                raise ValueError(f"covariates has {X_design_matrix.shape[0]} rows, expected {D}")
            if X_design_matrix.shape[1] == 0:
                raise ValueError("covariates matrix is empty (0 columns)")

            if not _have_names:
                covariate_names = [f"cov_{i}" for i in range(X_design_matrix.shape[1])]
        else:
            covariate_names = ["intercept"]

        # Store validated inputs
        self.counts = counts
        self.D = D
        self.V = V
        self.vocab = vocab
        self.K = num_topics
        self.batch_size = batch_size
        self.covariates: List[str] = covariate_names
        self.X_design_matrix = (
            jnp.array(X_design_matrix) if X_design_matrix is not None else jnp.ones((D, 1))
        )
        self.C = self.X_design_matrix.shape[1]

    # -- Model --
    def _model(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray) -> None:
        """
        Define the probabilistic generative model using NumPyro.

        Model structure:
        - Beta (K x V): topic-word distributions
        - Lambda (C x K): covariate effects on topics
        - Theta (D x K): document-topic distributions (covariate-dependent)
        - Y_batch (batch_size x V): observed word counts

        Parameters
        ----------
        Y_batch : jnp.ndarray
            Batch of observed word counts (batch_size, V).
        d_batch : jnp.ndarray
            Document indices in batch (batch_size,).
        """

        # Topic distributions
        with plate("k", size=self.K, dim=-2):
            with plate("k_v", size=self.V, dim=-1):
                beta = sample("beta", dist.Gamma(0.3, 0.3))

        # Covariate effects
        with plate("c", size=self.C, dim=-2):
            with plate("c_k", size=self.K, dim=-1):
                lambda_ = sample("phi", dist.Normal(0.0, 1.0))

        # Transform covariate effects via softplus
        a_theta_S = jnn.softplus(jnp.matmul(self.X_design_matrix, lambda_))[d_batch]

        # Document distribution
        with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
            with plate("d_k", size=self.K, dim=-1):
                theta = sample("theta", dist.Gamma(a_theta_S, 0.3))

            # Poisson rate
            P = jnp.matmul(theta, beta)

            with plate("d_v", size=self.V, dim=-1):
                sample("Y_batch", dist.Poisson(P), obs=Y_batch)

    # -- Guide, i.e. variational family --
    def _guide(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray) -> None:
        """
        Define the variational guide (approximate posterior).

        Uses Gamma family for topic distributions and Normal family for
        covariate effects.

        Parameters
        ----------
        Y_batch : jnp.ndarray
            Batch of observed word counts.
        d_batch : jnp.ndarray
            Document indices in batch.
        """

        # Define variational parameter
        a_beta = param(
            "beta_shape", init_value=jnp.ones([self.K, self.V]), constraint=constraints.positive
        )
        b_beta = param(
            "beta_rate",
            init_value=jnp.ones([self.K, self.V]) * self.D / 1000 * 2,
            constraint=constraints.positive,
        )

        a_theta = param(
            "theta_shape", init_value=jnp.ones([self.D, self.K]), constraint=constraints.positive
        )
        b_theta = param(
            "theta_rate",
            init_value=jnp.ones([self.D, self.K]) * self.D / 1000,
            constraint=constraints.positive,
        )

        location_lambda = param("lambda_location", init_value=jnp.zeros([self.C, self.K]))
        scale_lambda = param(
            "lambda_scale", init_value=jnp.ones([self.C, self.K]), constraint=constraints.positive
        )

        with plate("k", size=self.K, dim=-2):
            with plate("k_v", size=self.V, dim=-1):
                sample("beta", dist.Gamma(a_beta, b_beta))

        with plate("c", size=self.C, dim=-2):
            with plate("c_k", size=self.K, dim=-1):
                sample("phi", dist.Normal(location_lambda, scale_lambda))

        with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
            with plate("d_k", size=self.K, dim=-1):
                sample("theta", dist.Gamma(a_theta[d_batch], b_theta[d_batch]))

    def return_covariate_effects(self) -> pd.DataFrame:
        """Return point estimates of covariate effects (lambda).

        Returns
        -------
        pd.DataFrame
            DataFrame with covariates as rows and topics as columns.

        Raises
        ------
        ValueError
            If model has not been trained yet.
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling return_covariate_effects()")

        topic_names = [f"topic_{i + 1}" for i in range(self.K)]
        return pd.DataFrame(
            np.asarray(self.estimated_params["lambda_location"]),
            index=self.covariates,
            columns=topic_names,
        )

    def return_covariate_effects_ci(self, ci: float = 0.95) -> pd.DataFrame:
        """Return covariate effects with credible intervals.

        Uses the Normal variational posterior for lambda:
        ``mean = lambda_location``, ``CI = mean +/- z * lambda_scale``.

        Parameters
        ----------
        ci : float, optional
            Credible-interval level (default 0.95).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``['covariate', 'topic', 'mean',
            'lower', 'upper']``.

        Raises
        ------
        ValueError
            If model has not been trained yet.
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling return_covariate_effects_ci()")

        loc = np.asarray(self.estimated_params["lambda_location"])  # (C, K)
        scale = np.asarray(self.estimated_params["lambda_scale"])   # (C, K)
        z = sp_stats.norm.ppf(1.0 - (1.0 - ci) / 2.0)

        topic_names = [f"topic_{i + 1}" for i in range(self.K)]
        rows = []
        for c_idx, cov_name in enumerate(self.covariates):
            for k_idx, topic_name in enumerate(topic_names):
                rows.append({
                    "covariate": cov_name,
                    "topic": topic_name,
                    "mean": float(loc[c_idx, k_idx]),
                    "lower": float(loc[c_idx, k_idx] - z * scale[c_idx, k_idx]),
                    "upper": float(loc[c_idx, k_idx] + z * scale[c_idx, k_idx]),
                })
        return pd.DataFrame(rows)

    def plot_cov_effects(
        self,
        ci: float = 0.95,
        topics: Optional[List[str]] = None,
        figsize_per_topic: Tuple[float, float] = (5.0, 0.28),
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        r"""Forest plot of covariate effects (lambda) with credible intervals.

        Parameters
        ----------
        ci : float, optional
            Credible-interval level (default 0.95).
        topics : list of str, optional
            Subset of topic names to plot.  If ``None``, all topics are
            plotted.
        figsize_per_topic : tuple of float, optional
            ``(width, height_per_covariate)`` for panel sizing.
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        tuple of (plt.Figure, np.ndarray of Axes)
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling plot_cov_effects()")

        topic_names = [f"topic_{i + 1}" for i in range(self.K)]

        if topics is not None:
            topic_idx = [i for i, t in enumerate(topic_names) if t in topics]
            if not topic_idx:
                raise ValueError(f"None of {topics} found in {topic_names}")
            plot_topics = [topic_names[i] for i in topic_idx]
        else:
            plot_topics = topic_names
            topic_idx = list(range(self.K))

        loc = np.asarray(self.estimated_params["lambda_location"])  # (C, K)
        scale = np.asarray(self.estimated_params["lambda_scale"])   # (C, K)
        z = sp_stats.norm.ppf(1.0 - (1.0 - ci) / 2.0)

        n_topics = len(plot_topics)
        n_cov = loc.shape[0]

        with plt.rc_context(self._setup_academic_style()):
            fig_w = figsize_per_topic[0]
            fig_h = max(3.0, n_cov * figsize_per_topic[1])

            ncols = min(n_topics, 4)
            nrows = int(np.ceil(n_topics / ncols))
            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(fig_w * ncols, fig_h * nrows),
                sharey=True, squeeze=False,
            )
            axes_flat = axes.flatten()

            for panel_i, (ki, tname) in enumerate(zip(topic_idx, plot_topics)):
                ax = axes_flat[panel_i]
                means = loc[:, ki]
                lo = means - z * scale[:, ki]
                hi = means + z * scale[:, ki]
                y_pos = np.arange(n_cov)[::-1]

                ax.axvline(0, color="#999999", linewidth=0.5, zorder=0)
                for j in range(n_cov):
                    ax.plot([lo[j], hi[j]], [y_pos[j], y_pos[j]],
                            color="#4E79A7", linewidth=1.2, zorder=1)
                    ax.scatter(means[j], y_pos[j], color="#4E79A7",
                               s=18, zorder=2, edgecolors="white", linewidths=0.3)

                ax.set_yticks(y_pos)
                ax.set_yticklabels(list(self.covariates))
                ax.set_title(tname)
                ax.set_xlabel(r"$\lambda$")

            for idx in range(n_topics, len(axes_flat)):
                axes_flat[idx].set_visible(False)

            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, axes

    def _summary_extra(self) -> str:
        """CPF-specific summary information."""
        lines = [
            f"  Covariates (C):           {self.C}",
            f"  Covariate names:          {', '.join(self.covariates)}",
        ]
        return "\n".join(lines)
