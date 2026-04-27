import warnings
from typing import Optional, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import scipy.sparse as sparse
from jax import jit, random
from numpyro import param, plate, sample
from numpyro.distributions import constraints
from numpyro.infer import SVI, TraceMeanField_ELBO
from optax import adam
from tqdm import tqdm

# Abstract class - defining the minimum requirements for the probabilistic model
from .numpyro_model import NumpyroModel


class TBIP(NumpyroModel):
    """
    TBIP Model

    This class models topic-based ideal points (TBIP) in a set of documents authored by multiple individuals.
    """

    def __init__(
        self,
        counts: sparse.csr_matrix,
        vocab: np.ndarray,
        num_topics: int,
        authors: np.ndarray,
        batch_size: int,
        time_varying: bool = False,
        beta_shape_init: np.ndarray = None,
        beta_rate_init: np.ndarray = None,
    ) -> None:
        """
        Initialize the TBIP model.

        Parameters
        ----------
        counts : scipy.sparse.csr_matrix
            A 2D sparse array of shape (D, V) representing the word counts in each document,
            where D is the number of documents and V is the vocabulary size.
        vocab : np.ndarray
            A vocabulary array of shape (V,) containing word terms.
        num_topics : int
            The number of topics (K). Must be > 0.
        authors : np.ndarray or list
            An array of authors for each document.
        batch_size : int
            The number of documents to be processed in each batch.
            Must satisfy 0 < batch_size <= D.
        time_varying : bool, optional
            Whether to model time-varying ideal points (default is False).
        beta_shape_init : np.ndarray, optional
            Initial shape parameters for the topic-word distributions (default is None).
            Must have shape (K, V) if provided.
        beta_rate_init : np.ndarray, optional
            Initial rate parameters for the topic-word distributions (default is None).
            Must have shape (K, V) if provided.

        Raises
        ------
        TypeError
            If counts is not a sparse matrix.
        ValueError
            If dimensions are invalid or time_varying parameters have wrong shape.
        """

        super().__init__()

        # Input validation
        if not sparse.issparse(counts):
            raise TypeError(f"counts must be a scipy sparse matrix, got {type(counts).__name__}")

        D, V = counts.shape
        if D == 0 or V == 0:
            raise ValueError(f"counts matrix is empty: shape ({D}, {V})")

        if num_topics <= 0:
            raise ValueError(f"num_topics must be > 0, got {num_topics}")

        if batch_size <= 0 or batch_size > D:
            raise ValueError(f"batch_size must satisfy 0 < batch_size <= {D}, got {batch_size}")

        if vocab.shape[0] != V:
            raise ValueError(f"vocab size {vocab.shape[0]} != counts columns {V}")

        # Convert authors to array-like
        authors = np.asarray(authors)
        if len(authors) != D:
            raise ValueError(f"authors length {len(authors)} != counts rows {D}")

        self.authors_unique = np.unique(authors)
        self.author_map = {speaker: idx for idx, speaker in enumerate(self.authors_unique)}
        self.author_indices = np.array([self.author_map[a] for a in authors])
        self.N = len(self.authors_unique)  # number of people
        self.counts = counts
        self.D = D
        self.V = V
        self.K = num_topics
        self.batch_size = batch_size  # number of documents in a batch
        self.vocab = vocab

        # Add time varying component
        self.time_varying = time_varying
        if self.time_varying:
            warnings.warn("Time-varying TBIP model initiated.")
            warnings.warn(
                "Please notice: Setting time_varying=True requires to fit the TBIP model "
                "separately for each time period. Please initiate the TBIP model in t+1 with "
                "the estimated beta parameter in t. See documentation for more details."
            )

            # check if beta_rate_init and beta_shape_init have the correct shape and are jnp.arrays
            for inits in [beta_shape_init, beta_rate_init]:
                if inits is None:
                    warnings.warn(
                        "No initial values for beta parameters were provided. "
                        "The model will initialize them uniformly."
                    )
                if inits is not None:
                    if not isinstance(inits, (np.ndarray, jnp.ndarray)):
                        raise ValueError(
                            "beta_shape_init and beta_rate_init must be numpy or jnp.ndarray objects "
                            "with matching dimensions [num_topics times num_words]."
                        )
                    if inits.shape != (self.K, self.V):
                        raise ValueError(
                            f"beta_shape_init and beta_rate_init must have shape ({self.K}, {self.V}), "
                            f"got {inits.shape}"
                        )
        self.beta_rate_init = beta_rate_init
        self.beta_shape_init = beta_shape_init

    def _model(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray, i_batch: jnp.ndarray) -> None:  # type: ignore[override]
        """Define the probabilistic model using NumPyro.

        Model structure:
        - x (N,): ideal points for each author
        - beta (K x V): topic-word distributions
        - eta (K x V): ideal point loadings for words
        - theta (D x K): document-topic intensities
        - Y_batch: observed word counts with Poisson likelihood

        Parameters
        ----------
        Y_batch : jnp.ndarray
            The observed word counts for the current batch (batch_size, V).
        d_batch : jnp.ndarray
            Indices of documents in the current batch (batch_size,).
        i_batch : jnp.ndarray
            Indices of authors for the documents in the batch (batch_size,).
        """
        with plate("i", self.N):
            # Sample the per-unit latent variables (ideal points)
            x = sample("x", dist.Normal())

        with plate("k", size=self.K, dim=-2):
            with plate("k_v", size=self.V, dim=-1):
                beta = sample("beta", dist.Gamma(0.3, 0.3))
                eta = sample("eta", dist.Normal())

        with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
            with plate("d_k", size=self.K, dim=-1):
                # Sample document-level latent variables (topic intensities)
                theta = sample("theta", dist.Gamma(0.3, 0.3))

            # Compute Poisson rates for each word
            P = jnp.sum(
                jnp.expand_dims(theta, 2)
                * jnp.expand_dims(beta, 0)
                * jnp.exp(jnp.expand_dims(x[i_batch], (1, 2)) * jnp.expand_dims(eta, 0)),
                1,
            )

            with plate("v", size=self.V, dim=-1):
                # Sample observed words
                sample("Y_batch", dist.Poisson(P), obs=Y_batch)

    def _guide(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray, i_batch: jnp.ndarray) -> None:  # type: ignore[override]
        """Define the variational guide for the model.

        Uses Gamma and LogNormal variational families for approximate posterior inference.

        Parameters
        ----------
        Y_batch : jnp.ndarray
            The observed word counts for the current batch (batch_size, V).
        d_batch : jnp.ndarray
            Indices of documents in the current batch (batch_size,).
        i_batch : jnp.ndarray
            Indices of authors for the documents in the batch (batch_size,).
        """
        mu_x = param("mu_x", init_value=-1 + 2 * random.uniform(random.PRNGKey(1), (self.N,)))
        sigma_x = param("sigma_x", init_value=jnp.ones([self.N]), constraint=constraints.positive)

        mu_eta = param("mu_eta", init_value=random.normal(random.PRNGKey(2), (self.K, self.V)))
        sigma_eta = param(
            "sigma_eta",
            init_value=jnp.ones([self.K, self.V]),
            constraint=constraints.positive,
        )

        mu_theta = param("mu_theta", init_value=jnp.zeros([self.D, self.K]))
        sigma_theta = param(
            "sigma_theta",
            init_value=jnp.ones([self.D, self.K]),
            constraint=constraints.positive,
        )

        # Add initial values for beta parameters if provided for the tv-tbip model
        if self.beta_shape_init is not None and self.time_varying:
            mu_beta = param(
                "mu_beta",
                init_value=self.beta_shape_init,
            )
        else:
            mu_beta = param("mu_beta", init_value=jnp.zeros([self.K, self.V]))

        # check if beta_shape init is not none and self.time_yvarying is true
        if self.beta_shape_init is not None and self.time_varying:
            sigma_beta = param(
                "sigma_beta",
                init_value=self.beta_rate_init,
                constraint=constraints.positive,
            )
        else:
            sigma_beta = param(
                "sigma_beta",
                init_value=jnp.ones([self.K, self.V]),
                constraint=constraints.positive,
            )

        with plate("i", self.N):
            sample("x", dist.Normal(mu_x, sigma_x))

        with plate("k", size=self.K, dim=-2):
            with plate("k_v", size=self.V, dim=-1):
                sample("beta", dist.LogNormal(mu_beta, sigma_beta))
                sample("eta", dist.Normal(mu_eta, sigma_eta))

        with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
            with plate("d_k", size=self.K, dim=-1):
                sample("theta", dist.LogNormal(mu_theta[d_batch], sigma_theta[d_batch]))

    def _get_batch(
        self, rng: jnp.ndarray, Y: sparse.csr_matrix
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample a random mini-batch from the corpus.

        Helper function specified exclusively for TBIP objects.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            Random number generator key.
        Y : scipy.sparse.csr_matrix
            The word counts array.

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            Y_batch : Word counts for the batch (batch_size, V).
            D_batch : Indices of documents in the batch (batch_size,).
            I_batch : Indices of authors for the documents in the batch (batch_size,).

        Raises
        ------
        AssertionError
            If batch dimensions don't match expected shape.
        """
        D_batch = random.choice(rng, jnp.arange(self.D), shape=(self.batch_size,))
        Y_batch = jnp.array(Y[D_batch].toarray())

        # Ensure the shape of Y_batch is (batch_size, V)
        assert Y_batch.shape == (
            self.batch_size,
            self.V,
        ), f"Shape mismatch: {Y_batch.shape} != ({self.batch_size}, {self.V})"

        I_batch = np.array(self.author_indices[D_batch])
        return Y_batch, D_batch, I_batch

    def train_step(self, num_steps: int, lr: float) -> dict:  # type: ignore[override]
        """Train the TBIP model using stochastic variational inference.

        Custom train function specified exclusively for TBIP objects.

        Parameters
        ----------
        num_steps : int
            Number of training steps. Must be > 0.
        lr : float
            Learning rate for the optimizer. Must be > 0.

        Returns
        -------
        dict
            A dictionary containing the estimated parameter values after training.

        Raises
        ------
        ValueError
            If num_steps or lr are invalid.
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {num_steps}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        svi_batch = SVI(
            model=self._model, guide=self._guide, optim=adam(lr), loss=TraceMeanField_ELBO()
        )
        svi_batch_update = jit(svi_batch.update)

        Y_batch, D_batch, I_batch = self._get_batch(random.PRNGKey(1), self.counts)

        svi_state = svi_batch.init(
            random.PRNGKey(0), Y_batch=Y_batch, d_batch=D_batch, i_batch=I_batch
        )

        rngs = random.split(random.PRNGKey(2), num_steps)
        # losses = list()
        pbar = tqdm(range(num_steps))

        for step in pbar:
            Y_batch, D_batch, I_batch = self._get_batch(rngs[step], self.counts)
            svi_state, loss = svi_batch_update(
                svi_state, Y_batch=Y_batch, d_batch=D_batch, i_batch=I_batch
            )
            loss = loss / self.D
            self.Metrics.loss.append(float(loss))
            # losses.append(loss)
            if step % 10 == 0:
                pbar.set_description(
                    "Init loss: "
                    + "{:10.4f}".format(self.Metrics.loss[0])
                    + f"; Avg loss (last {10} iter): "
                    + "{:10.4f}".format(jnp.array(self.Metrics.loss[-10:]).mean())
                )

        self.estimated_params = svi_batch.get_params(svi_state)

        return self.estimated_params

    def return_topics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the dominant topic for each document.

        Uses the LogNormal variational posterior for theta:
        ``E[theta] = exp(mu + sigma^2 / 2)``.

        Returns
        -------
        categories : np.ndarray
            Array of topic indices for each document (shape: D,).
        E_theta : np.ndarray
            Estimated topic proportions for each document (shape: D, K).

        Raises
        ------
        ValueError
            If model has not been trained yet.
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling return_topics()")

        mu = np.asarray(self.estimated_params["mu_theta"])
        sigma = np.asarray(self.estimated_params["sigma_theta"])
        E_theta = np.exp(mu + sigma**2 / 2.0)
        return np.argmax(E_theta, axis=1), E_theta

    def return_beta(self) -> pd.DataFrame:
        """Return the topic-word association matrix.

        Uses the LogNormal variational posterior for beta:
        ``E[beta] = exp(mu + sigma^2 / 2)``.

        Returns
        -------
        pd.DataFrame
            DataFrame with words as index and topics as columns.

        Raises
        ------
        ValueError
            If model has not been trained yet.
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling return_beta()")

        mu = np.asarray(self.estimated_params["mu_beta"])
        sigma = np.asarray(self.estimated_params["sigma_beta"])
        E_beta = np.exp(mu + sigma**2 / 2.0)
        return pd.DataFrame(np.transpose(E_beta), index=self.vocab)

    def return_ideal_points(self) -> pd.DataFrame:
        """Return ideal point estimates for all authors.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``['author', 'ideal_point', 'std']``
            sorted by ideal point.

        Raises
        ------
        ValueError
            If model has not been trained yet.
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling return_ideal_points()")

        mu_x = np.asarray(self.estimated_params["mu_x"])
        sigma_x = np.asarray(self.estimated_params["sigma_x"])

        df = pd.DataFrame(
            {
                "author": list(self.author_map.keys()),
                "ideal_point": [float(mu_x[idx]) for idx in self.author_map.values()],
                "std": [float(sigma_x[idx]) for idx in self.author_map.values()],
            }
        )
        return df.sort_values("ideal_point").reset_index(drop=True)

    def return_ideological_words(self, topic: int, n: int = 10) -> pd.DataFrame:
        """Return words with the strongest ideological loading for a topic.

        For a given topic *k*, ranks words by the magnitude of their
        ideological coefficient ``eta[k, :]``. Words with large positive
        ``eta`` are associated with higher ideal-point values, and vice
        versa.

        Parameters
        ----------
        topic : int
            Topic index (0-based).
        n : int, optional
            Number of top words per direction (default 10).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``['word', 'eta', 'direction']`` where
            direction is ``'positive'`` or ``'negative'``.

        Raises
        ------
        ValueError
            If model has not been trained or topic index is invalid.
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling return_ideological_words()")
        if topic < 0 or topic >= self.K:
            raise ValueError(f"topic must be in [0, {self.K - 1}], got {topic}")

        mu_eta = np.asarray(self.estimated_params["mu_eta"])
        eta_k = mu_eta[topic, :]

        # Top positive
        pos_idx = np.argsort(eta_k)[::-1][:n]
        pos_df = pd.DataFrame(
            {
                "word": self.vocab[pos_idx],
                "eta": eta_k[pos_idx],
                "direction": "positive",
            }
        )
        # Top negative
        neg_idx = np.argsort(eta_k)[:n]
        neg_df = pd.DataFrame(
            {
                "word": self.vocab[neg_idx],
                "eta": eta_k[neg_idx],
                "direction": "negative",
            }
        )
        return pd.concat([pos_df, neg_df], ignore_index=True)

    def __create_author_ideal_map(self) -> dict:
        """Create a mapping of authors to their estimated ideal points.

        Returns
        -------
        dict
            A dictionary mapping each author to their estimated ideal point.
        """
        x_est = self.estimated_params["mu_x"]
        author_ideal_map = {author: x_est[idx] for author, idx in self.author_map.items()}
        return author_ideal_map

    def plot_ideal_points(
        self,
        selected_authors: Optional[list] = None,
        show_ci: bool = False,
        ci: float = 0.95,
        figsize: Tuple[float, float] = (12, 2),
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the ideal points of authors on a 1-D axis.

        Parameters
        ----------
        selected_authors : list, optional
            Authors to label (default: all authors).
        show_ci : bool, optional
            If True, display horizontal error bars showing the credible
            interval derived from ``sigma_x``.
        ci : float, optional
            Credible-interval level when *show_ci* is True (default 0.95).
        figsize : tuple, optional
            Figure size (default ``(12, 2)``).
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        tuple of (plt.Figure, plt.Axes)
        """
        from scipy import stats as sp_stats

        with plt.rc_context(self._setup_academic_style()):
            fig, ax = plt.subplots(figsize=figsize)

            if selected_authors is None:
                selected_authors = list(self.authors_unique)

            mu_x = np.asarray(self.estimated_params["mu_x"])
            sigma_x = np.asarray(self.estimated_params["sigma_x"])

            z = sp_stats.norm.ppf(1.0 - (1.0 - ci) / 2.0)

            for author, idx in self.author_map.items():
                x_val = float(mu_x[idx])
                if show_ci:
                    err = z * float(sigma_x[idx])
                    ax.errorbar(
                        x_val,
                        0,
                        xerr=err,
                        fmt="o",
                        color="black",
                        markersize=4,
                        capsize=2,
                        linewidth=0.6,
                    )
                else:
                    ax.scatter(x_val, 0, c="black", s=20, zorder=3)

                if author in selected_authors:
                    ax.annotate(
                        str(author),
                        xy=(x_val, 0.0),
                        xytext=(0, 8),
                        textcoords="offset points",
                        rotation=30,
                        fontsize=9,
                        ha="center",
                    )

            ax.set_yticks([])
            ax.set_xlabel("Ideal point")
            ax.set_title("Estimated ideal points")
            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax

    def _summary_extra(self) -> str:
        """TBIP-specific summary information."""
        lines = [f"  Authors (N):              {self.N}"]
        if self.estimated_params:
            mu_x = np.asarray(self.estimated_params["mu_x"])
            lines.append(f"  Ideal-point range:        [{mu_x.min():.3f}, {mu_x.max():.3f}]")
            lines.append(f"  Ideal-point std:          {mu_x.std():.3f}")
        return "\n".join(lines)
