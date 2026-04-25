from typing import Any, Dict, List, Optional, Tuple
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import scipy.sparse as sparse
import seaborn as sns
from jax import jit, random
from numpyro import param, plate, sample
from numpyro.distributions import constraints
from numpyro.infer import SVI, TraceMeanField_ELBO
from scipy.special import digamma
from optax import adam
from tqdm import tqdm
from wordcloud import WordCloud

# Abstract class - defining the minimum requirements for the probabilistic model
from .numpyro_model import NumpyroModel

class STBS(NumpyroModel):
    """
    STBS Model

    This class models structural text-based scaling (STBS), including 
    topic-specific ideal points and author-specific covariates for 
    documents authored by different individuals. The model aims to 
    capture how ideology can vary by topic and with external variables.
    """

    def __init__(
        self,
        counts: sparse.csr_matrix,
        vocab: np.ndarray,
        num_topics: int,
        authors: np.ndarray,
        batch_size: int,
        X_design_matrix: Optional[np.ndarray] = None, 
        beta_shape_init: np.ndarray = None,
        beta_rate_init: np.ndarray = None,
        theta_shape_init: np.ndarray = None,
        theta_rate_init: np.ndarray = None,
        i_mu_init: np.ndarray = None,
    ) -> None:
        """
        Initialize the STBS model.

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
        X_design_matrix : np.ndarray or pd.DataFrame
            Author-level covariates of shape (N, L). Row i must correspond to the
            i-th element of the sorted unique authors from `authors` (i.e., np.unique(authors)).
        batch_size : int
            The number of documents to be processed in each batch.
            Must satisfy 0 < batch_size <= D.
        beta_shape_init : np.ndarray, optional
            Initial shape parameters for the topic-word distributions (default is None).
            Must have shape (K, V) if provided.
        beta_rate_init : np.ndarray, optional
            Initial rate parameters for the topic-word distributions (default is None).
            Must have shape (K, V) if provided.
        theta_shape_init : np.ndarray, optional
            Initial shape parameters for the document-topic distributions (default is None).
            Must have shape (D, K) if provided.
        theta_rate_init : np.ndarray, optional
            Initial rate parameters for the document-topic distributions (default is None).
            Must have shape (D, K) if provided.
        i_mu_init : np.ndarray, optional
            Initial mean parameters for the ideology-topic distributions (default is None).
            Must have shape (N, ) if provided.

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

        if X_design_matrix is not None:
            if isinstance(X_design_matrix, pd.DataFrame):
                self.covariates = list(X_design_matrix.columns)
                X_design_matrix = X_design_matrix.values
            else: 
                self.covariates = [f"cov_{i}" for i in range(X_design_matrix.shape[1])]
        
            X_design_matrix = np.asarray(X_design_matrix)
            if X_design_matrix.ndim != 2:
                raise ValueError(f"covariates must be 2D, got shape {X_design_matrix.shape}")
            if X_design_matrix.shape[0] != self.N:
                raise ValueError(f"covariates has {X_design_matrix.shape[0]} rows, expected {self.N}")
            if X_design_matrix.shape[1] == 0:
                raise ValueError("covariates matrix is empty (0 columns)")

        self.X_design_matrix = (
            jnp.array(X_design_matrix) if X_design_matrix is not None else jnp.ones((self.N, 1))
        )
        self.L = self.X_design_matrix.shape[1]

        # check if initialization parameters have the correct shape and are jnp.arrays
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

        for inits in [theta_shape_init, theta_rate_init]:
            if inits is None:
                warnings.warn(
                    "No initial values for theta parameters were provided. "
                    "The model will initialize them uniformly."
                )
            if inits is not None:
                if not isinstance(inits, (np.ndarray, jnp.ndarray)):
                    raise ValueError(
                        "theta_shape_init and theta_rate_init must be numpy or jnp.ndarray objects "
                        "with matching dimensions [num_documents times num_topics]."
                    )
                if inits.shape != (self.D, self.K):
                    raise ValueError(
                        f"theta_shape_init and theta_rate_init must have shape ({self.D}, {self.K}), "
                        f"got {inits.shape}"
                    )
        self.theta_rate_init = theta_rate_init
        self.theta_shape_init = theta_shape_init

        if i_mu_init is None:
            warnings.warn(
                "No initial values for the ideology parameters were provided. "
                "The model will initialize them uniformly."
            )
        if i_mu_init is not None:
            if not isinstance(i_mu_init, (np.ndarray, jnp.ndarray)):
                raise ValueError(
                    "i_mu_init must be a numpy or jnp.ndarray object "
                    "with shape [num_authors, ]."
                )
            if i_mu_init.shape != (self.N, ):
                raise ValueError(
                    f"i_mu_init must have shape ({self.N},), "
                    f"got {i_mu_init.shape}"
                )
        self.i_mu_init = i_mu_init
            

    def _model(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray, i_batch: jnp.ndarray) -> None:  # type: ignore[override]
        """Define the probabilistic model using NumPyro.

        Model structure:
        - beta (K x V): topic-word distributions
        - eta (K x V): ideal point loadings for words
        - iota (K x L): topic regression coefficients for covariates
        - i (N x K): author-topic ideal points
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

        with plate("v", size = self.V, dim = -1):
            b_beta = sample("b_beta", dist.Gamma(0.3, 1.0))

        with plate("k", size = self.K, dim = -1):
            b_rho = sample("b_rho", dist.Gamma(0.3, 1.0))
            rho = sample("rho", dist.Gamma(0.3, b_rho))
        
        with plate("k", size = self.K, dim = -2):
            with plate("k_v", size = self.V, dim = -1):
                beta = sample("beta", dist.Gamma(0.3, b_beta))
                eta = sample("eta", dist.Normal(0, jnp.tile(1/jnp.sqrt(rho), (self.V, 1)).T))

        with plate("l", size = self.L, dim = -1):
            b_omega = sample("b_omega", dist.Gamma(0.3, 1.0))
            omega = sample("omega", dist.Gamma(0.3, b_omega))
            iota_dot = sample("iota_dot", dist.Normal(0, 1))

        with plate("l", size = self.L, dim = -2):
            with plate("l_k", size = self.K, dim = -1):
                iota = sample("iota", dist.Normal(jnp.tile(iota_dot, (self.K, 1)).T, jnp.tile(1/jnp.sqrt(omega), (self.K, 1)).T))

        i_mu = jnp.matmul(self.X_design_matrix, iota)

        with plate("n", size = self.N, dim = -1):
            I = sample("I", dist.Gamma(0.3, 0.3))

        with plate("n", size = self.N, dim = -2):      
            with plate("k", size=self.K, dim = -1):
                # Sample the per-unit latent variables (ideal points)
                i = sample("i", dist.Normal(i_mu, jnp.tile(1/jnp.sqrt(I), (self.K, 1)).T))

        with plate("n", size = self.N, dim = -1):
            b_author = sample("b_author", dist.Gamma(0.3, 1.0))
        
        with plate("d", size = self.D, subsample_size = self.batch_size, dim = -2):
            b_author_d = b_author[i_batch] 
            b_author_dk = jnp.tile(b_author_d.reshape(-1, 1), (1, self.K))
  
            with plate("d_k", size = self.K, dim = -1):
                # Sample document-level latent variables (topic intensities)
                theta = sample("theta", dist.Gamma(0.3, b_author_dk))

                # Compute Poisson rates for each word
                P = jnp.sum(
                jnp.expand_dims(theta, axis = -1)
                * jnp.expand_dims(beta, axis = 0)
                * jnp.exp(jnp.expand_dims(eta, axis = 0)
                          * jnp.expand_dims(i[i_batch], axis = -1)), 1)
                
            with plate("v", size = self.V, dim = -1):
                # Sample observed words
                sample("Y_batch", dist.Poisson(P), obs = Y_batch)       


    def _guide(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray, i_batch: jnp.ndarray) -> None:  # type: ignore[override]
        """Define the variational guide for the model.

        Uses Gamma and Normal variational families for approximate posterior inference.

        Parameters
        ----------
        Y_batch : jnp.ndarray
            The observed word counts for the current batch (batch_size, V).
        d_batch : jnp.ndarray
            Indices of documents in the current batch (batch_size,).
        i_batch : jnp.ndarray
            Indices of authors for the documents in the batch (batch_size,).
        """

        b_beta_shape = param(
            "b_beta_shape", 
            init_value = jnp.ones(self.V), 
            constraint = constraints.positive,
        )
        b_beta_rate = param(
            "b_beta_rate", 
            init_value = jnp.ones(self.V), 
            constraint = constraints.positive,
        )

        # Add initial values for beta parameters if provided for the stbs model
        if self.beta_rate_init is not None:
            beta_rate = param(
                "beta_rate", 
                init_value = self.beta_rate_init, 
                constraint = constraints.positive, 
            )
        else: 
            beta_rate = param(
                "beta_rate", 
                init_value = jnp.ones([self.K, self.V]), 
                constraint = constraints.positive, 
            )

        if self.beta_shape_init is not None:
            beta_shape = param(
                "beta_shape", 
                init_value = self.beta_shape_init, 
                constraint = constraints.positive,
            )
        else: 
            beta_shape = param(
                "beta_shape", 
                init_value = jnp.ones([self.K, self.V]), 
                constraint = constraints.positive, 
            )

        b_rho_shape = param(
            "b_rho_shape", 
            init_value = jnp.ones(self.K), 
            constraint = constraints.positive,
        )
        b_rho_rate = param(
            "b_rho_rate", 
            init_value = jnp.ones(self.K), 
            constraint = constraints.positive,
        )
        rho_shape = param(
            "rho_shape", 
            init_value = jnp.ones(self.K), 
            constraint = constraints.positive,
        )
        rho_rate = param(
            "rho_rate", 
            init_value = jnp.ones(self.K), 
            constraint = constraints.positive,
        )

        mu_eta = param(
            "mu_eta", 
            init_value = random.normal(random.PRNGKey(2), (self.K, self.V)),
        )
        sigma_eta = param(
            "sigma_eta", 
            init_value = jnp.ones([self.K, self.V]), 
            constraint = constraints.positive,
        ) 

        b_omega_shape = param(
            "b_omega_shape", 
            init_value = jnp.ones(self.L), 
            constraint = constraints.positive,
        )
        b_omega_rate = param(
            "b_omega_rate", 
            init_value = jnp.ones(self.L), 
            constraint = constraints.positive,
        )
        omega_shape = param(
            "omega_shape", 
            init_value = jnp.ones(self.L), 
            constraint = constraints.positive,
        )
        omega_rate = param(
            "omega_rate", 
            init_value = jnp.ones(self.L), 
            constraint = constraints.positive,
        )

        mu_iota_dot = param(
            "mu_iota_dot", 
            init_value = jnp.zeros(self.L),
        ) 
        sigma_iota_dot = param(
            "sigma_iota_dot", 
            init_value = jnp.ones(self.L), 
            constraint = constraints.positive,
        )
        mu_iota = param(
            "mu_iota", 
            init_value = jnp.zeros([self.L, self.K]),
        )
        sigma_iota = param(
            "sigma_iota", 
            init_value = jnp.ones([self.L, self.K]), 
            constraint = constraints.positive,
        )

        I_shape = param(
            "I_shape", 
            init_value = jnp.ones(self.N), 
            constraint = constraints.positive,
        )
        I_rate = param(
            "I_rate", 
            init_value = jnp.ones(self.N), 
            constraint = constraints.positive, 
        )
        
        # Add initial values for ideology parameters if provided for the stbs model
        if self.i_mu_init is not None:
            mu_i = param(
                "mu_i", 
                init_value = jnp.tile(self.i_mu_init, (self.K, 1)).T, 
            )
        else:
            mu_i = param(
                "mu_i", 
                init_value = jnp.zeros((self.N, self.K)), 
            )

        sigma_i = param(
            "sigma_i", 
            init_value = jnp.ones((self.N, self.K)), 
            constraint = constraints.positive, 
        )

        b_author_shape = param(
            "b_author_shape", 
            init_value = jnp.ones(self.N), 
            constraint = constraints.positive,
        )
        b_author_rate = param(
            "b_author_rate", 
            init_value = jnp.ones(self.N), 
            constraint = constraints.positive,
        )

        # Add initial values for theta parameters if provided for the stbs model
        if self.theta_rate_init is not None:
            theta_rate = param(
                "theta_rate", 
                init_value = self.theta_rate_init, 
                constraint = constraints.positive,
            )
        else: 
            theta_rate = param(
                "theta_rate", 
                init_value = jnp.ones([self.D, self.K]), 
                constraint = constraints.positive,
            )

        if self.theta_shape_init is not None:
            theta_shape = param(
                "theta_shape", 
                init_value = self.theta_shape_init, 
                constraint = constraints.positive,
            )
        else: 
            theta_shape = param(
                "theta_shape", 
                init_value = jnp.ones([self.D, self.K]), 
                constraint = constraints.positive,
            )

        with plate("v", size = self.V, dim = -1):
            sample("b_beta", dist.Gamma(b_beta_shape, b_beta_rate))

        with plate("k", size = self.K, dim = -1):
            sample("b_rho", dist.Gamma(b_rho_shape, b_rho_rate))
            sample("rho", dist.Gamma(rho_shape, rho_rate))

        with plate("k", size = self.K, dim = -2):  
            with plate("k_v", size = self.V, dim = -1):
                sample("beta", dist.Gamma(beta_shape, beta_rate))
                sample("eta", dist.Normal(mu_eta, sigma_eta))

        with plate("l", size = self.L, dim = -1): 
            sample("b_omega", dist.Gamma(b_omega_shape, b_omega_rate))
            sample("omega", dist.Gamma(omega_shape, omega_rate))
            sample("iota_dot", dist.Normal(mu_iota_dot, sigma_iota_dot))

        with plate("l", size = self.L, dim = -2):  
            with plate("l_k", size = self.K, dim = -1):
                sample("iota", dist.Normal(mu_iota, sigma_iota))

        with plate("n", size = self.N, dim = -1):
            sample("I", dist.Gamma(I_shape, I_rate))

        with plate("n", size = self.N, dim = -2):      
            with plate("k", size=self.K, dim = -1):
                sample("i", dist.Normal(mu_i, sigma_i))

        with plate("n", self.N, dim = -1):
            sample("b_author", dist.Gamma(b_author_shape, b_author_rate))

        with plate("d", size = self.D, subsample_size = self.batch_size, dim = -2):
            with plate("d_k", size = self.K, dim = -1):
                sample("theta", dist.Gamma(theta_shape[d_batch], theta_rate[d_batch]))


    def _get_batch(
        self, rng: jnp.ndarray, Y: sparse.csr_matrix
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample a random mini-batch from the corpus.

        Helper function specified exclusively for TBIP and STBS objects.

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
        """Train the STBS model using stochastic variational inference.

        Custom train function specified exclusively for TBIP and STBS objects.

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
 

    def plot_topic_wordclouds(
            self,
            n_words: int = 50,
            figsize: Tuple[int, int] = (16, 12),
            ideology_values: Optional[Tuple[float, ...]] = (-1, 0, 1),
            topics: Optional[List[int]] = None,
            log_corrected: bool = True,
            save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot wordclouds for each topic, optionally at multiple ideology positions.
        When ``ideology_values`` is ``None``, delegates to the base class and
        plots one wordcloud per topic using raw beta values. When ``ideology_values`` 
        is set (default: ``(-1, 0, 1)``), produces a grid of shape 
        ``(n_topics, len(ideology_values))``.

        Parameters
        ----------
        n_words : int, optional
            Maximum number of words per wordcloud (default 50).
        figsize : tuple, optional
            Figure size ``(width, height)`` (default ``(16, 12)``).
        save_path : str, optional
            Path to save the figure.
        ideology_values : tuple of float or None, optional
            Ideal point values for which to draw wordclouds. Default values
            are ``(-1, 0, 1)``. Pass ``None`` to fall back to base class 
            behaviour (raw beta, no ideology).
        topics : list of int or None, optional
            Subset of topic indices to plot. If None, all K topics are shown.
        log_corrected : bool, optional
            If True (default), uses log-scale ideology-corrected intensities. 
            If False, uses the linear approximation ``beta * exp(eta * i)`` instead. 
            Ignored when ideology_values is None.

        Returns
        -------
        tuple of (plt.Figure, np.ndarray of Axes)
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling plot_topic_wordclouds()")
        
        if ideology_values is None:
            return super().plot_topic_wordclouds(
                n_words = n_words,
                figsize = figsize or (16, 12),
                save_path = save_path)

        topic_indices = list(topics) if topics is not None else list(range(self.K))

        if not topic_indices:
            raise ValueError("topics list is empty.")
        if any(t < 0 or t >= self.K for t in topic_indices):
            raise ValueError(f"topics must be indices in [0, {self.K - 1}].")

        beta_shape = self.estimated_params["beta_shape"]
        beta_rate  = self.estimated_params["beta_rate"]
        mu_eta     = self.estimated_params["mu_eta"]

        word_scores: List[List[dict]] = []

        for k in topic_indices:
            topic_scores = []
            for i_val in ideology_values:
                if log_corrected:
                    s = digamma(beta_shape[k]) - np.log(beta_rate[k]) + i_val * mu_eta[k]
                    s = s - s.min() + 0.05 * (s.max() - s.min())
                else:
                    s = (beta_shape[k] / beta_rate[k]) * np.exp(mu_eta[k] * i_val)

                word_freq = dict(pd.Series(s, index=self.vocab).nlargest(n_words))
                topic_scores.append(word_freq)
            word_scores.append(topic_scores)

        topic_labels = [f"Topic {k}" for k in topic_indices]

        n_rows = len(topic_indices)
        n_cols = len(ideology_values)

        with plt.rc_context(self._setup_academic_style()):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

        score_label = "log-corrected" if log_corrected else "linear"
        for j, i_val in enumerate(ideology_values):
            sign = f"{i_val:+.0f}" if i_val != 0 else "0 (neutral)"
            axes[0, j].set_title(
                f"i = {sign}\n({score_label})",
                fontsize=10,
                fontweight="bold",)
            
        for row, label in enumerate(topic_labels):
            axes[row, 0].set_ylabel(label, fontsize=9, fontweight="bold", rotation=90, labelpad=4)
            for col, word_freq in enumerate(word_scores[row]):
                ax = axes[row, col]
                if word_freq:
                    wc = WordCloud(
                        width=400,
                        height=300,
                        background_color="white",
                        relative_scaling=0.5,
                        min_font_size=8,
                    ).generate_from_frequencies(word_freq)
                    ax.imshow(wc, interpolation="bilinear")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, axes
    

    def _summary_extra(self) -> str:
        """STBS-specific summary information."""
        lines = [f"  Authors (N):              {self.N}",
                 f"  Covariates (L):           {self.L}",
                 f"  Covariate names:          {', '.join(self.covariates)}",]
        
        if self.estimated_params:
            mu_i = np.asarray(self.estimated_params["mu_i"])
            lines.append(f"  Ideal-point range:        [{mu_i.min():.3f}, {mu_i.max():.3f}]")
            lines.append(f"  Ideal-point std:          {mu_i.std():.3f}")

            mu_iota = np.asarray(self.estimated_params["mu_iota"])
            topic_ranges = mu_iota.max(axis=1) - mu_iota.min(axis=1)
            topic_stds   = mu_iota.std(axis=1)

            lines.append(f"  Iota range (mean over topics): {topic_ranges.mean():.3f}  [{topic_ranges.min():.3f}, {topic_ranges.max():.3f}]")
            lines.append(f"  Iota std   (mean over topics): {topic_stds.mean():.3f}  [{topic_stds.min():.3f}, {topic_stds.max():.3f}]")

        return "\n".join(lines)
    

    def plot_topic_prevalence(
        self,
        topic_labels: dict = None,
        selected_topics: list = None,
        sort: bool = True,
        figsize: tuple = (8, 4),
        save_path: str = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Bar chart of mean normalised topic prevalence across the corpus."""

        if not self.estimated_params:
            raise ValueError("Model must be trained before calling plot_topic_prevalence()")

        theta = np.asarray(self.estimated_params["theta_shape"]) / np.asarray(self.estimated_params["theta_rate"])
        theta_norm = theta / theta.sum(axis=1, keepdims=True)
        mean_prev = theta_norm.mean(axis=0)

        K = theta_norm.shape[1]
        _label = lambda k: topic_labels[k] if topic_labels and k in topic_labels else f"Topic {k}"

        if selected_topics is not None:
            indices = np.array(selected_topics)
            mean_prev = mean_prev[indices]
            labels = [_label(k) for k in indices]
        else:
            labels = [_label(k) for k in range(K)]

        if sort:
            order = np.argsort(mean_prev)[::-1]
            labels_sorted = [labels[i] for i in order]
            prev_sorted   = mean_prev[order]
        else:
            labels_sorted = labels
            prev_sorted   = mean_prev

        with plt.rc_context(self._setup_academic_style()):
            fig, ax = plt.subplots(figsize=figsize)
            ax.bar(labels_sorted, prev_sorted, color="steelblue", edgecolor="none")
            ax.set_xlabel("Topic")
            ax.set_ylabel("Mean normalised proportion")
            ax.set_title("Corpus-level topic prevalence")
            plt.xticks(rotation=45, ha="right", fontsize=7)
            sns.despine()
            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig, ax


    def plot_author_topic_heatmap(
        self,
        topic_labels: dict = None,
        author_labels: dict = None,
        selected_topics: list = None,
        figsize: tuple = (16, 12),
        save_path: str = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Heatmap of mean normalised topic proportions per author (topics x authors).

        Authors are sorted by their dominant topic so similar authors cluster together.

        Parameters
        ----------
        topic_labels : dict or None
            {topic_index: "label"}
        author_labels : dict or None
            {author_index: "label"} — if None, uses raw author indices.
        selected_topics : list or None
            Integer topic indices to restrict the plot. If None, all topics shown.
        """

        if not self.estimated_params:
            raise ValueError("Model must be trained before calling plot_author_topic_heatmap()")

        theta = np.asarray(self.estimated_params["theta_shape"]) / np.asarray(self.estimated_params["theta_rate"])
        theta_norm = theta / theta.sum(axis=1, keepdims=True) 

        K = theta_norm.shape[1]
        _tlabel = lambda k: topic_labels[k] if topic_labels and k in topic_labels else f"Topic {k}"
        col_labels = [_tlabel(k) for k in range(K)]

        author_theta = (
            pd.DataFrame(theta_norm, columns=col_labels)
            .assign(author=self.author_indices)
            .groupby("author")
            .mean() 
        )

        if selected_topics is not None:
            sel_labels = [_tlabel(k) for k in selected_topics]
            author_theta = author_theta[sel_labels]

        # Sort authors by dominant topic so similar authors cluster
        dominant = author_theta.values.argmax(axis=1)
        sort_idx = np.argsort(dominant)
        author_theta = author_theta.iloc[sort_idx]

        # Author tick labels
        inv_map = {v: k for k, v in self.author_map.items()}
        if author_labels is not None:
            xtick_labels = [author_labels.get(a, str(a)) for a in author_theta.index]
        else:
            xtick_labels = [inv_map.get(a, str(a)) for a in author_theta.index]

        with plt.rc_context(self._setup_academic_style()):
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(
                author_theta.T,
                ax=ax,
                cmap="YlOrRd",
                linewidths=0,
                xticklabels=xtick_labels,
                yticklabels=author_theta.columns.tolist(),
                cbar_kws={"label": "Mean normalised proportion", "shrink": 0.6},
            )
            ax.set_xlabel("Author")
            ax.set_ylabel("Topic")
            ax.set_title("Author-topic intensity")
            ax.tick_params(axis="x", labelsize=6, rotation=90)
            ax.tick_params(axis="y", labelsize=7)
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig, ax


    def plot_ideol_points(
        self,
        group: bool = True,
        group_var: Optional[np.ndarray] = None,
        group_labels: Optional[dict] = None,
        group_palette: Optional[dict] = None,
        topic_labels: Optional[dict] = None,
        figsize: Tuple[float, float] = (16, 12),
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Dot plot of topic-specific ideological positions of all authors.

        Topics are ordered by the absolute difference between group-weighted
        average positions (most polarising topic at the top). Group-weighted
        averages are shown as black 'X' markers connected by a horizontal line.

        Parameters
        ----------
        group : bool, optional
            If True (default), colours dots by group. Falls back to
            ``self.i_mu_init`` if ``group_var`` is not provided.
            If False, all dots are plotted in a single colour.
        group_var : np.ndarray of shape (N,) or None
            Author-level grouping variable. Overrides ``self.i_mu_init`` when
            provided. Unique values are treated as group identifiers.
        group_labels : dict or None
            Mapping ``{value: "label"}``, e.g. ``{-1: "D", 0: "I", 1: "R"}``.
            If None, groups are labelled by their raw value.
        group_palette : dict or None
            Mapping ``{label: colour}``. If None, uses a default tab10 palette.
        topic_labels : dict or None
            Optional ``{topic_index: "label"}`` for y-axis tick labels.
        selected_topics : list or None
            Integer topic indices to restrict the plot. If None, all topics shown.
        figsize : tuple, optional
            Figure size (default ``(7, 5)``).
        save_path : str or None
            Path to save the figure.

        Returns
        -------
        tuple of (plt.Figure, plt.Axes)
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling plot_ideology_points()")

        # 1. Resolve groups
        if not group:
            groups = np.array(["all"] * self.N)
            group_palette = {"all": "steelblue"}
            group_labels  = {"all": "all"}
        else:
            if group_var is None:
                if self.i_mu_init is None:
                    raise ValueError("No group_var provided and i_mu_init was not stored on the model.")
                group_var = np.asarray(self.i_mu_init)
            else:
                group_var = np.asarray(group_var)
                if group_var.shape[0] != self.N:
                    raise ValueError(f"group_var must have length N={self.N}, got {group_var.shape[0]}.")

            unique_vals = sorted(np.unique(group_var))

            if group_labels is None:
                group_labels = {v: str(v) for v in unique_vals}
            groups = np.array([group_labels[v] for v in group_var])

            if group_palette is None:
                unique_group_names = [group_labels[v] for v in unique_vals]
                group_palette = dict(zip(
                    unique_group_names,
                    sns.color_palette("tab10", len(unique_group_names))
                ))

        theta = np.asarray(self.estimated_params["theta_shape"]) / np.asarray(self.estimated_params["theta_rate"])
        author_weights = (
            pd.DataFrame(theta, columns=[f"x{k}" for k in range(theta.shape[1])])
            .assign(author=self.author_indices)
            .groupby("author", as_index=False)
            .mean()
            .melt(id_vars="author", var_name="topic", value_name="weight")
        )
        author_weights["topic"] = author_weights["topic"].str.replace("^x", "", regex=True).astype(int)

        mu_i = np.asarray(self.estimated_params["mu_i"]) 
        author_ideology = (
            pd.DataFrame(mu_i, columns=[f"x{k}" for k in range(mu_i.shape[1])])
            .assign(author=list(self.author_map.values()), group=groups)
            .melt(id_vars=["author", "group"], var_name="topic", value_name="ideology")
        )
        author_ideology["topic"] = author_ideology["topic"].str.replace("^x", "", regex=True).astype(int)

        authors_weighted = author_ideology.merge(author_weights, on=["author", "topic"], how="left")

        group_ideology = (authors_weighted
            .groupby(["group", "topic"], as_index=False)
            .apply(lambda g: pd.Series({"ideology": (np.nansum(g["weight"] * g["ideology"]) / np.nansum(g["weight"]))}),
                include_groups=False,)
            .reset_index())

        top_groups = pd.Series(groups).value_counts().index[:2].tolist()

        if len(top_groups) < 2:
            K = group_ideology["topic"].nunique()
            topic_order = list(range(K))
        else:
            pivot = group_ideology[group_ideology["group"].isin(top_groups)].pivot(
                index="topic", columns="group", values="ideology")
            pivot["abs_delta"] = (pivot[top_groups[0]] - pivot[top_groups[1]]).abs()
            topic_order = (pivot.sort_values("abs_delta", ascending=False)
                        .reset_index()["topic"]
                        .tolist())
            
        _label = lambda t: topic_labels[int(t)] if topic_labels and int(t) in topic_labels else f"Topic {t}"
        label_order = [_label(t) for t in topic_order]

        author_ideology["topic_label"] = pd.Categorical(
            author_ideology["topic"].map(_label), categories=label_order, ordered=True)
        group_ideology["topic_label"] = pd.Categorical(
            group_ideology["topic"].map(_label), categories=label_order, ordered=True)

        with plt.rc_context(self._setup_academic_style()):
            fig, ax = plt.subplots(figsize=figsize)

            sns.scatterplot(
                data=author_ideology, x="ideology", y="topic_label",
                hue="group", palette=group_palette, alpha=0.55, s=18, ax=ax,
                legend=True,
            )

            for topic_lbl, grp in group_ideology.groupby("topic_label", observed=True):
                grp_top2 = grp[grp["group"].isin(top_groups)]
                if len(grp_top2) < 2:
                    continue
                xmin, xmax = grp_top2["ideology"].min(), grp_top2["ideology"].max()
                ax.hlines(y=topic_lbl, xmin=xmin, xmax=xmax, colors="black", linewidth=0.8, zorder=3)

                g0 = grp_top2[grp_top2["group"] == top_groups[0]]
                g1 = grp_top2[grp_top2["group"] == top_groups[1]]

                ax.scatter(g0["ideology"], [topic_lbl] * len(g0),
                        marker="D", s=60, facecolors=group_palette[top_groups[0]], 
                        edgecolors="black", linewidths=0.8, zorder=4)
                ax.scatter(g1["ideology"], [topic_lbl] * len(g1),
                        marker="s", s=60, facecolors=group_palette[top_groups[1]], 
                        edgecolors="black", linewidths=0.8, zorder=4)

            for topic_lbl in label_order:
                ax.axhline(y=topic_lbl, linestyle="--", color="lightgray", linewidth=0.8, zorder=0)
            ax.axvline(0, linestyle="--", color="gray", linewidth=0.8)

            ax.set_xlabel("Ideological position")
            ax.set_ylabel("Topic (sorted by polarisation)")

            ax.legend(
                title="",
                loc="upper center",
                ncol=len(group_palette),
                bbox_to_anchor=(0.5, 1.05),
                frameon=False,
            )

            sns.despine()
            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig, ax


    def plot_iota_credible_intervals(
        self,
        topic_labels: dict = None,
        covariate_labels: dict = None,
        selected_topics: list = None,
        selected_covariates: list = None,
        ci: float = 0.95,
        figsize: tuple = (16, 12),
        save_path: str = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Single CI plot with selected covariates on y-axis and topics as hue."""
        from scipy.stats import norm as sp_norm

        if not self.estimated_params:
            raise ValueError("Model must be trained before calling plot_iota_credible_intervals()")

        mu_iota    = np.asarray(self.estimated_params["mu_iota"]).T  
        sigma_iota = np.asarray(self.estimated_params["sigma_iota"]).T 

        K, P = mu_iota.shape
        z = sp_norm.ppf((1 + ci) / 2)

        _tlabel = lambda k: topic_labels[k] if topic_labels and k in topic_labels else f"Topic {k}"
        _clabel = lambda p: (
            covariate_labels[p] if covariate_labels and p in covariate_labels
            else self.covariates[p] if hasattr(self, "covariates") and p < len(self.covariates)
            else f"Cov {p}"
        )

        if selected_covariates is not None:
            if isinstance(selected_covariates[0], str):
                selected_covariates = [list(self.covariates).index(c) for c in selected_covariates]
            cov_idx = selected_covariates
        else:
            cov_idx = list(range(P))

        topic_idx = selected_topics if selected_topics is not None else list(range(K))

        mu_sub    = mu_iota[np.ix_(topic_idx, cov_idx)] 
        sigma_sub = sigma_iota[np.ix_(topic_idx, cov_idx)]

        col_labels = [_clabel(p) for p in cov_idx]
        n_covs     = len(cov_idx)
        n_topics   = len(topic_idx)

        palette = sns.color_palette("tab10", n_topics)
        offsets = np.linspace(-0.3, 0.3, n_topics)

        with plt.rc_context(self._setup_academic_style()):
            fig, ax = plt.subplots(figsize=figsize)
            ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", zorder=0)

            for i, k in enumerate(topic_idx):
                mu_k    = mu_sub[i]
                sigma_k = sigma_sub[i]
                lo      = mu_k - z * sigma_k
                hi      = mu_k + z * sigma_k
                excludes_zero = (lo > 0) | (hi < 0)
                color   = palette[i]

                for j in range(n_covs):
                    y = j + offsets[i]
                    ax.plot([lo[j], hi[j]], [y, y],
                            color=color, linewidth=0.8, alpha=0.7, zorder=1)
                    ax.scatter(mu_k[j], y,
                            color=color,
                            s=30 if excludes_zero[j] else 15,
                            zorder=2,
                            marker="D" if excludes_zero[j] else "o")

            ax.set_yticks(range(n_covs))
            ax.set_yticklabels(col_labels, fontsize=7)
            ax.set_xlabel("Iota (ideology coefficient)")
            ax.set_title(f"Iota credible intervals ({int(ci*100)}%)")
            ax.tick_params(axis="x", labelsize=7)

            handles = [
                plt.Line2D([0], [0], color=palette[i], linewidth=1.5,
                        label=_tlabel(topic_idx[i]))
                for i in range(n_topics)
            ]
            ax.legend(handles=handles, title="Topic", frameon=False,
                    fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")

            sns.despine()
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig, ax


    def return_ideal_points(self) -> pd.DataFrame:
        """Return ideal point estimates for all authors and topics.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``['author', 'topic', 'ideal_point', 'std']``
            sorted by topic then ideal point.

        Raises
        ------
        ValueError
            If model has not been trained yet.
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling return_ideal_points()")

        mu_i    = np.asarray(self.estimated_params["mu_i"])
        sigma_i = np.asarray(self.estimated_params["sigma_i"])

        rows = []
        for author, idx in self.author_map.items():
            for k in range(self.K):
                rows.append({"author": author,
                             "topic": k,
                             "ideal_point": float(mu_i[idx, k]),
                             "std": float(sigma_i[idx, k]),})

        df = pd.DataFrame(rows)
        return df.sort_values(["topic", "ideal_point"]).reset_index(drop=True)


    def return_ideal_covariates(self) -> pd.DataFrame:
        """Return covariate regression coefficient estimates (iota).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``['covariate', 'topic', 'iota', 'std']``
            sorted by topic then covariate.

        Raises
        ------
        ValueError
            If model has not been trained yet.
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling return_ideal_covariates()")

        mu_iota    = np.asarray(self.estimated_params["mu_iota"])
        sigma_iota = np.asarray(self.estimated_params["sigma_iota"])

        rows = []
        for l, covariate in enumerate(self.covariates):
            for k in range(self.K):
                rows.append({"covariate": covariate,
                             "topic": k,
                             "iota": float(mu_iota[l, k]),
                             "std": float(sigma_iota[l, k]),})

        df = pd.DataFrame(rows)
        return df.sort_values(["topic", "covariate"]).reset_index(drop=True)




