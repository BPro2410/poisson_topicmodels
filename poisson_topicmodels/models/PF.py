from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import scipy.sparse as sparse
from numpyro import plate, sample
from numpyro.distributions import constraints

# Abstract class - defining the minimum requirements for the probabilistic model
from .numpyro_model import NumpyroModel


class PF(NumpyroModel):
    """
    Poisson Factorization (PF) topic model.

    Unsupervised baseline topic model using Poisson likelihood for word counts.
    Suitable for exploratory topic discovery in document collections.

    This model learns low-rank representations of documents and words, enabling
    interpretable topic extraction and downstream analysis.

    Parameters
    ----------
    counts : scipy.sparse.csr_matrix
        Document-term matrix of shape (D, V) with word counts.
    vocab : np.ndarray
        Vocabulary array of shape (V,) containing word terms.
    num_topics : int
        Number of topics K. Must be > 0.
    batch_size : int
        Mini-batch size for stochastic variational inference.
        Must satisfy 0 < batch_size <= D.
    initparams : dict, optional
        User-specified initial values for variational parameters in the guide.
    constantparams : dict, optional
        User-specified constant values for latent variables (not updated by SVI).
    hyperparams : dict, optional
        User-specified hyperparameters overriding default prior settings.

    Attributes
    ----------
    D : int
        Number of documents.
    V : int
        Vocabulary size.
    K : int
        Number of topics.
    counts : scipy.sparse.csr_matrix
        Document-term matrix.
    vocab : np.ndarray
        Vocabulary array.

    Examples
    --------
    >>> from scipy.sparse import random
    >>> import numpy as np
    >>> from topicmodels import PF
    >>> counts = random(100, 500, density=0.01, format='csr')
    >>> vocab = np.array([f'word_{i}' for i in range(500)])
    >>> model = PF(counts, vocab, num_topics=10, batch_size=32)
    >>> params = model.train_step(num_steps=100, lr=0.01, random_seed=42)
    >>> topics, proportions = model.return_topics()
    """

    def __init__(
        self,
        counts: sparse.csr_matrix,
        vocab: np.ndarray,
        num_topics: int,
        batch_size: int,
        initparams: Optional[Dict[str, Any]] = None,
        constantparams: Optional[Dict[str, Any]] = None,
        hyperparams: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the PF model with input validation.

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
        initparams : dict, optional
            Initial values for variational parameters.
        constantparams : dict, optional
            Fixed values for latent variables.
        hyperparams : dict, optional
            Hyperparameters overriding default priors.

        Raises
        ------
        TypeError
            If counts is not a sparse matrix or vocab is not array-like.
        ValueError
            If dimensions are invalid or inconsistent.
        """

        super().__init__(
            initparams=initparams, constantparams=constantparams, hyperparams=hyperparams
        )

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

        # Store validated inputs
        self.counts = counts
        self.V = V
        self.D = D
        self.vocab = vocab
        self.K = num_topics
        self.batch_size = batch_size

    def _model(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray) -> None:
        """
        Define the probabilistic generative model using NumPyro.

        Model structure:
        - Beta (K x V): topic-word distributions, Gamma(.3, .3) prior
        - Theta (D x K): document-topic distributions, Gamma(.3, .3) prior
        - Y_batch (batch_size x V): observed word counts, Poisson(Theta @ Beta)

        Parameters
        ----------
        Y_batch : jnp.ndarray
            Batch of observed word counts (batch_size, V).
        d_batch : jnp.ndarray
            Document indices in batch (batch_size,).
        """
        # Topic-word distributions: Beta ~ Gamma(0.3, 0.3)
        with plate("k", size=self.K, dim=-2):
            with plate("k_v", size=self.V, dim=-1):
                beta = self._sample(
                    "beta",
                    dist.Gamma(
                        self._hyperparam("a_beta", 0.3, positive=True),
                        self._hyperparam("b_beta", 0.3, positive=True),
                    ),
                    dimensions=(self.K, self.V),
                    positive=True,
                )

        # Document-topic distributions: Theta ~ Gamma(0.3, 0.3)
        with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
            with plate("d_k", size=self.K, dim=-1):
                theta = self._sample(
                    "theta",
                    dist.Gamma(
                        self._hyperparam("a_theta", 0.3, positive=True),
                        self._hyperparam("b_theta", 0.3, positive=True),
                    ),
                    dimensions=(self.batch_size, self.K),
                    positive=True,
                )

            # Poisson rate parameter
            P = jnp.matmul(theta, beta)

            # Word counts likelihood
            with plate("v", size=self.V, dim=-1):
                sample("Y_batch", dist.Poisson(P), obs=Y_batch)

    def _guide(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray) -> None:
        """
        Define the variational guide (approximate posterior).

        Uses Gamma variational family for all latent variables.

        Parameters
        ----------
        Y_batch : jnp.ndarray
            Batch of observed word counts.
        d_batch : jnp.ndarray
            Document indices in batch.
        """
        # Variational parameters for beta
        if not self._is_constant("beta"):
            a_beta = self._param(
                "beta_shape", init_value=jnp.ones([self.K, self.V]), constraint=constraints.positive
            )
            b_beta = self._param(
                "beta_rate",
                init_value=jnp.ones([self.K, self.V]) * self.D / 1000 * 2,
                constraint=constraints.positive,
            )

            # Variational distribution for beta
            with plate("k", size=self.K, dim=-2):
                with plate("k_v", size=self.V, dim=-1):
                    sample("beta", dist.Gamma(a_beta, b_beta))

        # Variational parameters for theta
        if not self._is_constant("theta"):
            a_theta = self._param(
                "theta_shape",
                init_value=jnp.ones([self.D, self.K]),
                constraint=constraints.positive,
            )
            b_theta = self._param(
                "theta_rate",
                init_value=jnp.ones([self.D, self.K]) * self.D / 1000,
                constraint=constraints.positive,
            )

            # Variational distribution for theta
            with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
                with plate("d_k", size=self.K, dim=-1):
                    sample("theta", dist.Gamma(a_theta[d_batch], b_theta[d_batch]))
