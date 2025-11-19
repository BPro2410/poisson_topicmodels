from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import scipy.sparse as sparse
from jax import jit, random
from numpyro import param, plate, sample
from numpyro.distributions import constraints

# Abstract class - defining the minimum requirements for the probabilistic model
from packages.models.numpyro_model import NumpyroModel


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
            if isinstance(X_design_matrix, pd.DataFrame):
                X_design_matrix = X_design_matrix.values

            X_design_matrix = np.asarray(X_design_matrix)
            if X_design_matrix.ndim != 2:
                raise ValueError(f"covariates must be 2D, got shape {X_design_matrix.shape}")
            if X_design_matrix.shape[0] != D:
                raise ValueError(f"covariates has {X_design_matrix.shape[0]} rows, expected {D}")
            if X_design_matrix.shape[1] == 0:
                raise ValueError("covariates matrix is empty (0 columns)")

        # Store validated inputs
        self.counts = counts
        self.D = D
        self.V = V
        self.vocab = vocab
        self.K = num_topics
        self.batch_size = batch_size
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
                beta = sample("beta", dist.Gamma(a_beta, b_beta))

        with plate("c", size=self.C, dim=-2):
            with plate("c_k", size=self.K, dim=-1):
                lambda_ = sample("phi", dist.Normal(location_lambda, scale_lambda))

        with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
            with plate("d_k", size=self.K, dim=-1):
                theta = sample("theta", dist.Gamma(a_theta[d_batch], b_theta[d_batch]))

    def return_covariate_effects(self):
        """
        Return the covariate effects for the model.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the covariate effects with covariates as rows and topics as columns.
        """
        rs_names = [f"residual_topic_{i+1}" for i in range(self.K)]
        index = self.covariates
        return pd.DataFrame(self.estimated_params["lambda_location"], index=index, columns=rs_names)
