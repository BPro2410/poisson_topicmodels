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
class CSPF(NumpyroModel):
    """
    Covariate Seeded Poisson Factorization (CSPF) topic model.

    Combines guided topic discovery with covariate effects. CSPF incorporates both
    keyword priors for topic guidance and document-level covariates to capture how
    topics vary with external variables.

    Parameters
    ----------
    counts : scipy.sparse.csr_matrix
        Document-term matrix of shape (D, V) with word counts.
    vocab : np.ndarray
        Vocabulary array of shape (V,) containing word terms.
    keywords : Dict[int, List[str]]
        Dictionary mapping topic indices to lists of seed words.
    residual_topics : int
        Number of residual (unsupervised) topics. Must be >= 0.
    batch_size : int
        Mini-batch size for stochastic variational inference.
    X_design_matrix : np.ndarray or pd.DataFrame
        Document-level covariates of shape (D, C).

    Attributes
    ----------
    D : int
        Number of documents.
    V : int
        Vocabulary size.
    K : int
        Total number of topics (seeded + residual).
    C : int
        Number of covariate features.
    """

    def __init__(
        self,
        counts: sparse.csr_matrix,
        vocab: np.ndarray,
        keywords: Dict[int, List[str]],
        residual_topics: int,
        batch_size: int,
        X_design_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the CSPF model with input validation.

        Parameters
        ----------
        counts : scipy.sparse.csr_matrix
            Document-term matrix.
        vocab : np.ndarray
            Vocabulary array.
        keywords : Dict[int, List[str]]
            Seed words for guided topics.
        residual_topics : int
            Number of unsupervised topics.
        batch_size : int
            Mini-batch size.
        X_design_matrix : np.ndarray or pd.DataFrame, optional
            Document-level covariates.

        Raises
        ------
        TypeError
            If inputs have wrong types.
        ValueError
            If dimensions or content are invalid.
        """
        super().__init__()

        # Input validation (similar to SPF and CPF)
        if not sparse.issparse(counts):
            raise TypeError(f"counts must be a scipy sparse matrix, got {type(counts).__name__}")

        D, V = counts.shape
        if D == 0 or V == 0:
            raise ValueError(f"counts matrix is empty: shape ({D}, {V})")

        if vocab.shape[0] != V:
            raise ValueError(f"vocab size {vocab.shape[0]} != counts columns {V}")

        if not isinstance(keywords, dict):
            raise TypeError(f"keywords must be dict, got {type(keywords).__name__}")

        if residual_topics < 0:
            raise ValueError(f"residual_topics must be >= 0, got {residual_topics}")

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

        # Validate keywords
        vocab_set = set(vocab)
        for topic_id, words in keywords.items():
            for word in words:
                if word not in vocab_set:
                    raise ValueError(f"Keyword '{word}' (topic {topic_id}) not in vocabulary")

        # Store validated inputs
        self.counts = counts
        self.D = D
        self.V = V
        self.vocab = vocab
        self.K = residual_topics + len(keywords)
        self.keywords = keywords
        self.residual_topics = residual_topics
        self.batch_size = batch_size

        # Compute keyword indices
        kw_indices_topics = [
            (idx, list(vocab).index(keyword))
            for idx, topic_id in enumerate(keywords.keys())
            for keyword in keywords[topic_id]
            if keyword in vocab
        ]
        self.Tilde_V = len(kw_indices_topics)
        self.kw_indices = tuple(zip(*kw_indices_topics)) if kw_indices_topics else ((), ())

        # Covariates
        self.X_design_matrix = (
            jnp.array(X_design_matrix) if X_design_matrix is not None else jnp.ones((D, 1))
        )
        self.C = self.X_design_matrix.shape[1]
        self.covariates = (
            list(X_design_matrix.columns)
            if isinstance(X_design_matrix, pd.DataFrame)
            else [f"cov_{i}" for i in range(self.C)]
        )

    # -- Model --
    def _model(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray) -> None:
        """
        Define the probabilistic generative model using NumPyro.

        Combines seeded topics with covariate effects.

        Parameters
        ----------
        Y_batch : jnp.ndarray
            Batch of observed word counts.
        d_batch : jnp.ndarray
            Document indices in batch.
        """
        with plate("k", size=self.K, dim=-2):
            with plate("k_v", size=self.V, dim=-1):
                beta = sample("beta", dist.Gamma(0.3, 0.3))

        with plate("tilde_v", size=self.Tilde_V):
            beta_tilde = sample("beta_tilde", dist.Gamma(1.0, 0.3))

        # Update beta
        beta = beta.at[self.kw_indices].add(beta_tilde)

        # hier covariates
        with plate("c", size=self.C):
            zeta = sample("zeta", dist.Normal(0.0, 1.0))
            rho = sample("rho", dist.Gamma(0.3, 1.0))
            omega = sample("omega", dist.Gamma(0.3, rho))

        with plate("c", size=self.C, dim=-2):
            with plate("c_k", size=self.K, dim=-1):
                lambda_ = sample(
                    "lambda",
                    dist.Normal(jnp.tile(zeta, (self.K, 1)).T, jnp.tile(omega, (self.K, 1)).T),
                )

        a_theta_S = jax.nn.softplus(jnp.matmul(self.X_design_matrix, lambda_))[d_batch]

        # ACHTUNG: Hier haben wir eine subsample_size wegen der batch size!
        with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
            with plate("d_k", size=self.K, dim=-1):
                theta = sample("theta", dist.Gamma(a_theta_S, 0.3))

            # ACHTUNG: P = muss eingerückt sein in der d plate - ich weiß aber nicht genau warum das so ist
            P = jnp.matmul(theta, beta)

            with plate("d_v", size=self.V, dim=-1):
                sample("Y_batch", dist.Poisson(P), obs=Y_batch)

    # -- Guide, i.e. variational family --
    def _guide(self, Y_batch, d_batch):
        """
        Define the variational guide for the model.

        Parameters
        ----------
        Y_batch : numpy.ndarray
            The observed word counts for the current batch.
        d_batch : numpy.ndarray
            Indices of documents in the current batch.
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

        a_beta_tilde = param(
            "beta_tilde_shape",
            init_value=jnp.ones([self.Tilde_V]) * 2,
            constraint=constraints.positive,
        )
        b_beta_tilde = param(
            "beta_tilde_rate", init_value=jnp.ones([self.Tilde_V]), constraint=constraints.positive
        )

        location_lambda = param("lambda_location", init_value=jnp.zeros([self.C, self.K]))
        scale_lambda = param(
            "lambda_scale", init_value=jnp.ones([self.C, self.K]), constraint=constraints.positive
        )

        location_zeta = param("zeta_location", init_value=jnp.zeros(self.C))
        scale_zeta = param(
            "zeta_scale", init_value=jnp.ones(self.C), constraint=constraints.positive
        )

        a_rho = param("rho_shape", init_value=jnp.ones(self.C), constraint=constraints.positive)
        b_rho = param("rho_rate", init_value=jnp.ones(self.C), constraint=constraints.positive)

        a_omega = param("omega_shape", init_value=jnp.ones(self.C), constraint=constraints.positive)
        b_omega = param("omega_rate", init_value=jnp.ones(self.C), constraint=constraints.positive)

        with plate("k", size=self.K, dim=-2):
            with plate("k_v", size=self.V, dim=-1):
                beta = sample("beta", dist.Gamma(a_beta, b_beta))

        with plate("tilde_v", size=self.Tilde_V):
            beta_tilde = sample("beta_tilde", dist.Gamma(a_beta_tilde, b_beta_tilde))

        with plate("c", size=self.C):
            zeta = sample("zeta", dist.Normal(location_zeta, scale_zeta))
            rho = sample("rho", dist.Gamma(a_rho, b_rho))
            omega = sample("omega", dist.Gamma(a_omega, b_omega))

        with plate("c", size=self.C, dim=-2):
            with plate("c_k", size=self.K, dim=-1):
                lambda_ = sample("lambda", dist.Normal(location_lambda, scale_lambda))

        with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
            with plate("d_k", size=self.K, dim=-1):
                theta = sample("theta", dist.Gamma(a_theta[d_batch], b_theta[d_batch]))

    def return_topics(self):
        """
        Return the topics for each document.

        Returns
        -------
        tuple
            topics : numpy.ndarray
                Array of topic names for each document.
            E_theta : numpy.ndarray
                Estimated topic proportions for each document.
        """

        def recode_cats(argmaxes, keywords):
            """
            Recodes the argmax index into topic strings.

            :param argmaxes: np.array() or jnp.array() because of vectorized parallel computing
            :param keywords: Dictionary containing keyword topics
            :return: Array of recoded topics
            """
            num_keywords = len(keywords.keys())
            max_index = num_keywords - 1
            keyword_keys = np.array(list(keywords.keys())).astype(str)

            # clip argmaxes to be within the valid range of keyword topics
            argmaxes_clipped = np.clip(argmaxes, 0, max_index)

            topics = np.where(
                argmaxes <= max_index,
                keyword_keys[argmaxes_clipped],
                f"No_keyword_topic_{argmaxes - max_index}",
            )

            return topics

        E_theta = self.estimated_params["theta_shape"] / self.estimated_params["theta_rate"]

        categories = np.argmax(E_theta, axis=1)
        topics = recode_cats(np.array(categories), self.keywords)

        return topics, E_theta

    def return_beta(self):
        """
        Return the beta matrix for the model.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the beta matrix with words as rows and topics as columns.
        """
        E_beta = self.estimated_params["beta_shape"] / self.estimated_params["beta_rate"]
        E_beta_tilde = (
            self.estimated_params["beta_tilde_shape"] / self.estimated_params["beta_tilde_rate"]
        )
        E_beta = E_beta.at[self.kw_indices].add(E_beta_tilde)

        rs_names = [f"residual_topic_{i+1}" for i in range(self.residual_topics)]

        return pd.DataFrame(
            jnp.transpose(E_beta), index=self.vocab, columns=list(self.keywords.keys()) + rs_names
        )

    def return_covariate_effects(self):
        """
        Return the covariate effects for the model.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the covariate effects with covariates as rows and topics as columns.
        """
        topic_names = list(self.keywords.keys())
        rs_names = [f"residual_topic_{i+1}" for i in range(self.residual_topics)]
        cols = topic_names + rs_names
        index = self.covariates
        return pd.DataFrame(self.estimated_params["lambda_location"], index=index, columns=cols)
