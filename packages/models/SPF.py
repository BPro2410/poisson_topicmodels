from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
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
from packages.models.numpyro_model import NumpyroModel


class SPF(NumpyroModel):
    """
    Seeded Poisson Factorization (SPF) topic model.

    Guided topic modeling with keyword priors. SPF allows researchers to incorporate
    domain knowledge by specifying seed words for each topic, which increases the
    topical prevalence of those words in the model.

    Parameters
    ----------
    counts : scipy.sparse.csr_matrix
        Document-term matrix of shape (D, V) with word counts.
    vocab : np.ndarray
        Vocabulary array of shape (V,) containing word terms.
    keywords : Dict[int, List[str]]
        Dictionary mapping topic indices to lists of seed words.
        Example: {0: ['climate', 'environment'], 1: ['economy', 'trade']}
    residual_topics : int
        Number of residual (unsupervised) topics. Must be >= 0.
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
        Total number of topics (seeded + residual).
    counts : scipy.sparse.csr_matrix
        Document-term matrix.
    vocab : np.ndarray
        Vocabulary array.
    keywords : Dict[int, List[str]]
        Seed words for guided topics.
    residual_topics : int
        Number of unsupervised topics.

    Examples
    --------
    >>> from scipy.sparse import random
    >>> import numpy as np
    >>> from topicmodels import SPF
    >>> counts = random(100, 500, density=0.01, format='csr')
    >>> vocab = np.array([f'word_{i}' for i in range(500)])
    >>> keywords = {
    ...     0: ['word_1', 'word_2', 'word_3'],
    ...     1: ['word_10', 'word_11', 'word_12'],
    ... }
    >>> model = SPF(counts, vocab, keywords, residual_topics=5, batch_size=32)
    >>> params = model.train_step(num_steps=100, lr=0.01, random_seed=42)
    """

    def __init__(
        self,
        counts: sparse.csr_matrix,
        vocab: np.ndarray,
        keywords: Dict[int, List[str]],
        residual_topics: int,
        batch_size: int,
    ) -> None:
        """
        Initialize the SPF model with input validation.

        Parameters
        ----------
        counts : scipy.sparse.csr_matrix
            Document-term matrix.
        vocab : np.ndarray
            Vocabulary array.
        keywords : Dict[int, List[str]]
            Seed words for each seeded topic.
        residual_topics : int
            Number of unsupervised topics.
        batch_size : int
            Mini-batch size.

        Raises
        ------
        TypeError
            If counts is not sparse or keywords is not dict.
        ValueError
            If dimensions are invalid or keywords contain unknown terms.
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

        if not isinstance(keywords, dict):
            raise TypeError(f"keywords must be dict, got {type(keywords).__name__}")

        if residual_topics < 0:
            raise ValueError(f"residual_topics must be >= 0, got {residual_topics}")

        if batch_size <= 0 or batch_size > D:
            raise ValueError(f"batch_size must satisfy 0 < batch_size <= {D}, got {batch_size}")

        # Validate keywords are in vocabulary
        vocab_set = set(vocab)
        for topic_id, words in keywords.items():
            for word in words:
                if word not in vocab_set:
                    raise ValueError(f"Keyword '{word}' (topic {topic_id}) not in vocabulary")

        # Store validated inputs
        self.counts = counts
        self.V = V
        self.D = D
        self.vocab = vocab
        self.residual_topics = residual_topics
        self.K = residual_topics + len(keywords)
        self.keywords = keywords

        # Compute keyword indices
        kw_indices_topics = [
            (idx, list(vocab).index(keyword))
            for idx, topic_id in enumerate(keywords.keys())
            for keyword in keywords[topic_id]
            if keyword in vocab
        ]
        self.Tilde_V = len(kw_indices_topics)
        self.kw_indices = tuple(zip(*kw_indices_topics)) if kw_indices_topics else ((), ())
        self.batch_size = batch_size

    def _model(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray) -> None:
        """
        Define the probabilistic generative model using NumPyro.

        Model structure:
        - Beta (K x V): topic-word distributions with keyword boosts
        - Beta_tilde: additional weights for seeded words
        - Theta (D x K): document-topic distributions
        - Y_batch (batch_size x V): observed word counts

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
                beta = sample("beta", dist.Gamma(0.3, 0.3))

        # Boost for seed words: Beta_tilde ~ Gamma(1, 0.3)
        with plate("tilde_v", size=self.Tilde_V):
            beta_tilde = sample("beta_tilde", dist.Gamma(1.0, 0.3))

        # Add seed word boosts to beta
        beta = beta.at[self.kw_indices].add(beta_tilde)

        # Document-topic distributions: Theta ~ Gamma(0.3, 0.3)
        with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
            with plate("d_k", size=self.K, dim=-1):
                theta = sample("theta", dist.Gamma(0.3, 0.3))

            # Poisson rate parameter
            P = jnp.matmul(theta, beta)

            # Word counts likelihood
            with plate("v", size=self.V, dim=-1):
                sample("Y_batch", dist.Poisson(P), obs=Y_batch)

    def _guide(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray) -> None:
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

        with plate("k", size=self.K, dim=-2):
            with plate("k_v", size=self.V, dim=-1):
                sample("beta", dist.Gamma(a_beta, b_beta))

        with plate("tilde_v", size=self.Tilde_V):
            sample("beta_tilde", dist.Gamma(a_beta_tilde, b_beta_tilde))

        with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
            with plate("d_k", size=self.K, dim=-1):
                sample("theta", dist.Gamma(a_theta[d_batch], b_theta[d_batch]))

    def return_topics(self):
        """
        Return the topics for each document. Reimplemented from the base class due to the guided
        topic modeling approach, where topics are not fully unsupervised.

        Returns
        -------
        tuple
            topics : numpy.ndarray
                Array of recoded topics.
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
        Return the beta matrix for the model, i.e. topic-word intensities.
        Reimplemented from the base class due to the higher rates approach for seed words.

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

    def return_top_words_per_topic(self, n=10):
        beta = self.return_beta()
        return {topic: beta[topic].nlargest(n).index.tolist() for topic in beta}

    def _recode_topics(self, indices: np.ndarray) -> np.ndarray:
        keyword_keys = np.array(list(self.keywords.keys()))
        num_keywords = len(keyword_keys)
        indices_clipped = np.clip(indices, 0, num_keywords - 1)
        return np.where(
            indices < num_keywords,
            keyword_keys[indices_clipped],
            [f"No_keyword_topic_{i - num_keywords}" for i in indices],
        )

    # Not implemented yet.
    # def infer_new_documents(self, new_counts):
    #     """
    #     Infer topic proportions for new documents using fixed learned parameters.

    #     Parameters
    #     ----------
    #     new_counts : scipy.sparse.csr_matrix
    #         Word counts for new documents.

    #     Returns
    #     -------
    #     np.ndarray
    #         Predicted topic names.
    #     """
    #     new_D = new_counts.shape[0]
    #     Y_new = jnp.array(new_counts.toarray())

    #     # Fixed learned topic-word matrix
    #     E_beta = self.estimated_params["beta_shape"] / self.estimated_params["beta_rate"]
    #     E_beta_tilde = self.estimated_params["beta_tilde_shape"] / self.estimated_params["beta_tilde_rate"]
    #     E_beta = E_beta.at[self.kw_indices].add(E_beta_tilde)

    #     # Set prior Gamma(.3, .3) for theta
    #     a_theta = jnp.ones((new_D, self.K)) * 0.3
    #     b_theta = jnp.ones((new_D, self.K)) * 0.3

    #     # Posterior mean of theta (MAP estimation under Gamma-Poisson conjugacy)
    #     P = jnp.dot(a_theta / b_theta, E_beta)  # Poisson rates
    #     theta_est = a_theta / b_theta  # Posterior mean of theta

    #     # Normalize to get topic proportions (optional)
    #     theta_norm = theta_est / jnp.sum(theta_est, axis=1, keepdims=True)

    #     # Predict topic index
    #     topic_idx = np.argmax(theta_norm, axis=1)

    #     # Return human-readable topic names
    #     return self._recode_topics(topic_idx), theta_est
