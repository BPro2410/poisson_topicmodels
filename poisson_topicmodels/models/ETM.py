from typing import Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import scipy.sparse as sparse
from jax import random
from numpyro import param, plate, sample
from numpyro.contrib.module import flax_module

# Abstract class - defining the minimum requirements for the probabilistic model
from .numpyro_model import NumpyroModel


class FlaxEncoder(nn.Module):
    """Neural network encoder for variational inference.

    Attributes
    ----------
    num_topics : int
        Number of topics K.
    hidden : int
        Hidden layer dimension.
    """

    num_topics: int
    hidden: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass through encoder.

        Parameters
        ----------
        inputs : jnp.ndarray
            Input data of shape (batch_size, V).

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            Mean and log-scale parameters for topic proportions.
        """
        h1 = nn.relu(nn.Dense(self.hidden)(inputs))
        h2 = nn.relu(nn.Dense(self.hidden)(h1))
        h21 = nn.Dense(self.num_topics)(h2)
        h22 = nn.Dense(self.num_topics)(h2)
        return h21, h22


# -- ETM class --
class ETM(NumpyroModel):
    """Embedded Topic Model (ETM).

    Learns topic representations in word embedding space using neural variational inference.
    Combines neural networks with Bayesian topic modeling for improved interpretability.

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
    embeddings_mapping : dict
        Mapping from words to embedding vectors.
    embed_size : int, optional
        Embedding dimension (default is 300).

    Attributes
    ----------
    D : int
        Number of documents.
    V : int
        Vocabulary size.
    K : int
        Number of topics.
    rho : np.ndarray
        Word embedding matrix of shape (V, embed_size).
    encoder : FlaxEncoder
        Neural encoder for variational inference.
    """

    def __init__(
        self,
        counts: sparse.csr_matrix,
        vocab: np.ndarray,
        num_topics: int,
        batch_size: int,
        embeddings_mapping: Dict,
        embed_size: int = 300,
    ) -> None:
        """Initialize the ETM model.

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
        embeddings_mapping : dict
            Word to embedding mapping.
        embed_size : int, optional
            Embedding dimension (default is 300).

        Raises
        ------
        TypeError
            If counts is not a sparse matrix.
        ValueError
            If dimensions are invalid or embeddings_mapping is empty.
        """
        super().__init__()

        # Input validation
        if not sparse.issparse(counts):
            raise TypeError(f"counts must be a scipy sparse matrix, got {type(counts).__name__}")

        D, V = counts.shape
        if D == 0 or V == 0:
            raise ValueError(f"counts matrix is empty: shape ({D}, {V})")

        if len(embeddings_mapping) == 0:
            raise ValueError("embeddings_mapping cannot be empty")

        if num_topics <= 0:
            raise ValueError(f"num_topics must be > 0, got {num_topics}")

        if batch_size <= 0 or batch_size > D:
            raise ValueError(f"batch_size must satisfy 0 < batch_size <= {D}, got {batch_size}")

        if embed_size <= 0:
            raise ValueError(f"embed_size must be > 0, got {embed_size}")

        # Store validated inputs
        self.counts = counts
        self.D = D
        self.K = num_topics
        self.V = V
        self.vocab = vocab
        self.batch_size = batch_size
        self.encoder = FlaxEncoder(num_topics=self.K, hidden=800)
        self.embeddings_mapping = embeddings_mapping
        self.embed_size = embed_size

        # Initialize word embeddings
        rho = np.zeros((self.V, embed_size))
        for i, word in enumerate(vocab):
            try:
                rho[i] = self.embeddings_mapping[word]
            except KeyError:
                rho[i] = np.random.normal(scale=0.6, size=(self.embed_size,))
        self.rho = rho

    def _model(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray) -> None:
        """Define the probabilistic generative model using NumPyro.

        Model structure:
        - Alpha: embedding projection to topic space
        - Theta (D x K): document-topic proportions
        - Beta (K x V): topic-word distributions
        - Y_batch: observed word counts with Poisson likelihood

        Parameters
        ----------
        Y_batch : jnp.ndarray
            Batch of observed word counts (batch_size, V).
        d_batch : jnp.ndarray
            Document indices in batch (batch_size,).
        """
        alpha = param(
            "alpha", init_value=random.normal(random.PRNGKey(42), shape=(self.embed_size, self.K))
        )
        with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
            with plate("d_k", size=self.K, dim=-1):
                theta = sample("theta", dist.Normal(0, 1))

            theta = jax.nn.softmax(theta, axis=1)

            beta = jnp.matmul(self.rho, alpha)
            beta = jnp.transpose(beta)
            beta = jax.nn.softmax(beta, axis=1)

            P = jnp.matmul(theta, beta)

            with plate("d_v", size=self.V, dim=-1):
                sample("Y_batch", dist.Poisson(P), obs=Y_batch)

    def _guide(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray) -> None:
        """Define the variational guide (approximate posterior).

        Uses neural encoder for amortized variational inference.

        Parameters
        ----------
        Y_batch : jnp.ndarray
            Batch of observed word counts (batch_size, V).
        d_batch : jnp.ndarray
            Document indices in batch (batch_size,).
        """
        net = flax_module("encoder", self.encoder, input_shape=(1, self.V))

        with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
            with plate("d_k", size=self.K, dim=-1):
                z_loc, z_std = net(Y_batch / (Y_batch.sum(axis=1).reshape(-1, 1)))
                sample("theta", dist.Normal(z_loc, z_std))

    # def get_batch(self, rng: jnp.ndarray, Y: sparse.csr_matrix) -> Tuple[jnp.ndarray, jnp.ndarray]:
    #     """Sample a random mini-batch from the corpus.

    #     Parameters
    #     ----------
    #     rng : jnp.ndarray
    #         JAX random key.
    #     Y : scipy.sparse.csr_matrix
    #         Document-term matrix.

    #     Returns
    #     -------
    #     Tuple[jnp.ndarray, jnp.ndarray]
    #         Y_batch : Word counts for the batch (batch_size, V).
    #         D_batch : Document indices in batch (batch_size,).

    #     Raises
    #     ------
    #     AssertionError
    #         If batch dimensions don't match expected shape.
    #     """
    #     D_batch = random.choice(rng, jnp.arange(self.D), shape=(self.batch_size,))
    #     Y_batch = jnp.array(Y[D_batch].toarray())
    #     # Ensure the shape of Y_batch is (batch_size, V)
    #     assert Y_batch.shape == (
    #         self.batch_size,
    #         self.V,
    #     ), f"Shape mismatch: {Y_batch.shape} != ({self.batch_size}, {self.V})"

    #     return Y_batch, D_batch

    def return_topics(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Extract learned topics from model parameters.

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            Topics and document-topic proportions.

        Raises
        ------
        NotImplementedError
            This method requires using fitted neural network parameters.
        """
        raise NotImplementedError("Use the fitted parameter of the NN to extract topics.")

    def return_beta(self) -> jnp.ndarray:
        """Extract topic-word distribution matrix.

        Returns
        -------
        jnp.ndarray
            Topic-word distribution of shape (K, V).

        Raises
        ------
        NotImplementedError
            This method is to be implemented.
        """
        raise NotImplementedError("To be implemented.")
