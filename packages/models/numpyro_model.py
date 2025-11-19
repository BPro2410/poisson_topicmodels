from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from jax import jit, random
from numpyro.infer import SVI, TraceMeanField_ELBO
from optax import adam
from tqdm import tqdm

from packages.models.Metrics import Metrics


class NumpyroModel(ABC):
    """
    Abstract base class for all used probabilistic models.
    Each model has to implement at least their own Model and Guide.

    Attributes
    ----------
    counts : scipy.sparse.csr_matrix
        Document-term matrix.
    D : int
        Number of documents.
    V : int
        Size of vocabulary.
    vocab : np.ndarray
        Vocabulary array.
    K : int
        Number of topics.
    batch_size : int
        Size of mini-batches for training.
    Metrics : Metrics
        Instance metrics tracker (per instance, not shared).
    estimated_params : dict
        Estimated parameters after training.
    """

    def __init__(self) -> None:
        """Initialize base model with per-instance metrics."""
        self.Metrics = Metrics(loss=[])
        self.estimated_params: Dict[str, Any] = {}

    @abstractmethod
    def _model(self) -> None:
        """Define the probabilistic model."""
        pass

    @abstractmethod
    def _guide(self) -> None:
        """Define the variational guide."""
        pass

    def _get_batch(
        self, rng: jax.Array, Y: sparse.csr_matrix
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Helper function to obtain a batch of data, convert from scipy.sparse to jax.numpy.array.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            Random number generator key.
        Y : scipy.sparse.csr_matrix
            The word counts array.

        Returns
        -------
        tuple
            Y_batch : numpy.ndarray
                Word counts for the batch.
            D_batch : numpy.ndarray
                Indices of documents in the batch.
        """
        D_batch = random.choice(rng, jnp.arange(self.D), shape=(self.batch_size,))
        # Y_batch = jax.device_put(jnp.array(Y[D_batch].toarray()), jax.devices("cpu")[0])
        Y_batch = jnp.array(Y[D_batch].toarray())

        # Ensure the shape of Y_batch is (batch_size, V)
        assert Y_batch.shape == (
            self.batch_size,
            self.V,
        ), f"Shape mismatch: {Y_batch.shape} != ({self.batch_size}, {self.V})"

        return Y_batch, D_batch

    def train_step(
        self,
        num_steps: int,
        lr: float,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Train the model using Stochastic Variational Inference (SVI).

        Parameters
        ----------
        num_steps : int
            Number of training iterations. Must be > 0.
        lr : float
            Learning rate for the optimizer. Must be > 0.
        random_seed : int, optional
            Seed for JAX random number generator. If provided, ensures
            reproducible results. Default is None (random initialization).

        Returns
        -------
        dict
            Estimated parameters after training.

        Raises
        ------
        ValueError
            If num_steps <= 0 or lr <= 0.
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {num_steps}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")

        svi_batch = SVI(
            model=self._model, guide=self._guide, optim=adam(lr), loss=TraceMeanField_ELBO()
        )
        svi_batch_update = jit(svi_batch.update)

        # Initialize RNG
        if random_seed is not None:
            init_rng = jax.random.PRNGKey(random_seed)
        else:
            init_rng = jax.random.PRNGKey(0)

        Y_batch, D_batch = self._get_batch(init_rng, self.counts)

        svi_state = svi_batch.init(jax.random.PRNGKey(1), Y_batch=Y_batch, d_batch=D_batch)

        rngs = random.split(jax.random.PRNGKey(2), num_steps)
        pbar = tqdm(range(num_steps))

        for step in pbar:
            Y_batch, D_batch = self._get_batch(rngs[step], self.counts)
            svi_state, loss = svi_batch_update(svi_state, Y_batch=Y_batch, d_batch=D_batch)
            loss = loss / self.D
            self.Metrics.loss.append(loss)
            if step % 10 == 0:
                pbar.set_description(
                    "Init loss: "
                    + "{:10.4f}".format(self.Metrics.loss[0])
                    + f"; Avg loss (last 10 iter): "
                    + "{:10.4f}".format(jnp.array(self.Metrics.loss[-10:]).mean())
                )

        self.estimated_params = svi_batch.get_params(svi_state)

        return self.estimated_params

    def return_topics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the topics for each document.

        Returns
        -------
        categories : np.ndarray
            Array of topic indices for each document (shape: D,).
        E_theta : np.ndarray
            Estimated topic proportions for each document (shape: D, K).

        Raises
        ------
        ValueError
            If model has not been trained yet (no estimated parameters).
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling return_topics()")

        E_theta = self.estimated_params["theta_shape"] / self.estimated_params["theta_rate"]
        return np.argmax(E_theta, axis=1), E_theta

    def return_beta(self) -> pd.DataFrame:
        """
        Return the beta matrix (word-topic associations) for the model.

        Returns
        -------
        pd.DataFrame
            DataFrame with words as index and topics as columns,
            containing word-topic probability estimates.

        Raises
        ------
        ValueError
            If model has not been trained yet (no estimated parameters).
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling return_beta()")

        E_beta = self.estimated_params["beta_shape"] / self.estimated_params["beta_rate"]
        return pd.DataFrame(jnp.transpose(E_beta), index=self.vocab)
