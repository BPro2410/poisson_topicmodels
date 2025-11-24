import warnings
from typing import Tuple

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
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

    def _model(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray, i_batch: jnp.ndarray) -> None:
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

    def _guide(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray, i_batch: jnp.ndarray) -> None:
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
        sigma_x = param("sigma_y", init_value=jnp.ones([self.N]), constraint=constraints.positive)

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
        if self.beta_shape_init is not None and self.time_varying == True:
            mu_beta = param(
                "mu_beta",
                init_value=self.beta_shape_init,
            )
        else:
            mu_beta = param("mu_beta", init_value=jnp.zeros([self.K, self.V]))

        # check if beta_shape init is not none and self.time_yvarying is true
        if self.beta_shape_init is not None and self.time_varying == True:
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

    def train_step(self, num_steps: int, lr: float) -> dict:
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
            self.Metrics.loss.append(loss)
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
