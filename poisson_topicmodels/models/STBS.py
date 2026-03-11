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
from optax import adam
from tqdm import tqdm

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
                X_design_matrix = X_design_matrix.values

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
        self.covariates = (
            list(X_design_matrix.columns)
            if isinstance(X_design_matrix, pd.DataFrame)
            else [f"cov_{i}" for i in range(self.L)]
        )

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

