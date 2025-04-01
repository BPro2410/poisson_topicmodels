import jax
from jax import random, jit
import jax.numpy as jnp
from numpyro import param, plate, sample
import numpyro.distributions as dist
from numpyro.distributions import constraints
# from optax import adam
# from numpyro.infer import SVI, TraceMeanField_ELBO
# from tqdm import tqdm

# Abstract class - defining the minimum requirements for the probabilistic model
from packages.models.numpyro_model import NumpyroModel

class PF(NumpyroModel):
    """
    Documentation of the Poisson factorization (PF) topic model.
    PF is an unsupervised topic model.
    """
    
    def __init__(self, counts, vocab, num_topics, batch_size):
        """
        Initialize the PF model.
        
        Parameters
        ----------
        counts : scipy.sparse.csr_matrix
            A 2D sparse array representing the word counts in each document.
        vocab : list
            A list of vocabulary terms.
        num_topics : int
            The number of topics (K).
        batch_size : int
            The number of documents to be processed in each batch.
        """

        self.counts = counts
        self.V = counts.shape[1]
        self.D = counts.shape[0]
        self.vocab = vocab
        # assert counts.shape1 == len(vocab)
        self.K = num_topics
        self.batch_size = batch_size

    def _model(self, Y_batch, d_batch):
        """
        Define the probabilistic model using NumPyro.
        
        Parameters
        ----------
        Y_batch : numpy.ndarray
            The observed word counts for the current batch.
        d_batch : numpy.ndarray
            Indices of documents in the current batch.
        """
        # Topic level
        with plate("k", size = self.K, dim = -2):
            with plate("k_v", size = self.V, dim = -1):
                beta = sample("beta", dist.Gamma(.3,.3))

        # Document level
        with plate("d", size = self.D, subsample_size = self.batch_size, dim = -2):
            with plate("d_k", size = self.K, dim = -1):
                theta = sample("theta", dist.Gamma(.3, .3))

            # Calculate Poisson rates
            P = jnp.matmul(theta, beta)

            # Reconstruction
            with plate("v", size = self.V, dim = -1):
                sample("Y_batch", dist.Poisson(P), obs = Y_batch)

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
        a_beta = param("beta_shape", init_value = jnp.ones([self.K, self.V]), constraint=constraints.positive)
        b_beta = param("beta_rate", init_value = jnp.ones([self.K, self.V]) * self.D / 1000 * 2, constraint=constraints.positive)
        a_theta = param("theta_shape", init_value = jnp.ones([self.D, self.K]), constraint=constraints.positive)
        b_theta = param("theta_rate", init_value = jnp.ones([self.D, self.K]) * self.D / 1000, constraint=constraints.positive)

        with plate("k", size = self.K, dim = -2):
            with plate("k_v", size = self.V, dim = -1):
                sample("beta", dist.Gamma(a_beta, b_beta))

        with plate("d", size = self.D, subsample_size=self.batch_size, dim = -2):
            with plate("d_k", size = self.K, dim = -1):
                sample("theta", dist.Gamma(a_theta[d_batch], b_theta[d_batch]))

  