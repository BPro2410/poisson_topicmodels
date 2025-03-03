import jax
from jax import random, jit
import jax.numpy as jnp
from numpyro import param, plate, sample
import numpyro.distributions as dist
from numpyro.distributions import constraints
from optax import adam
from numpyro.infer import SVI, TraceMeanField_ELBO
from tqdm import tqdm

# Abstract class - defining the minimum requirements for the probabilistic model
from packages.modules.numpyro_model import NumpyroModel

class SPF(NumpyroModel):
    """
    Documentation of SPF
    """
    
    def __init__(self, counts, vocab, keywords, residual_topics, batch_size):

        self.counts = counts
        self.V = counts.shape[1]
        self.D = counts.shape[0]
        self.vocab = vocab
        # assert counts.shape1 == len(vocab)
        self.K = residual_topics + len(keywords.keys())
        kw_indices_topics = [(idx, list(vocab).index(keyword)) for idx, topic in enumerate(keywords.keys()) for keyword in keywords[topic] if keyword in vocab]
        self.Tilde_V = len(kw_indices_topics)
        self.kw_indices = tuple(zip(*kw_indices_topics))
        self.batch_size = batch_size

    # -- MODEL --
    def _model(self, Y_batch, d_batch):

        # Topic level
        with plate("k", size = self.K, dim = -2):
            with plate("k_v", size = self.V, dim = -1):
                beta = sample("beta", dist.Gamma(.3,.3))

        with plate("tilde_v", size = self.Tilde_V):
            beta_tilde = sample("beta_tilde", dist.Gamma(1., .3))

        # Update beta - vectorized jax operation prefered over loops
        beta = beta.at[self.kw_indices].add(beta_tilde)

        # Document level
        with plate("d", size = self.D, subsample_size = self.batch_size, dim = -2):
            with plate("d_k", size = self.K, dim = -1):
                theta = sample("theta", dist.Gamma(.3, .3))

            # Calculate Poisson rates
            P = jnp.matmul(theta, beta)

            # Reconstruction
            with plate("v", size = self.V, dim = -1):
                sample("Y_batch", dist.Poisson(P), obs = Y_batch)

    # -- GUIDE --
    def _guide(self, Y_batch, d_batch):

        # Define variational parameter
        a_beta = param("beta_shape", init_value = jnp.ones([self.K, self.V]), constraint=constraints.positive)
        b_beta = param("beta_rate", init_value = jnp.ones([self.K, self.V]) * self.D / 1000 * 2, constraint=constraints.positive)
        a_theta = param("theta_shape", init_value = jnp.ones([self.D, self.K]), constraint=constraints.positive)
        b_theta = param("theta_rate", init_value = jnp.ones([self.D, self.K]) * self.D / 1000, constraint=constraints.positive)
        a_beta_tilde = param("beta_tilde_shape", init_value = jnp.ones([self.Tilde_V]) * 2, constraint=constraints.positive)
        b_beta_tilde = param("beta_tilde_rate", init_value = jnp.ones([self.Tilde_V]), constraint=constraints.positive)

        with plate("k", size = self.K, dim = -2):
            with plate("k_v", size = self.V, dim = -1):
                sample("beta", dist.Gamma(a_beta, b_beta))

        with plate("tilde_v", size = self.Tilde_V):
            sample("beta_tilde", dist.Gamma(a_beta_tilde, b_beta_tilde))

        with plate("d", size = self.D, subsample_size=self.batch_size, dim = -2):
            with plate("d_k", size = self.K, dim = -1):
                sample("theta", dist.Gamma(a_theta[d_batch], b_theta[d_batch]))

  