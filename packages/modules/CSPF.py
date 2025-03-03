import jax
from jax import random, jit
import jax.numpy as jnp
from numpyro import param, plate, sample
import numpyro.distributions as dist
from numpyro.distributions import constraints
import jax.nn as jnn

# Abstract class - defining the minimum requirements for the probabilistic model
from packages.modules.numpyro_model import NumpyroModel

# Create numpyro model
class CSPF(NumpyroModel):
    def __init__(self, counts, vocab, keywords, residual_topics, batch_size, X_design_matrix):
        self.counts = counts
        self.D = counts.shape[0]
        self.V = counts.shape[1]
        self.vocab = vocab
        self.K = residual_topics + len(keywords.keys())
        kw_indices_topics = [(idx, list(vocab).index(keyword)) for idx, topic in enumerate(keywords.keys()) for keyword in keywords[topic] if keyword in vocab]
        self.Tilde_V = len(kw_indices_topics)
        self.kw_indices = tuple(zip(*kw_indices_topics))
        self.batch_size = batch_size
        self.X_design_matrix = jnp.array(X_design_matrix)
        self.C = self.X_design_matrix.shape[1]

    # -- Model --
    def _model(self, Y_batch, d_batch):

        # Topic distributions
        with plate("k", size = self.K, dim = -2):
            with plate("k_v", size = self.V, dim = -1):
                beta = sample("beta", dist.Gamma(.3,.3))

        with plate("tilde_v", size = self.Tilde_V):
            beta_tilde = sample("beta_tilde", dist.Gamma(5., .3))

        # Update beta
        beta = beta.at[self.kw_indices].add(beta_tilde)

        # Covariates
        with plate("c", size = self.C, dim = -2):
            with plate("c_k", size = self.K, dim = -1):
                phi = sample("phi", dist.Normal(0., 1.))
        # Transform a_theta
        a_theta_S = jnn.softplus(jnp.matmul(self.X_design_matrix, phi))[d_batch]

        # Document distribution
        with plate("d", size = self.D, subsample_size = self.batch_size, dim = -2):
            with plate("d_k", size = self.K, dim = -1):
                theta = sample("theta", dist.Gamma(a_theta_S, .3))

            # Poisson rate
            P = jnp.matmul(theta, beta)

            with plate("d_v", size = self.V, dim = -1):
                sample("Y_batch", dist.Poisson(P), obs = Y_batch)

    # -- Guide, i.e. variational family --
    def _guide(self, Y_batch, d_batch):

        # Define variational parameter
        a_beta = param("beta_shape", init_value = jnp.ones([self.K, self.V]), constraint=constraints.positive)
        b_beta = param("beta_rate", init_value = jnp.ones([self.K, self.V]) * self.D / 1000 * 2, constraint=constraints.positive)

        a_theta = param("theta_shape", init_value = jnp.ones([self.D, self.K]), constraint=constraints.positive)
        b_theta = param("theta_rate", init_value = jnp.ones([self.D, self.K]) * self.D / 1000, constraint=constraints.positive)

        a_beta_tilde = param("beta_tilde_shape", init_value = jnp.ones([self.Tilde_V]) * 2, constraint=constraints.positive)
        b_beta_tilde = param("beta_tilde_rate", init_value = jnp.ones([self.Tilde_V]), constraint=constraints.positive)

        mu_phi = param("phi_mittelwert", init_value = jnp.zeros([self.C, self.K]))
        sigma_phi = param("phi_sigma", init_value = jnp.ones([self.C, self.K]), constraint=constraints.positive)

        with plate("k", size = self.K, dim = -2):
            with plate("k_v", size = self.V, dim = -1):
                beta = sample("beta", dist.Gamma(a_beta, b_beta))

        with plate("tilde_v", size = self.Tilde_V):
            beta_tilde = sample("beta_tilde", dist.Gamma(a_beta_tilde, b_beta_tilde))

        with plate("c", size = self.C, dim = -2):
            with plate("c_k", size = self.K, dim = -1):
                phi = sample("phi", dist.Normal(mu_phi, sigma_phi))

        with plate("d", size = self.D, subsample_size=self.batch_size, dim = -2):
            with plate("d_k", size = self.K, dim = -1):
                theta = sample("theta", dist.Gamma(a_theta[d_batch], b_theta[d_batch]))

