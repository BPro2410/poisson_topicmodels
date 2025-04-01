import jax
from jax import random, jit
import jax.numpy as jnp
from numpyro import param, plate, sample
import numpyro.distributions as dist
from numpyro.distributions import constraints
import jax.nn as jnn

# Abstract class - defining the minimum requirements for the probabilistic model
from packages.models.numpyro_model import NumpyroModel

# Create numpyro model
class CPF(NumpyroModel):
    """
    Implementation of the standard Poisson factorization topic model including covariates.
    """

    def __init__(self, counts, vocab, num_topics, batch_size, X_design_matrix):
        """
        Initialize the CPF model.
        
        Parameters
        ----------
        counts : numpy.ndarray
            A 2D array representing the word counts in each document.
        vocab : list
            A list of vocabulary terms.
        num_topics : int
            The number of topics.
        batch_size : int
            The number of documents to be processed in each batch.
        X_design_matrix : pandas.DataFrame
            Design matrix for covariates.
        """
        self.counts = counts
        self.D = counts.shape[0]
        self.V = counts.shape[1]
        self.vocab = vocab
        self.K = num_topics
        self.batch_size = batch_size
        self.X_design_matrix = jnp.array(X_design_matrix)
        self.C = self.X_design_matrix.shape[1]

    # -- Model --
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

        # Topic distributions
        with plate("k", size = self.K, dim = -2):
            with plate("k_v", size = self.V, dim = -1):
                beta = sample("beta", dist.Gamma(.3,.3))

        # Covariates
        with plate("c", size = self.C, dim = -2):
            with plate("c_k", size = self.K, dim = -1):
                lambda_ = sample("phi", dist.Normal(0., 1.))
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
        a_beta = param("beta_shape", init_value = jnp.ones([self.K, self.V]), constraint=constraints.positive)
        b_beta = param("beta_rate", init_value = jnp.ones([self.K, self.V]) * self.D / 1000 * 2, constraint=constraints.positive)

        a_theta = param("theta_shape", init_value = jnp.ones([self.D, self.K]), constraint=constraints.positive)
        b_theta = param("theta_rate", init_value = jnp.ones([self.D, self.K]) * self.D / 1000, constraint=constraints.positive)

        location_lambda = param("lambda_location", init_value = jnp.zeros([self.C, self.K]))
        scale_lambda = param("lambda_scale", init_value = jnp.ones([self.C, self.K]), constraint=constraints.positive)

        with plate("k", size = self.K, dim = -2):
            with plate("k_v", size = self.V, dim = -1):
                beta = sample("beta", dist.Gamma(a_beta, b_beta))
                
        with plate("c", size = self.C, dim = -2):
            with plate("c_k", size = self.K, dim = -1):
                lambda_ = sample("phi", dist.Normal(location_lambda, scale_lambda))

        with plate("d", size = self.D, subsample_size=self.batch_size, dim = -2):
            with plate("d_k", size = self.K, dim = -1):
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
        return pd.DataFrame(self.estimated_params["lambda_location"], index = index, columns = rs_names)
       

