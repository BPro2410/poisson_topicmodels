import jax
from jax import random, jit
import jax.numpy as jnp
from numpyro import param, plate, sample
import numpyro.distributions as dist
from numpyro.distributions import constraints
import jax.nn as jnn
import numpy as np
import pandas as pd


# Abstract class - defining the minimum requirements for the probabilistic model
from packages.models.numpyro_model import NumpyroModel

# Create numpyro model
class CSPF(NumpyroModel):
    """
    Implementation of the Seeded Poisson Factorization (SPF) topic model including covariates.

    CSPF allows to fit a guided topic model and estimate covariate effects that drive topic polarization.
    """

    def __init__(self, counts, vocab, keywords, residual_topics, batch_size, X_design_matrix):
        """
        Initialize the CSPF model.
        
        Parameters
        ----------
        counts : numpy.ndarray
            A 2D array representing the word counts in each document.
        vocab : list
            A list of vocabulary terms.
        keywords : dict
            A dictionary where keys are topics and values are lists of keywords.
        residual_topics : int
            The number of residual topics.
        batch_size : int
            The number of documents to be processed in each batch.
        X_design_matrix : pandas.DataFrame
            Design matrix for covariates.
        """
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
        self.keywords = keywords
        self.residual_topics = residual_topics
        self.covariates = list(X_design_matrix.columns)

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

        with plate("k", size = self.K, dim = -2):
            with plate("k_v", size = self.V, dim = -1):
                beta = sample("beta", dist.Gamma(.3,.3))

        with plate("tilde_v", size = self.Tilde_V):
            beta_tilde = sample("beta_tilde", dist.Gamma(1., .3))

        # Update beta
        beta = beta.at[self.kw_indices].add(beta_tilde)

        # hier covariates
        with plate("c", size = self.C):
            zeta = sample("zeta", dist.Normal(0., 1.))
            rho = sample("rho", dist.Gamma(.3, 1.))
            omega = sample("omega", dist.Gamma(.3, rho))

        with plate("c", size = self.C, dim = -2):
            with plate("c_k", size = self.K, dim = -1):
                lambda_ = sample("lambda", dist.Normal(jnp.tile(zeta, (self.K, 1)).T, jnp.tile(omega, (self.K, 1)).T))

        a_theta_S = jax.nn.softplus(jnp.matmul(self.X_design_matrix, lambda_))[d_batch]

        # ACHTUNG: Hier haben wir eine subsample_size wegen der batch size!
        with plate("d", size = self.D, subsample_size = self.batch_size, dim = -2):
            with plate("d_k", size = self.K, dim = -1):
                theta = sample("theta", dist.Gamma(a_theta_S, .3))

            # ACHTUNG: P = muss eingerückt sein in der d plate - ich weiß aber nicht genau warum das so ist
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

        a_beta_tilde = param("beta_tilde_shape", init_value = jnp.ones([self.Tilde_V]) * 2, constraint=constraints.positive)
        b_beta_tilde = param("beta_tilde_rate", init_value = jnp.ones([self.Tilde_V]), constraint=constraints.positive)

        location_lambda = param("lambda_location", init_value = jnp.zeros([self.C, self.K]))
        scale_lambda = param("lambda_scale", init_value = jnp.ones([self.C, self.K]), constraint=constraints.positive)

        location_zeta = param("zeta_location", init_value = jnp.zeros(self.C))
        scale_zeta = param("zeta_scale", init_value = jnp.ones(self.C), constraint =constraints.positive)

        a_rho = param("rho_shape", init_value = jnp.ones(self.C), constraint = constraints.positive)
        b_rho = param("rho_rate", init_value = jnp.ones(self.C), constraint = constraints.positive)

        a_omega = param("omega_shape", init_value = jnp.ones(self.C), constraint = constraints.positive)
        b_omega = param("omega_rate", init_value = jnp.ones(self.C), constraint = constraints.positive)


        with plate("k", size = self.K, dim = -2):
            with plate("k_v", size = self.V, dim = -1):
                beta = sample("beta", dist.Gamma(a_beta, b_beta))

        with plate("tilde_v", size = self.Tilde_V):
            beta_tilde = sample("beta_tilde", dist.Gamma(a_beta_tilde, b_beta_tilde))

        with plate("c", size = self.C):
            zeta = sample("zeta", dist.Normal(location_zeta, scale_zeta))
            rho = sample("rho", dist.Gamma(a_rho, b_rho))
            omega = sample("omega", dist.Gamma(a_omega, b_omega))

        with plate("c", size = self.C, dim = -2):
            with plate("c_k", size = self.K, dim = -1):
                lambda_ = sample("lambda", dist.Normal(location_lambda, scale_lambda))

        with plate("d", size = self.D, subsample_size=self.batch_size, dim = -2):
            with plate("d_k", size = self.K, dim = -1):
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

            topics = np.where(argmaxes <= max_index,
                    keyword_keys[argmaxes_clipped],
                    f"No_keyword_topic_{argmaxes - max_index}")
            
            return topics
        
        E_theta = self.estimated_params["theta_shape"] / self.estimated_params["theta_rate"]

        categories = np.argmax(E_theta, axis = 1)
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
        E_beta_tilde = self.estimated_params["beta_tilde_shape"] / self.estimated_params["beta_tilde_rate"]
        E_beta = E_beta.at[self.kw_indices].add(E_beta_tilde)

        rs_names = [f"residual_topic_{i+1}" for i in range(self.residual_topics)]

        return pd.DataFrame(jnp.transpose(E_beta), index = self.vocab, columns = list(self.keywords.keys()) + rs_names)


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
        return pd.DataFrame(self.estimated_params["lambda_location"], index = index, columns = cols)
       