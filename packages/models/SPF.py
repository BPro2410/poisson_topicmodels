import jax
from jax import random, jit
import jax.numpy as jnp
from numpyro import param, plate, sample
import numpyro.distributions as dist
from numpyro.distributions import constraints
from optax import adam
from numpyro.infer import SVI, TraceMeanField_ELBO
from tqdm import tqdm
import numpy as np
import pandas as pd

# Abstract class - defining the minimum requirements for the probabilistic model
from packages.models.numpyro_model import NumpyroModel

class SPF(NumpyroModel):
    """
    Implementation of the Seeded Poisson Factorization (SPF) topic model.
    
    SPF allows to fit guided topics by increasing the topical prevalence of seed words a-priori.
    """
    
    def __init__(self, counts, vocab, keywords, residual_topics, batch_size):
        """
        Initialize the SPF model.
        
        Parameters
        ----------
        counts : scipy.sparse.csr_matrix
            A 2D sparse array representing the word counts in each document.
        vocab : list
            A list of vocabulary terms.
        keywords : dict
            A dictionary of keywords for each topic.
        residual_topics : int
            The number of residual topics.
        batch_size : int
            The number of documents to be processed in each batch.
        """

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
        self.keywords = keywords
        self.residual_topics = residual_topics

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

        with plate("k", size = self.K, dim = -2):
            with plate("k_v", size = self.V, dim = -1):
                sample("beta", dist.Gamma(a_beta, b_beta))

        with plate("tilde_v", size = self.Tilde_V):
            sample("beta_tilde", dist.Gamma(a_beta_tilde, b_beta_tilde))

        with plate("d", size = self.D, subsample_size=self.batch_size, dim = -2):
            with plate("d_k", size = self.K, dim = -1):
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

            topics = np.where(argmaxes <= max_index,
                    keyword_keys[argmaxes_clipped],
                    "No_keyword_topic_" + (argmaxes - max_index).astype(str))
            
            return topics
        
        E_theta = self.estimated_params["theta_shape"] / self.estimated_params["theta_rate"]

        categories = np.argmax(E_theta, axis = 1)
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
        E_beta_tilde = self.estimated_params["beta_tilde_shape"] / self.estimated_params["beta_tilde_rate"]
        E_beta = E_beta.at[self.kw_indices].add(E_beta_tilde)

        rs_names = [f"residual_topic_{i+1}" for i in range(self.residual_topics)]

        return pd.DataFrame(jnp.transpose(E_beta), index = self.vocab, columns = list(self.keywords.keys()) + rs_names)
