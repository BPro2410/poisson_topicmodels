from jax import random
import numpy as np
from numpyro import param, plate, sample
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
import flax.linen as nn
from numpyro.contrib.module import flax_module

# Abstract class - defining the minimum requirements for the probabilistic model
from packages.models.numpyro_model import NumpyroModel


# --- Define encoder ---
class FlaxEncoder(nn.Module):
    num_topics: int
    hidden: int

    @nn.compact
    def __call__(self, inputs):
        h1 = nn.relu(nn.Dense(self.hidden)(inputs))
        h2 = nn.relu(nn.Dense(self.hidden)(h1))
        h21 = nn.Dense(self.num_topics)(h2)
        h22 = nn.Dense(self.num_topics)(h2)
        return h21, h22


# -- ETM class --
class ETM(NumpyroModel):
    def __init__(self, counts, vocab, num_topics, batch_size, embeddings_mapping, embed_size = 300):
        self.counts = counts
        self.D = counts.shape[0]
        self.K = num_topics
        self.V = len(vocab)
        self.vocab = vocab
        self.batch_size = batch_size
        self.encoder = FlaxEncoder(num_topics = self.K, hidden = 800)
        self.embeddings_mapping = embeddings_mapping
        self.embed_size = embed_size

        rho = np.zeros((self.V, embed_size))

        for i, word in enumerate(vocab):
            try:
                rho[i] = self.embeddings_mapping[word]
            except KeyError:
                rho[i] = np.random.normal(
                    scale=0.6, size=(self.embed_size,)
                )
        self.rho = rho

    def _model(self, Y_batch, d_batch):

        alpha = param("alpha", init_value = random.normal(random.PRNGKey(42), shape = (self.embed_size, self.K)))
        with plate("d", size = self.D, subsample_size = self.batch_size, dim = -2):
            with plate("d_k", size = self.K, dim = -1):
                theta = sample("theta", dist.Normal(0,1))
            
            theta = jax.nn.softmax(theta, axis = 1)
            
            beta = jnp.matmul(self.rho, alpha)
            beta = jnp.transpose(beta)
            beta = jax.nn.softmax(beta, axis = 1)

            P = jnp.matmul(theta, beta)

            with plate("d_v", size = self.V, dim = -1):
                sample("Y_batch", dist.Poisson(P), obs = Y_batch)

    def _guide(self, Y_batch, d_batch):

        net = flax_module("encoder", self.encoder, input_shape = (1, self.V))

        with plate("d", size = self.D, subsample_size = self.batch_size, dim = -2):
            with plate("d_k", size = self.K, dim = -1):
                z_loc, z_std = net(Y_batch / (Y_batch.sum(axis = 1).reshape(-1,1)))
                theta = sample("theta", dist.Normal(z_loc, z_std))

    def get_batch(self, rng, Y):
        D_batch = random.choice(rng, jnp.arange(self.D), shape = (self.batch_size,))
        Y_batch = jnp.array(Y[D_batch].toarray())
        # Ensure the shape of Y_batch is (batch_size, V)
        assert Y_batch.shape == (self.batch_size, self.V), f"Shape mismatch: {Y_batch.shape} != ({self.batch_size}, {self.V})"

        return Y_batch, D_batch

    def return_topics(self):
        raise NotImplementedError("Use the fitted parameter of the NN to extract topics.")
    
    def return_beta(self):
        raise NotImplementedError("To be implemented.")
