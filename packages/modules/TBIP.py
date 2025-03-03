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

# Abstract class - defining the minimum requirements for the probabilistic model
from packages.modules.numpyro_model import NumpyroModel


class TBIP(NumpyroModel):
    def __init__(self, counts, vocab, num_topics, authors, batch_size):
        self.authors_unique = np.unique(authors)
        self.author_map = {speaker: idx for idx, speaker in enumerate(self.authors_unique)}
        self.author_indices = authors.map(self.author_map)
        self.N = len(self.authors_unique) # number of people
        self.counts = counts
        self.D = counts.shape[0]
        self.V = counts.shape[1]
        self.K = num_topics
        self.batch_size = batch_size  # number of documents in a batch

    def _model(self, Y_batch, d_batch, i_batch):
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
                * jnp.exp(
                    jnp.expand_dims(x[i_batch], (1, 2)) * jnp.expand_dims(eta, 0)
                ),
                1,
            )

            with plate("v", size=self.V, dim=-1):
                # Sample observed words
                sample("Y_batch", dist.Poisson(P), obs=Y_batch)

    def _guide(self, Y_batch, d_batch, i_batch):
        # This defines variational family. Notice that each of the latent variables
        # defined in the sample statements in the model above has a corresponding
        # sample statement in the guide. The guide is responsible for providing
        # variational parameters for each of these latent variables.

        # Also notice it is required that model and the guide have the same call.

        mu_x = param(
            "mu_x", init_value=-1 + 2 * random.uniform(random.PRNGKey(1), (self.N,))
        )
        sigma_x = param(
            "sigma_y", init_value=jnp.ones([self.N]), constraint=constraints.positive
        )

        mu_eta = param(
            "mu_eta", init_value=random.normal(random.PRNGKey(2), (self.K, self.V))
        )
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

        mu_beta = param("mu_beta", init_value=jnp.zeros([self.K, self.V]))
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
        
    def _get_batch(self, rng, Y):
        # Helper functions to obtain a batch of data, convert from scipy.sparse
        # to jax.numpy.array and move to gpu

        D_batch = random.choice(rng, jnp.arange(self.D), shape=(self.batch_size,))
        Y_batch = jnp.array(Y[D_batch].toarray())

        # Ensure the shape of Y_batch is (batch_size, V)
        assert Y_batch.shape == (self.batch_size, self.V), f"Shape mismatch: {Y_batch.shape} != ({self.batch_size}, {self.V})"

        I_batch = np.array(self.author_indices[D_batch])
        return Y_batch, D_batch, I_batch
    

    def train_step(self, num_steps, lr):
        """
        Train the model.
        """

        svi_batch = SVI(
            model = self._model,
            guide = self._guide,
            optim = adam(lr),
            loss = TraceMeanField_ELBO()
        )
        svi_batch_update = jit(svi_batch.update)

        Y_batch, D_batch, I_batch = self._get_batch(random.PRNGKey(1), self.counts)

        svi_state = svi_batch.init(
            random.PRNGKey(0), Y_batch = Y_batch, d_batch = D_batch, i_batch = I_batch
        )
    
        rngs = random.split(random.PRNGKey(2), num_steps)
        # losses = list()
        pbar = tqdm(range(num_steps))

        for step in pbar:
            Y_batch, D_batch, I_batch = self._get_batch(rngs[step], self.counts)
            svi_state, loss = svi_batch_update(
                svi_state, Y_batch=Y_batch, d_batch = D_batch, i_batch = I_batch
            )
            loss = loss / self.D
            self.Metrics.loss.append(loss)
            # losses.append(loss)
            if step % 10 == 0:
                pbar.set_description(
                    "Init loss: "
                    + "{:10.4f}".format(self.Metrics.loss[0])
                    + f"; Avg loss (last {10} iter): "
                    + "{:10.4f}".format(jnp.array(self.Metrics.loss[-10:]).mean()))
        
        return svi_batch, svi_state
