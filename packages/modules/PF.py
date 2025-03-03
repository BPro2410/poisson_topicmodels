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
from packages.modules.numpyro_model import NumpyroModel

class PF(NumpyroModel):
    """
    Documentation of PF
    """
    
    def __init__(self, counts, vocab, num_topics, batch_size):

        self.counts = counts
        self.V = counts.shape[1]
        self.D = counts.shape[0]
        self.vocab = vocab
        # assert counts.shape1 == len(vocab)
        self.K = num_topics
        self.batch_size = batch_size

    # -- MODEL --
    def _model(self, Y_batch, d_batch):

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

    # -- GUIDE --
    def _guide(self, Y_batch, d_batch):

        # Define variational parameter
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

    # def get_batch(self, rng, Y):
    #     D_batch = random.choice(rng, jnp.arange(self.D), shape = (self.batch_size,))
    #     # Y_batch = jax.device_put(jnp.array(Y[D_batch].toarray()), jax.devices("cpu")[0])
    #     Y_batch = jnp.array(Y[D_batch].toarray())

    #     # Ensure the shape of Y_batch is (batch_size, V)
    #     assert Y_batch.shape == (self.batch_size, self.V), f"Shape mismatch: {Y_batch.shape} != ({self.batch_size}, {self.V})"

    #     return Y_batch, D_batch
    

    # def train_step(self, num_steps, lr):
    #     """
    #     Train the model.
    #     """

    #     svi_batch = SVI(
    #         model = self.model,
    #         guide = self.guide,
    #         optim = adam(lr),
    #         loss = TraceMeanField_ELBO()
    #     )
    #     svi_batch_update = jit(svi_batch.update)

    #     Y_batch, D_batch = self.get_batch(random.PRNGKey(1), self.counts)

    #     svi_state = svi_batch.init(
    #         random.PRNGKey(0), Y_batch = Y_batch, d_batch = D_batch
    #     )
    
    #     rngs = random.split(random.PRNGKey(2), num_steps)
    #     losses = list()
    #     pbar = tqdm(range(num_steps))

    #     for step in pbar:
    #         Y_batch, D_batch = self.get_batch(rngs[step], self.counts)
    #         svi_state, loss = svi_batch_update(
    #             svi_state, Y_batch=Y_batch, d_batch = D_batch
    #         )
    #         loss = loss / self.D
    #         losses.append(loss)
    #         if step % 10 == 0:
    #             pbar.set_description(
    #                 "Init loss: "
    #                 + "{:10.4f}".format(jnp.array(losses[0]))
    #                 + f"; Avg loss (last {10} iter): "
    #                 + "{:10.4f}".format(jnp.array(losses[-10:]).mean()))
        
    #     return svi_batch, svi_state




