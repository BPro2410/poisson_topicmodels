from abc import abstractmethod, ABC
import jax
from jax import random, jit
import jax.numpy as jnp
from optax import adam
from numpyro.infer import SVI, TraceMeanField_ELBO
from tqdm import tqdm

from packages.modules.Metrics import Metrics

class NumpyroModel(ABC):

    Metrics = Metrics(loss = list())
    
    @abstractmethod
    def _model(self):
        pass

    @abstractmethod
    def _guide(self):
        pass

    def _get_batch(self, rng, Y):
        D_batch = random.choice(rng, jnp.arange(self.D), shape = (self.batch_size,))
        # Y_batch = jax.device_put(jnp.array(Y[D_batch].toarray()), jax.devices("cpu")[0])
        Y_batch = jnp.array(Y[D_batch].toarray())

        # Ensure the shape of Y_batch is (batch_size, V)
        assert Y_batch.shape == (self.batch_size, self.V), f"Shape mismatch: {Y_batch.shape} != ({self.batch_size}, {self.V})"

        return Y_batch, D_batch
    
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

        Y_batch, D_batch = self._get_batch(random.PRNGKey(1), self.counts)

        svi_state = svi_batch.init(
            random.PRNGKey(0), Y_batch = Y_batch, d_batch = D_batch
        )
    
        rngs = random.split(random.PRNGKey(2), num_steps)
        # losses = list()
        pbar = tqdm(range(num_steps))

        for step in pbar:
            Y_batch, D_batch = self._get_batch(rngs[step], self.counts)
            svi_state, loss = svi_batch_update(
                svi_state, Y_batch=Y_batch, d_batch = D_batch
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
