from abc import abstractmethod, ABC
import jax
from jax import random, jit
import jax.numpy as jnp
from optax import adam
from numpyro.infer import SVI, TraceMeanField_ELBO
from tqdm import tqdm
import numpy as np
import pandas as pd

from packages.models.Metrics import Metrics

class NumpyroModel(ABC):
    """
    Abstract base class for all used probabilistic models.
    Each model has to implement at least their own Model and Guide.
    """
    

    Metrics = Metrics(loss = list())
    
    @abstractmethod
    def _model(self):
        pass

    @abstractmethod
    def _guide(self):
        pass

    def _get_batch(self, rng, Y):
        """
        Helper function to obtain a batch of data, convert from scipy.sparse to jax.numpy.array.
        
        Parameters
        ----------
        rng : jax.random.PRNGKey
            Random number generator key.
        Y : scipy.sparse.csr_matrix
            The word counts array.
        
        Returns
        -------
        tuple
            Y_batch : numpy.ndarray
                Word counts for the batch.
            D_batch : numpy.ndarray
                Indices of documents in the batch.
        """
        D_batch = random.choice(rng, jnp.arange(self.D), shape = (self.batch_size,))
        # Y_batch = jax.device_put(jnp.array(Y[D_batch].toarray()), jax.devices("cpu")[0])
        Y_batch = jnp.array(Y[D_batch].toarray())

        # Ensure the shape of Y_batch is (batch_size, V)
        assert Y_batch.shape == (self.batch_size, self.V), f"Shape mismatch: {Y_batch.shape} != ({self.batch_size}, {self.V})"

        return Y_batch, D_batch
    
    def train_step(self, num_steps, lr):
        """
        Train the model using SVI.
        
        Parameters
        ----------
        num_steps : int
            Number of training steps.
        lr : float
            Learning rate for the optimizer.
        
        Returns
        -------
        dict
            The estimated parameters after training.
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
        
        self.estimated_params = svi_batch.get_params(svi_state)
        
        return self.estimated_params

    
    def return_topics(self):
        """
        Return the topics for each document.

        Returns
        -------
        tuple
            categories : numpy.ndarray
                Array of topics indices for each document.
            E_theta : numpy.ndarray
                Estimated topic proportions for each document.
        """
        E_theta = self.estimated_params["theta_shape"] / self.estimated_params["theta_rate"]
        return np.argmax(E_theta, axis = 1), E_theta

    def return_beta(self):
        """
        Return the beta matrix for the model.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing the beta matrix with words as rows and topics as columns.
        """
        E_beta = self.estimated_params["beta_shape"] / self.estimated_params["beta_rate"]
        return pd.DataFrame(jnp.transpose(E_beta), index = self.vocab)



    
