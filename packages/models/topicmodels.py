from packages.models.CSPF import CSPF
from packages.models.SPF import SPF
from packages.models.TBIP import TBIP
from packages.models.PF import PF
from packages.models.CPF import CPF
from packages.models.ETM import ETM


from jax import jit
from optax import adam
from numpyro.infer import SVI, TraceMeanField_ELBO
from tqdm import tqdm

import os
os.getcwd()



def get_base_class(model):
    """
    Return the base class for a given model name.
    
    Parameters
    ----------
    model : str
        Name of the model.
    
    Returns
    -------
    class
        Corresponding model class.
    """
    if model == "SPF":
        return SPF
    if model == "CSPF":
        return CSPF
    if model == "TBIP":
        return TBIP
    if model == "PF":
        return PF
    if model == "CPF":
        return CPF
    if model == "ETM":
        return ETM


class topicmodels:

    # inherits model() and guide() upon initialization
    def __new__(cls, model, *args, **kwargs):
        """
        Create a new instance of the topic model dynamically based on the model name.
        
        Parameters
        ----------
        model : str
            Name of the model.
        
        Returns
        -------
        object
            Instance of the dynamically created model class.
        """
        base_class = get_base_class(model)
        
        class DynamicTM(base_class):
            """
            Dynamic Topic Model class inheriting from the specified base class.
            """
            # define here class attributes from parent class (if available)

            def __init__(self, model, *args, **kwargs):
                """
                Initialize the dynamic topic model.
                
                Parameters
                ----------
                model : str
                    Name of the model.
                """
                super().__init__(*args, **kwargs)
                self.model_name = model
            
            # def get_batch(self, rng, Y):
            #     D_batch = random.choice(rng, jnp.arange(self.D), shape = (self.batch_size,))
            #     # Y_batch = jax.device_put(jnp.array(Y[D_batch].toarray()), jax.devices("cpu")[0])
            #     Y_batch = jnp.array(Y[D_batch].toarray())

            #     # Ensure the shape of Y_batch is (batch_size, V)
            #     assert Y_batch.shape == (self.batch_size, self.V), f"Shape mismatch: {Y_batch.shape} != ({self.batch_size}, {self.V})"

            #     return Y_batch, D_batch
        
        return DynamicTM(model, *args, **kwargs)
    

    





