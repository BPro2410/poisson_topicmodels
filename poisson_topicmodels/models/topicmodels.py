import os

from jax import jit
from numpyro.infer import SVI, TraceMeanField_ELBO
from optax import adam
from tqdm import tqdm

from .CPF import CPF
from .CSPF import CSPF
from .ETM import ETM
from .PF import PF
from .SPF import SPF
from .TBIP import TBIP

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
    supported_models = ["SPF", "CSPF", "TBIP", "TVTBIP", "PF", "CPF", "ETM"]
    if model not in supported_models:
        raise ValueError(
            "Please select a model that is supported in topicmodels package. Supported models are: {supported_models}"
        )

    if model == "SPF":
        return SPF
    if model == "CSPF":
        return CSPF
    if model == "TBIP":
        return TBIP
    if model == "TVTBIP":
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

            def __repr__(self):
                return f"{self.model_name} initialized with {self.K} topics to be estimated on {self.D} documents."

        return DynamicTM(model, *args, **kwargs)
