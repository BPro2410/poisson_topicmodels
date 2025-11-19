============
Installation
============

Requirements
------------

- Python >= 3.9
- jax
- jaxlib
- numpyro
- optax
- tqdm
- scipy
- pandas
- numpy

Install via pip
---------------

.. code-block:: bash

   pip install topicmodels

For GPU acceleration, install `jax` with CUDA support.
See the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_.


Docker
---------------
We also provide a Dockerfile that sets up an environment with all dependencies installed.
You can find the Dockerfile in our GitHub repository `here <https://github.com/BPro2410/topicmodels_package/blob/main/Dockerfile>`_.


Troubleshooting
---------------

In case you are running into troubles installing topicmodels, or have limited GPU support, please consider the tips mentioned in the `Numpyro Installation documentation <https://num.pyro.ai/en/latest/getting_started.html#installation>`_.

If you need further assistance with GPU support, please visit the `JAX GPU installation instructions <https://github.com/jax-ml/jax#pip-installation-gpu-cuda>`_.


You can also install topicmodels from `source <https://github.com/BPro2410/topicmodels_package>`_.
