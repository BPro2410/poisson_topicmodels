.. _installation:

================================================================================
Installation
================================================================================

poisson-topicmodels works with Python 3.11+. We recommend using a fresh environment
for installation to avoid dependency conflicts.

Quick Install (Recommended)
==========================

For most users, the simplest approach is to install from PyPI:

.. code-block:: bash

   pip install poisson-topicmodels

This installs the latest stable release with all dependencies.

.. note::
   We strongly recommend using a virtual environment to isolate your project dependencies.

Installation from Source
========================

For development work or to use the latest features, you can install from source:

.. code-block:: bash

   git clone https://github.com/BPro2410/topicmodels_package.git
   cd topicmodels_package
   pip install -e .

The ``-e`` flag installs the package in editable mode, allowing you to modify the
source code and have changes reflected immediately.

Development Installation
=========================

For contributing to the project or running tests, install with development dependencies:

.. code-block:: bash

   git clone https://github.com/BPro2410/topicmodels_package.git
   cd topicmodels_package
   pip install -e ".[dev,docs]"

This includes:

- **dev**: Testing frameworks (pytest, pytest-cov), code quality tools (black, isort, mypy, pylint, flake8)
- **docs**: Documentation tools (sphinx, sphinx-rtd-theme, myst-parser)

GPU Support (JAX)
=================

**poisson-topicmodels** leverages JAX for transparent GPU acceleration. Out of the box,
JAX is installed with CPU support. To enable GPU acceleration:

For NVIDIA GPUs (CUDA), install the CUDA-enabled JAX:

.. code-block:: bash

   pip install jax[cuda12_cudnn]

For AMD GPUs (ROCm):

.. code-block:: bash

   pip install jax[rocm]

For Apple Silicon GPUs:

.. code-block:: bash

   pip install jax[metal]

See `JAX Installation <https://jax.readthedocs.io/en/latest/installation.html>`_
for detailed GPU setup instructions.

Conda Installation
==================

If you prefer Conda, you can install from conda-forge once the package is published there:

.. code-block:: bash

   conda install -c conda-forge poisson-topicmodels

For now, pip install is recommended.

Environment Management
======================

Using venv
----------

Create a virtual environment with venv:

.. code-block:: bash

   python3 -m venv topicmodels_env
   source topicmodels_env/bin/activate  # On Windows: topicmodels_env\Scripts\activate
   pip install poisson-topicmodels

Using Conda
-----------

Create and manage environments with Conda:

.. code-block:: bash

   conda create -n topicmodels python=3.11
   conda activate topicmodels
   pip install poisson-topicmodels

Verifying the Installation
===========================

After installation, verify that everything works correctly:

.. code-block:: python

   import poisson_topicmodels
   print(poisson_topicmodels.__version__)

   from poisson_topicmodels import PF
   print("Installation successful!")

Or run the test suite:

.. code-block:: bash

   pytest tests/test_imports.py

Troubleshooting
===============

**ImportError: No module named 'poisson_topicmodels'**

- Ensure you've activated the correct virtual environment
- Try reinstalling: ``pip install --force-reinstall poisson-topicmodels``

**JAX CUDA/GPU errors**

- Verify JAX was installed correctly: ``python -c "import jax; print(jax.devices())``
- Check JAX GPU documentation: `JAX GPU Guide <https://jax.readthedocs.io/en/latest/installation.html>`_

**Dependency conflicts**

- Use a fresh virtual environment: ``python3 -m venv fresh_env && source fresh_env/bin/activate``
- Install only the base package first, then add extras as needed

**Documentation build errors (when installing from source)**

- Ensure sphinx and related tools are installed: ``pip install ".[docs]"``

Need Help?
==========

- 📖 Read the :doc:`../getting_started/index`
- 🐛 Report issues on `GitHub <https://github.com/BPro2410/topicmodels_package/issues>`_
- 💬 Join discussions on `GitHub Discussions <https://github.com/BPro2410/topicmodels_package/discussions>`_

System Requirements
====================

- **Python**: 3.11, 3.12, or 3.13
- **OS**: Linux, macOS, or Windows
- **RAM**: 2 GB minimum (8 GB+ recommended for large corpora)
- **Disk**: ~500 MB for installation, depends on corpus size

Optional dependencies for GPU acceleration:

- **NVIDIA**: CUDA Toolkit 11.8+ and cuDNN 8.6+ (for GPU support)
- **AMD**: ROCm 5.0+ (for AMD GPU support)
