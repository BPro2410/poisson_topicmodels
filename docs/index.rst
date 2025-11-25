.. topicmodels documentation master file

========================================================
topicmodels: Probabilistic Topic Modeling with JAX
========================================================

``topicmodels`` is a comprehensive Python package for probabilistic topic modeling using
Bayesian inference built on `JAX <https://github.com/google/jax>`_ and `NumPyro <https://github.com/pyro-ppl/numpyro>`_.

Powered by GPU-accelerated inference and professional-grade type hints (90% coverage).

It provides implementations of several advanced topic models:

- **Poisson Factorization (PF)** – unsupervised baseline topic model.
- **Seeded Poisson Factorization (SPF)** – guided topic modeling with keyword priors.
- **Covariate Poisson Factorization (CPF)** – models topics influenced by external covariates.
- **Covariate Seeded Poisson Factorization (CSPF)** – combines seeded guidance with covariate effects.
- **Text-Based Ideal Points (TBIP)** – estimates ideal points of authors from text.
- **Time-Varying Text-Based Ideal Points (TVTBIP)** – captures temporal dynamics in authors' ideal points.
- **Structual Text-Based Scaling (STBS)** – models text data with structural information.
- **Embedded Topic Models (ETM)** – integrates word embeddings into topic modeling.
- ... and more models to come!


The package emphasizes **scalability**, **interpretability**, and **flexibility**.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   intro/installation
   intro/user_guide
   intro/examples
   modules
   intro/contributing
