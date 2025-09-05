.. topicmodels documentation master file

=================================
topicmodels: Probabilistic Models
=================================

``topicmodels`` is a Python package for probabilistic topic modeling using
Bayesian inference built on `JAX <https://github.com/google/jax>`_ and `NumPyro <https://github.com/pyro-ppl/numpyro>`_.

It provides implementations of several advanced topic models:

- **Poisson Factorization (PF)** – unsupervised baseline topic model.
- **Seeded Poisson Factorization (SPF)** – guided topic modeling with keyword priors.
- **Covariate Poisson Factorization (CPF)** – models topics influenced by external covariates.
- **Covariate Seeded Poisson Factorization (CSPF)** – combines seeded guidance with covariate effects.
- **Text-Based Ideal Points (TBIP)** – estimates ideal points of authors from text.

The package emphasizes **scalability**, **interpretability**, and **flexibility**.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   intro/installation
   intro/user_guide
   intro/examples
   modules
   intro/contributing
