"""
Poisson Topic Models: Probabilistic topic modeling with Bayesian inference using JAX and NumPyro

This package provides implementations of various Poisson-based topic models including:
- PF: Poisson Factorization
- SPF: Supervised Poisson Factorization
- CPF: Covariates-augmented Poisson Factorization
- CSPF: Covariates and Supervised Poisson Factorization
- TBIP: Topic-Based Ideological Point Estimation
- ETM: Embedded Topic Model

Example:
    >>> from poisson_topicmodels import PF
    >>> model = PF(counts=counts, vocab=vocab, num_topics=10, batch_size=100)
    >>> model.fit(num_epochs=100)
"""

from .models import CPF, CSPF, ETM, PF, SPF, TBIP, Metrics, NumpyroModel, topicmodels

__all__ = ["CPF", "CSPF", "ETM", "PF", "SPF", "TBIP", "Metrics", "NumpyroModel", "topicmodels"]

__version__ = "0.1.0"
