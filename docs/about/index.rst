.. _about:

================================================================================
About poisson-topicmodels
================================================================================

What is poisson-topicmodels?
=============================

**poisson-topicmodels** is a modern Python package for probabilistic topic modeling
using Bayesian inference. Built on the foundation of `JAX <https://github.com/google/jax>`_
and `NumPyro <https://github.com/pyro-ppl/numpyro>`_, it provides researchers and
practitioners with powerful tools for extracting interpretable semantic structure from
text data.

Statement of Need
=================

Traditional topic modeling packages (e.g., Gensim, scikit-learn's LDA) rely on older
inference methods and lack the flexibility needed for modern research. **poisson-topicmodels**
addresses key gaps:

1. **Modern Probabilistic Inference**

   Built on NumPyro, the package enables automatic differentiation, probabilistic
   programming, and integration with cutting-edge Bayesian methods. This provides
   a solid foundation for advanced inference techniques.

2. **Advanced Topic Models**

   Goes beyond LDA with guided topic discovery (keyword priors), covariate effects,
   ideal point estimation, and embeddings—all with principled Bayesian inference.

3. **GPU Acceleration**

   Leverages JAX for transparent GPU computation, essential for large-scale corpus
   analysis and enabling research that would be prohibitively slow on CPU.

4. **Scalability & Reproducibility**

   Optimized for mini-batch SVI training with built-in seed control for exact
   reproducibility—critical for research validation and publication.

5. **Research-Friendly API**

   Purpose-built for computational social science and NLP researchers who need
   interpretable, flexible models beyond black-box approaches.

Use Cases
=========

**poisson-topicmodels** is ideal for:

- **Computational Social Science**: Analyze legislative texts, social media discourse,
  and policy documents to understand political positions and debate evolution.

- **Computational Linguistics**: Extract semantic structures from linguistic corpora,
  study language variation, and uncover latent themes in text collections.

- **Digital Humanities**: Analyze large text archives, track conceptual change over time,
  and understand thematic evolution in literature, historical documents, and cultural texts.

- **Market Research**: Understand customer sentiment, topic distribution in reviews,
  and brand perception from unstructured text data.

- **Academic Research**: Efficiently analyze large corpora of academic papers, identify
  research trends, and discover connections between topics and fields.

Core Philosophy
===============

The design of **poisson-topicmodels** is guided by these principles:

**Interpretability First**
   Topic models should produce human-interpretable results. The package emphasizes
   clear semantics and provides tools to understand and validate discovered topics.

**Flexibility**
   Different research questions require different models. The package provides a
   suite of related models with shared APIs, allowing researchers to choose the
   right tool for their problem.

**Reproducibility**
   Research results must be reproducible. Every component supports deterministic
   execution through seed control and careful API design.

**Modern Stack**
   Building on JAX and NumPyro allows the package to leverage modern automatic
   differentiation and probabilistic programming capabilities.

**Performance**
   The package is designed for large-scale analysis through GPU acceleration and
   efficient mini-batch training procedures.

Related Packages
================

For context, here are some related packages in the topic modeling and Bayesian
inference ecosystem:

- **Gensim**: Classic topic modeling library with LDA, LSI, and word embeddings
- **scikit-learn**: Machine learning toolkit includes LDA implementation
- **PyMC**: Probabilistic programming framework for Bayesian modeling
- **Stan**: Probabilistic programming language used via pystan interface
- **PyTorch and TensorFlow**: Deep learning frameworks with probabilistic extensions

What Sets poisson-topicmodels Apart
====================================

1. **JAX-based**: Modern automatic differentiation backend with transparent GPU support
2. **NumPyro Integration**: Direct access to probabilistic programming tools
3. **Scalable SVI**: Efficient stochastic variational inference with mini-batch training
4. **Multiple Models**: Comprehensive suite of related models (PF, SPF, CPF, CSPF, ETM, TBIP)
5. **Research-Oriented**: Designed for researchers who need flexibility and interpretability
6. **Type Hints**: Comprehensive type hints for better IDE support and code clarity
7. **Active Development**: Continuously improved with modern inference techniques

Theory Behind Topic Models
===========================

Topic models are statistical models that discover abstract "topics" in a collection
of documents. Each topic is a distribution over words, and each document is a mixture
of topics.

**Poisson Factorization (PF)** is the foundational model in this package. Unlike LDA
(which uses a multinomial distribution), PF uses a Poisson distribution to model document
word counts. This has several advantages:

- More natural for count data
- Computational efficiency
- Extension flexibility for covariates and side information

**Guided variants** (SPF, CPF, CSPF) extend the basic model to incorporate:

- **Domain knowledge** through keyword priors
- **Auxiliary information** through document-level covariates
- **Combined constraints** for sophisticated analyses

**Ideal point models** (TBIP) extend the framework to estimate positions of authors
on latent dimensions based on their language use.

**Embedded models** (ETM) incorporate pre-trained word embeddings to improve semantic
coherence.

Getting Help
============

- **First time?** Start with :doc:`../getting_started/index`
- **Want details?** Check the :doc:`../fundamentals/index`
- **Need examples?** See the :doc:`../examples_guide/index`
- **API questions?** Refer to :doc:`../api/index`
- **Contributing?** Read :doc:`../contributing_guide/index`
