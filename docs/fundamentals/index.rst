.. _fundamentals:

================================================================================
Fundamentals
================================================================================

This section covers the core concepts and models in poisson-topicmodels, providing
a deeper understanding of topic modeling and the different model variants available.

.. toctree::
   :maxdepth: 2
   :caption: Fundamentals

   core_concepts
   poisson_factorization
   seeded_models
   covariate_models
   ideal_points
   embedded_models

Overview of Available Models
=============================

The **poisson-topicmodels** package provides several related models addressing different
use cases:

**Unsupervised Baseline**

- **Poisson Factorization (PF)**: Discover topics without guidance or external information

**Guided Discovery**

- **Seeded PF (SPF)**: Incorporate domain knowledge through keyword priors

**Covariate Modeling**

- **Covariate PF (CPF)**: Model how topics are influenced by document-level covariates
- **Covariate Seeded PF (CSPF)**: Combine covariate effects with keyword guidance

**Advanced Models**

- **Text-Based Ideal Points (TBIP)**: Estimate author positions on latent dimensions
- **Embedded Topic Models (ETM)**: Integrate pre-trained word embeddings

Which Model Should I Use?
==========================

**Your question**: "I want to discover topics in my text data"

→ **Use**: Poisson Factorization (PF) or Seeded PF (SPF)

- **PF** if you have no prior knowledge about topics
- **SPF** if you can define some keyword seeds for expected topics

**Your question**: "I want to understand how topics vary by document attributes"

→ **Use**: Covariate PF (CPF) or Covariate Seeded PF (CSPF)

- Use when you have metadata (author, date, category) and want to model how topics
  are affected by these attributes

**Your question**: "I want to estimate author or speaker positions"

→ **Use**: Text-Based Ideal Points (TBIP)

- Model ideal points (positions on a latent scale) based on language use
- Useful for political polarization, sentiment analysis across authors

**Your question**: "I want to use pre-trained word embeddings"

→ **Use**: Embedded Topic Models (ETM)

- Incorporates semantic information from embeddings like Word2Vec or FastText
- Often produces more semantically coherent topics

Model Comparison Table
======================

+---------------------+---------------+-----------+----------+----------+
| Model               | Unsupervised? | Guides?   | Covariates? | Embeddings? |
+=====================+===============+===========+==========+==========+
| **PF**              | ✓             |           |          |        |
+---------------------+---------------+-----------+----------+----------+
| **SPF**             | ✓ (guided)    | ✓         |          |        |
+---------------------+---------------+-----------+----------+----------+
| **CPF**             | ✓             |           | ✓        |        |
+---------------------+---------------+-----------+----------+----------+
| **CSPF**            | ✓ (guided)    | ✓         | ✓        |        |
+---------------------+---------------+-----------+----------+----------+
| **TBIP**            |               |           |          |        |
+---------------------+---------------+-----------+----------+----------+
| **ETM**             | ✓             |           |          | ✓      |
+---------------------+---------------+-----------+----------+----------+

Common Patterns
===============

All models in poisson-topicmodels follow a consistent API:

**Create**: Initialize with data and parameters

.. code-block:: python

   model = PF(counts=counts, vocab=vocab, num_topics=10)

**Train**: Fit the model to data

.. code-block:: python

   params = model.train(num_iterations=100, learning_rate=0.01)

**Extract**: Get interpretable results

.. code-block:: python

   topics = model.get_topics()
   doc_topics = model.get_document_topics()
   top_words = model.get_top_words(n=10)

**Analyze**: Downstream analysis

.. code-block:: python

   coherence = model.compute_coherence()  # or other metrics

Probabilistic Background
=========================

All models in this package are built on **Poisson Factorization**, a probabilistic
framework for count data. Here's the core idea:

**Poisson Factorization Model**

For each document d and word w:

- **Observed**: word count $C_{dw}$ (how many times word w appears in document d)
- **Latent**: topic z (which topic generated this word)
- **Model**: $C_{dw} \sim \text{Poisson}(\sum_z \beta_z^w \theta_d^z)$

Where:
- $\beta_z^w$ = word w probability in topic z
- $\theta_d^z$ = topic z intensity in document d

**Why Poisson?**

- Natural for count data
- Mathematically elegant with exponential family
- Computationally efficient with SVI
- Flexible foundation for extensions

**Bayesian Inference**

We use **Stochastic Variational Inference (SVI)** to estimate posterior distributions:

- Place prior distributions on $\beta$ and $\theta$
- Learn approximate posterior through variational optimization
- Mini-batch training for scalability

Learn More
==========

Each model variant has dedicated documentation:

- :doc:`core_concepts` - Detailed statistical background
- :doc:`poisson_factorization` - PF and understanding results
- :doc:`seeded_models` - SPF for incorporating domain knowledge
- :doc:`covariate_models` - CPF and CSPF for structured data
- :doc:`ideal_points` - TBIP for position estimation
- :doc:`embedded_models` - ETM for embedding integration

Ready to dive deeper? Start with :doc:`core_concepts`.
