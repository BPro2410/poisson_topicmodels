================================================================================
poisson-topicmodels: Probabilistic Topic Modeling with Bayesian Inference
================================================================================

.. image:: _static/logo_small.png
   :alt: poisson-topicmodels
   :align: center
   :width: 300px

**poisson-topicmodels** is a modern Python package for probabilistic topic modeling
using Bayesian inference, built on `JAX <https://github.com/google/jax>`_ and
`NumPyro <https://github.com/pyro-ppl/numpyro>`_.

It enables researchers and practitioners to extract interpretable semantic structure
from text data through advanced topic modeling techniques with transparent GPU acceleration
and reproducible results.

.. toctree::
   :maxdepth: 2
   :caption: Documentation Contents
   :hidden:

   about/index
   installation/index
   getting_started/index
   fundamentals/index
   tutorials/index
   how_to_guides/index
   api/index
   models
   examples_guide/index
   testing/index
   contributing_guide/index
   release_notes/index

Quick Links
===========

- **Get Started** – :doc:`getting_started/index` – 5-minute introduction and basic examples
- **Installation** – :doc:`installation/index` – Install poisson-topicmodels from PyPI or source
- **Fundamentals** – :doc:`fundamentals/index` – Learn core concepts and model variants
- **Tutorials** – :doc:`tutorials/index` – Step-by-step guides for different use cases
- **API Reference** – :doc:`api/index` – Complete API documentation with examples
- **How-To Guides** – :doc:`how_to_guides/index` – Practical recipes for common tasks
- **Examples** – :doc:`examples_guide/index` – Real-world examples and applications
- **Testing** – :doc:`testing/index` – How to test your code
- **Contributing** – :doc:`contributing_guide/index` – Contribute to the project
- **Release Notes** – :doc:`release_notes/index` – Version history and changelog

Key Features
============

✨ **Modern Probabilistic Inference**
   Built on NumPyro for automatic differentiation, probabilistic programming,
   and integration with cutting-edge Bayesian methods.

✨ **Advanced Topic Models**
   Beyond LDA: guided topic discovery, covariate effects, ideal point estimation,
   and word embeddings—all with principled Bayesian inference.

✨ **GPU Acceleration**
   Leverages JAX for transparent GPU computation, essential for large-scale
   corpus analysis.

✨ **Reproducible & Scalable**
   Mini-batch SVI training with built-in seed control for exact reproducibility.

✨ **Research-Friendly API**
   Purpose-built for computational social science and NLP researchers.

The Package at a Glance
=======================

The **poisson-topicmodels** library provides multiple topic modeling approaches:

+------------------------------------+------------------------+----------------------------------+
| Model                              | Use Case               | Key Feature                      |
+====================================+========================+==================================+
| **Poisson Factorization (PF)**     | Unsupervised baseline  | Fast, interpretable word-topic   |
|                                    |                        | associations                     |
+------------------------------------+------------------------+----------------------------------+
| **Seeded PF (SPF)**                | Guided discovery       | Incorporate domain knowledge via |
|                                    |                        | keyword priors                   |
+------------------------------------+------------------------+----------------------------------+
| **Covariate PF (CPF)**             | Covariate effects      | Model topics influenced by       |
|                                    |                        | document metadata                |
+------------------------------------+------------------------+----------------------------------+
| **Covariate Seeded PF (CSPF)**     | Guided + covariates    | Combine keyword guidance with    |
|                                    |                        | external factors                 |
+------------------------------------+------------------------+----------------------------------+
| **Text-Based Ideal Points (TBIP)** | Ideal point estimation | Estimate author positions from   |
|                                    |                        | legislative/social text          |
+------------------------------------+------------------------+----------------------------------+
| **Embedded Topic Models (ETM)**    | Modern embeddings      | Integrate pre-trained word       |
|                                    |                        | embeddings                       |
+------------------------------------+------------------------+----------------------------------+

**Core Capabilities**:

- ✓ Stochastic Variational Inference (SVI) with mini-batch training
- ✓ Transparent GPU acceleration via JAX
- ✓ Reproducible results with seed control
- ✓ Type hints and comprehensive API documentation
- ✓ >70% test coverage with continuous integration
- ✓ Clear error messages and input validation

Quick Start Example
===================

.. code-block:: python

   import numpy as np
   from scipy.sparse import csr_matrix
   from poisson_topicmodels import PF

   # Prepare data: document-term matrix and vocabulary
   counts = csr_matrix(np.random.poisson(2, (100, 500)).astype(np.float32))
   vocab = np.array([f'word_{i}' for i in range(500)])

   # Initialize and train model
   model = PF(counts, vocab, num_topics=10, batch_size=32)
   params = model.train_step(num_steps=200, lr=0.01, random_seed=42)

   # Summarize and inspect
   model.summary()
   top_words = model.return_top_words_per_topic(n=10)
   for topic_id, words in top_words.items():
       print(f"Topic {topic_id}: {', '.join(words)}")

   # Evaluate and visualize
   print(f"Topic diversity: {model.compute_topic_diversity():.3f}")
   model.plot_model_loss()
   model.plot_topic_prevalence()

See :doc:`getting_started/index` for a detailed walkthrough.

Community & Contributing
=========================

We welcome contributions! For guidelines, see the :doc:`contributing_guide/index`.

- 🐛 **Found a bug?** `Open an issue <https://github.com/BPro2410/topicmodels_package/issues>`_
- 💡 **Have a feature request?** `Start a discussion <https://github.com/BPro2410/topicmodels_package/discussions>`_
- 📚 **Want to contribute?** Check out our `contribution guidelines <https://github.com/BPro2410/topicmodels_package/blob/main/CONTRIBUTING.md>`_

License
=======

This project is licensed under the MIT License. See the `LICENSE <https://github.com/BPro2410/topicmodels_package/blob/main/LICENSE>`_
file for details.

Citation
========

If you use poisson-topicmodels in your research, please cite:

.. code-block:: bibtex

   @software{prostmaier2026poisson,
     title={poisson-topicmodels: Probabilistic Topic Modeling with Bayesian Inference},
     author={Prostmaier, Bernd and Grün, Bettina and Hofmarcher, Paul},
     year={2026},
     url={https://github.com/BPro2410/topicmodels_package}
   }
