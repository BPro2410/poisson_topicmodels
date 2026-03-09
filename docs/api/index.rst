.. _api:

================================================================================
API Reference
================================================================================

Complete API documentation for poisson-topicmodels.

.. note::

   API documentation is auto-generated from docstrings in the source code.
   For detailed documentation on specific classes and methods, see the
   source code or use Python's ``help()`` function.

Core Classes
============

**Models**

- `PF` – Poisson Factorization (unsupervised)
- `SPF` – Seeded Poisson Factorization (guided)
- `CPF` – Covariate Poisson Factorization (with metadata)
- `CSPF` – Covariate Seeded Poisson Factorization (both)
- `ETM` – Embedded Topic Models (with embeddings)
- `TBIP` – Text-Based Ideal Points (author positions)

**Base Classes**

- `NumpyroModel` – Base class for all models
- `Metrics` – Evaluation metrics

**Utilities**

- Data loading and preprocessing utilities
- Matrix utilities
- Visualization helpers

Module Organization
====================

The public API is organized as:

.. code-block:: python

   from poisson_topicmodels import (
       # Models
       PF,
       SPF,
       CPF,
       CSPF,
       ETM,
       TBIP,
       # Base classes
       NumpyroModel,
       Metrics,
   )

   from poisson_topicmodels.utils import (
       # Utility functions here
   )

Model API Pattern
=================

All models follow the same interface:

**1. Initialize**

.. code-block:: python

   model = PF(counts=dtm, vocab=vocab, num_topics=10, **kwargs)

**2. Train**

.. code-block:: python

   params = model.train_step(num_steps=100, lr=0.01)

**3. Extract**

.. code-block:: python

   topics, topic_probs = model.return_topics()
   beta = model.return_beta()
   top_words = model.return_top_words_per_topic(n=10)

**4. Inspect**

.. code-block:: python

   # For covariate models (CPF, CSPF)
   effects = model.return_covariate_effects()

Common Parameters
=================

**Data**

- ``counts`` (csr_matrix): Document-term matrix (documents × terms)
- ``vocab`` (ndarray): Vocabulary terms, shape (num_words,)

**Model configuration**

- ``num_topics`` (int): Number of topics to discover
- ``num_dimensions`` (int, TBIP only): Ideal points dimensionality
- ``embeddings`` (ndarray, ETM only): Pre-trained embeddings

**Training**

- ``num_steps`` (int): Training steps
- ``lr`` (float): Optimization learning rate
- ``batch_size`` (int): Documents per iteration

**Other**

- ``random_seed`` (int): Reproducibility seed

Common Methods
==============

**return_topics()**
   Returns topic-word distributions

   Returns: tuple of (vocab_size, num_topics) array and topic probabilities

**return_beta()**
   Returns topic-word probability matrix as a DataFrame

   Returns: DataFrame of shape (vocab_size, num_topics)

**return_top_words_per_topic(n=10)**
   Returns top n words per topic

   Returns: list of lists of top words per topic

**return_covariate_effects()** *(CPF, CSPF only)*
   Returns covariate effect estimates

**train_step(num_steps, lr, ...)**
   Trains the model using SVI

   Returns: dictionary with training parameters

Essential Documentation
=======================

**For basic usage**

- :doc:`../getting_started/index` – Quick start
- :doc:`../tutorials/tutorial_training` – Training guide
- :doc:`../fundamentals/index` – Model explanations
- :doc:`../how_to_guides/index` – Practical guides

Type Hints
==========

All functions include type hints for IDE support:

.. code-block:: python

   def return_topics(self) -> Tuple[np.ndarray, np.ndarray]:
       """Get topic-word distributions.

       Returns:
           Tuple of (topics array, topic probabilities)
       """
       ...

This enables autocomplete and helps catch errors early.

Error Handling
==============

Models validate inputs and provide clear error messages:

.. code-block:: python

   try:
       model = PF(counts, vocab, num_topics=10)
   except ValueError as e:
       print(f"Invalid input: {e}")

Common errors and solutions documented in :doc:`../how_to_guides/index`.

Performance Notes
=================

- Use sparse matrices (CSR format) for large vocabularies
- GPU acceleration requires setting JAX_PLATFORMS=gpu
- Batch size affects memory usage and speed
- See :doc:`../tutorials/tutorial_gpu` for optimization

API Stability
=============

- Public API (what you import) is stable
- Internal implementation may change
- Breaking changes documented in release notes
- Deprecations announced one version ahead

Version Information
===================

Check installed version:

.. code-block:: python

   import poisson_topicmodels
   print(poisson_topicmodels.__version__)

For version history, see :doc:`../release_notes/index`.

Links
=====

- **Full docs**: :ref:`genindex` – Index of all classes/functions
- **GitHub**: https://github.com/BPro2410/poisson_topicmodels

Next Steps
==========

- Learn models: :doc:`../fundamentals/index`
- Train models: :doc:`../tutorials/index`
- Solve tasks: :doc:`../how_to_guides/index`
