.. _api:

================================================================================
API Reference
================================================================================

Complete API documentation for poisson-topicmodels.

.. toctree::
   :maxdepth: 2
   :caption: API Components

   models
   utils
   metrics

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

   params = model.train(num_iterations=100, learning_rate=0.01)

**3. Extract**

.. code-block:: python

   topics = model.get_topics()
   doc_topics = model.get_document_topics()
   top_words = model.get_top_words(n=10)

**4. Evaluate**

.. code-block:: python

   coherence = model.compute_coherence()

Common Parameters
================

**Data**

- ``counts`` (csr_matrix): Document-term matrix (documents × terms)
- ``vocab`` (ndarray): Vocabulary terms, shape (num_words,)

**Model configuration**

- ``num_topics`` (int): Number of topics to discover
- ``num_dimensions`` (int, TBIP only): Ideal points dimensionality
- ``embeddings`` (ndarray, ETM only): Pre-trained embeddings

**Training**

- ``num_iterations`` (int): Training steps
- ``learning_rate`` (float): Optimization step size
- ``batch_size`` (int): Documents per iteration

**Other**

- ``random_seed`` (int): Reproducibility seed

Common Methods
==============

**get_topics()**
   Returns topic-word distributions

   Returns: (vocab_size, num_topics) array

**get_document_topics()**
   Returns document-topic distributions

   Returns: (num_documents, num_topics) array

**get_top_words(n=10)**
   Returns top n words per topic

   Returns: (num_topics, n) string array

**compute_coherence()**
   Computes coherence metric per topic

   Returns: (num_topics,) array of coherence scores

**train(num_iterations, learning_rate, ...)**
   Trains the model

   Returns: dictionary with training parameters

Essential Documentation
=======================

**For basic usage**

- :doc:`../getting_started/index` – Quick start
- :doc:`../tutorials/tutorial_training` – Training guide

**For model details**

- :doc:`models` – All model classes
- :doc:`../fundamentals/index` – Model explanations

**For specific tasks**

- :doc:`utils` – Utility functions
- :doc:`metrics` – Evaluation metrics
- :doc:`../how_to_guides/index` – Practical guides

Type Hints
==========

All functions include type hints for IDE support:

.. code-block:: python

   def get_topics(self) -> np.ndarray:
       """Get topic-word distributions.

       Returns:
           Array of shape (vocab_size, num_topics) with topic distributions
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

Common errors and solutions documented in :doc:`../how_to_guides/troubleshoot`.

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

- **Models**: :doc:`models` – Detailed model API
- **Utils**: :doc:`utils` – Utility functions
- **Metrics**: :doc:`metrics` – Evaluation metrics
- **Full docs**: :ref:`genindex` – Index of all classes/functions
- **GitHub**: https://github.com/BPro2410/topicmodels_package

Next Steps
==========

- Learn models: :doc:`../fundamentals/index`
- Train models: :doc:`../tutorials/index`
- Solve tasks: :doc:`../how_to_guides/index`
