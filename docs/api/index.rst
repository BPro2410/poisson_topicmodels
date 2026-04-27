.. _api:

================================================================================
API Reference
================================================================================

Complete API documentation for poisson-topicmodels.

.. note::

   For auto-generated class and method documentation from docstrings, see
   :doc:`../models`.

Module Organization
====================

.. code-block:: python

   from poisson_topicmodels import (
       # Models
       PF,       # Poisson Factorization (unsupervised)
       SPF,      # Seeded Poisson Factorization (guided)
       CPF,      # Covariate Poisson Factorization (with metadata)
       CSPF,     # Covariate Seeded Poisson Factorization (both)
       ETM,      # Embedded Topic Models (with embeddings)
       TBIP,     # Text-Based Ideal Points (author positions)
       STBS,     # Structured Text-Based Scaling (topic-specific ideal points)
       # Base classes
       NumpyroModel,
       Metrics,
   )

Model API Pattern
=================

All models follow the same workflow:

**1. Initialize**

.. code-block:: python

   model = PF(counts=dtm, vocab=vocab, num_topics=10, batch_size=32)

**2. Train**

.. code-block:: python

   params = model.train_step(num_steps=200, lr=0.01, random_seed=42)

**3. Summarize**

.. code-block:: python

   model.summary()              # Formatted text summary of the fitted model

**4. Extract**

.. code-block:: python

   topics, e_theta = model.return_topics()         # Dominant topic per doc + proportions
   beta = model.return_beta()                       # Word–topic DataFrame (words × topics)
   top_words = model.return_top_words_per_topic(n=10)  # dict {topic_id: [words]}

**5. Evaluate**

.. code-block:: python

   coherence_df = model.compute_topic_coherence()   # NPMI coherence per topic
   diversity = model.compute_topic_diversity()       # Fraction of unique top words (0–1)

**6. Visualize**

.. code-block:: python

   model.plot_model_loss()               # Training loss curve
   model.plot_topic_prevalence()         # Mean topic prevalence bar chart
   model.plot_topic_correlation()        # Cosine-similarity heatmap
   model.plot_document_topic_heatmap()   # Document × topic heatmap
   model.plot_topic_wordclouds()         # Wordcloud per topic

Common Parameters
=================

**Data**

- ``counts`` (csr_matrix): Document-term matrix (documents × terms)
- ``vocab`` (ndarray): Vocabulary terms, shape ``(num_words,)``

**Model configuration**

- ``num_topics`` (int): Number of topics to discover (PF, CPF, TBIP, STBS, ETM)
- ``keywords`` (dict): Seed words per topic (SPF, CSPF)
- ``residual_topics`` (int): Extra unsupervised topics (SPF, CSPF)
- ``X_design_matrix`` (ndarray | DataFrame): Covariates (document-level for CPF/CSPF, author-level for STBS)
- ``authors`` (ndarray): Author labels per document (TBIP, STBS)
- ``embeddings_mapping`` (dict): Word → embedding vector (ETM)

**Training**

- ``num_steps`` (int): Training iterations
- ``lr`` (float): Learning rate (step size for optimizer)
- ``batch_size`` (int): Documents per training step
- ``random_seed`` (int): Reproducibility seed (supported by PF/SPF/CPF/CSPF/ETM)

Common Methods (all models)
============================

**``train_step(num_steps, lr, random_seed=None, ...)``**
   Train the model via Stochastic Variational Inference (SVI).

   Returns: ``dict`` of estimated parameters.

   Note: ``TBIP`` and ``STBS`` currently expose ``train_step(num_steps, lr)``
   without a ``random_seed`` argument.

**``return_topics()``**
   Returns ``(categories, E_theta)`` — dominant topic per document and
   document-topic proportions.

**``return_beta()``**
   Returns a ``pd.DataFrame`` of word–topic associations (words × topics).

**``return_top_words_per_topic(n=10)``**
   Returns a ``dict`` mapping topic identifiers to their top-n words.

**``summary(n_top_words=5)``**
   Prints a formatted summary of the fitted model, including loss,
   top words, and model-specific details.

**``compute_topic_coherence(texts=None, metric='c_npmi', top_n=10)``**
   Computes per-topic coherence scores (NPMI or UMass).

   Returns: ``pd.DataFrame`` with topic and coherence columns.

**``compute_topic_diversity(top_n=25)``**
   Fraction of unique words across all topics' top-n lists. Range 0–1.

**``plot_model_loss(window=10, save_path=None)``**
   Line chart of training loss (raw + smoothed). Returns ``(fig, ax)``.

**``plot_topic_prevalence(save_path=None)``**
   Horizontal bar chart of mean topic prevalence. Returns ``(fig, ax)``.

**``plot_topic_correlation(save_path=None)``**
   Cosine-similarity heatmap between topics. Returns ``(fig, ax)``.

**``plot_document_topic_heatmap(n_docs=50, sort_by_topic=False, save_path=None)``**
   Document × topic heatmap. Returns ``(fig, ax)``.

**``plot_topic_wordclouds(n_words=50, figsize=(16,12), save_path=None)``**
   One wordcloud per topic. Returns ``(fig, axes)``.

SPF-specific Methods
====================

**``plot_seed_effectiveness(save_path=None)``**
   Grouped bar chart comparing mean seed vs. non-seed word weights per topic.

   Returns: ``(fig, axes)``.

CPF-specific Methods
====================

**``return_covariate_effects()``**
   Point estimates of covariate effect matrix λ (covariates × topics).

   Returns: ``pd.DataFrame``.

**``return_covariate_effects_ci(ci=0.95)``**
   Covariate effects with Bayesian credible intervals.

   Returns: ``pd.DataFrame`` with columns ``covariate, topic, mean, lower, upper``.

**``plot_cov_effects(ci=0.95, topics=None, save_path=None)``**
   Forest plot of covariate effects with credible intervals.

   Returns: ``(fig, axes)``.

CSPF-specific Methods
=====================

Inherits all SPF methods (seeded topics) plus all CPF methods (covariate effects):

- ``return_covariate_effects()``
- ``return_covariate_effects_ci(ci=0.95)``
- ``plot_cov_effects(ci=0.95, ...)``

TBIP-specific Methods
=====================

**``return_ideal_points()``**
   Returns a ``pd.DataFrame`` with columns ``author, ideal_point, std``,
   sorted by ideal point.

**``return_ideological_words(topic, n=10)``**
   Top-n words with the strongest ideological loading (η) for a given topic.

   Returns: ``pd.DataFrame`` with columns ``word, eta, direction``.

**``plot_ideal_points(selected_authors=None, show_ci=False, ci=0.95, save_path=None)``**
   1-D scatter of author ideal points with optional credible intervals.

   Returns: ``(fig, ax)``.

STBS-specific Methods
=====================

**``return_ideal_points()``**
   Returns a ``pd.DataFrame`` with columns ``author, topic, ideal_point, std``
   for topic-specific author positions.

**``return_ideal_covariates()``**
   Returns a ``pd.DataFrame`` with columns ``covariate, topic, iota, std``
   for covariate effects on ideological positions.

**``plot_author_topic_heatmap(...)``**
   Heatmap of mean normalized author-topic intensities.

   Returns: ``(fig, ax)``.

**``plot_ideol_points(...)``**
   Dot plot of author ideology by topic, with optional grouping overlays.

   Returns: ``(fig, ax)``.

**``plot_iota_credible_intervals(ci=0.95, ...)``**
   Credible-interval plot for covariate-topic ideology coefficients.

   Returns: ``(fig, ax)``.

ETM-specific Methods
====================

ETM overrides ``return_topics()`` and ``return_beta()`` to use its neural encoder
and embedding-based topic–word computation. No additional public methods beyond
the common set.

Metrics Dataclass
=================

``Metrics`` tracks training diagnostics per model instance:

- ``loss`` (list): ELBO loss per training step
- ``coherence_scores`` (pd.DataFrame | None): Per-topic coherence if computed
- ``diversity`` (float | None): Topic diversity if computed
- ``reset()``: Clear all stored metrics

Error Handling
==============

Models validate inputs and provide clear error messages:

.. code-block:: python

   try:
       model = PF(counts, vocab, num_topics=10, batch_size=32)
   except ValueError as e:
       print(f"Invalid input: {e}")

Type Hints
==========

All functions include type hints for IDE support and static analysis.

Performance Notes
=================

- Use sparse matrices (CSR format) for large vocabularies
- GPU acceleration requires ``JAX_PLATFORMS=gpu``
- Batch size affects memory usage and speed
- See :doc:`../tutorials/tutorial_gpu` for optimization

API Stability
=============

- Public API (what you import) is stable
- Internal implementation may change
- Breaking changes documented in release notes

Next Steps
==========

- Auto-generated docs: :doc:`../models`
- Learn models: :doc:`../fundamentals/index`
- Train models: :doc:`../tutorials/index`
- Solve tasks: :doc:`../how_to_guides/index`
