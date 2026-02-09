.. _covariate_models:

================================================================================
Covariate Models (CPF & CSPF)
================================================================================

**Covariate-augmented models** extend basic topic modeling to account for document-level
metadata. Use them when you want to understand how topics vary across groups or conditions.

Covariate Poisson Factorization (CPF)
=====================================

CPF models how **external factors** (covariates) influence topic distributions.

**When to use**:

✓ Document metadata available (author, date, category)
✓ Want to understand topic variation across groups
✓ Interested in covariate effects on topics

**Example use cases**:

- Topic proportions across different authors
- How topics evolve over time
- Topic differences between datasets/corpora
- Topic structure by document category

The Model
---------

CPF extends PF by making topic distributions depend on covariates:

.. code-block:: text

   Standard PF:
   θ_d ~ Gamma(α, α)  [independent from anything]

   CPF:
   θ_d ~ Gamma(exp(γ + x_d * β), α)  [depends on document covariates x_d]

Where:
- x_d = covariate values for document d
- β = covariate effects (regression coefficients)
- γ = baseline (intercept)

**Interpretation**:

- If β_k > 0: Higher covariate value → higher topic k intensity
- If β_k < 0: Higher covariate value → lower topic k intensity
- β_k ≈ 0: Covariate has little effect on topic k

Usage Example
-------------

.. code-block:: python

   from poisson_topicmodels import CPF
   import numpy as np

   # Document covariates (e.g., author type scores)
   # Shape: (num_documents, num_covariates)
   covariates = np.random.randn(100, 2)  # 2 covariates

   model = CPF(
       counts=counts,
       vocab=vocab,
       num_topics=10,
       covariates=covariates,
       batch_size=32,
       random_seed=42
   )

   params = model.train(num_iterations=100, learning_rate=0.01)

   # Extract covariate effects
   effects = model.get_covariate_effects()  # Shape: (num_topics, num_covariates)
   # effects[k, j] = effect of covariate j on topic k

Interpreting Covariate Effects
-------------------------------

.. code-block:: python

   effects = model.get_covariate_effects()
   # Shape: (num_topics, num_covariates)

   # Topic 0: how does each covariate affect it?
   topic_0_effects = effects[0, :]
   # e.g., [2.5, -0.8]
   #   → Covariate 0: +2.5 effect (strong positive)
   #   → Covariate 1: -0.8 effect (moderate negative)

**Visualization**:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Heatmap of effects
   plt.imshow(effects, cmap='RdBu_r', vmin=-3, vmax=3)
   plt.xlabel('Covariates')
   plt.ylabel('Topics')
   plt.colorbar(label='Effect size')
   plt.title('Covariate Effects on Topics')
   plt.show()

Practical Example: Time Evolution
---------------------------------

Analyze how topics change across time periods:

.. code-block:: python

   # Create time-based covariate
   time_covariate = np.repeat(np.arange(10), 10)  # 10 decades, 10 docs each
   covariates = time_covariate.reshape(-1, 1) / 10  # Normalize

   model = CPF(counts, vocab, num_topics=5, covariates=covariates)
   model.train(num_iterations=100, learning_rate=0.01)

   # Topic 0's time effect
   effects = model.get_covariate_effects()
   time_effect = effects[0, 0]
   # If positive: topic 0 increases over time
   # If negative: topic 0 decreases over time

Covariate Seeded PF (CSPF)
==========================

**CSPF** combines seeded guidance (from SPF) with covariate modeling (from CPF).

Use when:

✓ You have prior knowledge about topics (seeds)
✓ You have document metadata (covariates)
✓ You want guided discovery with metadata effects

Usage
-----

.. code-block:: python

   from poisson_topicmodels import CSPF

   # Seeds for guided discovery
   seeds = {
       0: ['election', 'vote', 'candidate'],
       1: ['market', 'economy', 'trade'],
   }

   # Metadata effects
   covariates = np.random.randn(100, 1)

   model = CSPF(
       counts=counts,
       vocab=vocab,
       num_topics=2,
       seeds=seeds,
       covariates=covariates,
       seed_strength=10.0,
       batch_size=32,
       random_seed=42
   )

   params = model.train(num_iterations=100, learning_rate=0.01)

Practical Example: Geographic Topic Analysis
---------------------------------------------

.. code-block:: python

   # Documents with geographic metadata
   # Analyze how topics differ by region

   regions = np.array([0, 0, 1, 1, 2, 2, ...])  # Region IDs

   # Convert to one-hot encoding
   num_regions = 3
   covariates = np.zeros((len(regions), num_regions))
   for i, region in enumerate(regions):
       covariates[i, region] = 1

   model = CSPF(
       counts=counts,
       vocab=vocab,
       num_topics=5,
       covariates=covariates,
       random_seed=42
   )

   model.train(num_iterations=100, learning_rate=0.01)

   # Analyze regional topic differences
   effects = model.get_covariate_effects()
   # Row i = effect on topic i
   # Column j = region j

Tips for Covariate Modeling
============================

**Centering**: Center continuous covariates

.. code-block:: python

   covariates = (covariates - covariates.mean(axis=0)) / covariates.std(axis=0)

**Scaling**: Normalize to [0,1] or standardize

.. code-block:: python

   from sklearn.preprocessing import StandardScaler
   covariates = StandardScaler().fit_transform(covariates)

**Categorical**: Convert to dummy variables

.. code-block:: python

   import pandas as pd
   df = pd.DataFrame({'category': ['A', 'B', 'A', 'B', ...]})
   covariates = pd.get_dummies(df, drop_first=False).values

**Interpretation**: Document what covariates represent

.. code-block:: python

   # Label your covariates
   covariate_names = ['time_period', 'political_lean', 'media_type']
   # Use when reporting effects

Common Patterns
===============

**Author Effects**: How different authors use topics

.. code-block:: python

   author_ids = np.array([0, 1, 0, 2, 1, ...])
   author_dummies = np.eye(num_authors)[author_ids]

   model = CPF(counts, vocab, num_topics=10, covariates=author_dummies)

**Category Effects**: How topics differ by category

.. code-block:: python

   categories = ['news', 'opinion', 'news', 'opinion', ...]
   category_ids = [1 if c == 'opinion' else 0 for c in categories]
   covariates = np.array(category_ids).reshape(-1, 1)

   model = CPF(counts, vocab, num_topics=10, covariates=covariates)

**Temporal Effects**: How topics evolve over time

.. code-block:: python

   timestamps = np.array([2010, 2011, 2015, ...])
   covariates = ((timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())).reshape(-1, 1)

   model = CPF(counts, vocab, num_topics=10, covariates=covariates)

Visualization Examples
======================

**Effect Heatmap**:

.. code-block:: python

   effects = model.get_covariate_effects()
   import seaborn as sns
   sns.heatmap(effects, cmap='RdBu_r', center=0, annot=True)
   plt.title('Covariate Effects by Topic')
   plt.ylabel('Topic')
   plt.xlabel('Covariate')

**Effect by Category**:

.. code-block:: python

   # Plot effect of category on each topic
   for topic_id in range(num_topics):
       effect = effects[topic_id, 0]
       plt.bar(topic_id, effect, color='red' if effect < 0 else 'blue')
   plt.xlabel('Topic')
   plt.ylabel('Category Effect')

**Document-Topic by Category**:

.. code-block:: python

   doc_topics = model.get_document_topics()

   # Average topics by category
   for category_id in range(num_categories):
       mask = categories == category_id
       avg_topics = doc_topics[mask].mean(axis=0)
       plt.plot(avg_topics, marker='o', label=f'Category {category_id}')

   plt.xlabel('Topic')
   plt.ylabel('Average Intensity')
   plt.legend()

Troubleshooting
===============

**Problem**: Covariate effects are near zero

*Solution*:
- Covariates may not influence topics meaningfully
- Increase number of iterations
- Check covariate variation (are they constant?)
- Covariates might truly have no effect

**Problem**: Training diverges or NaNs

*Solution*:
- Normalize covariates to reasonable scale
- Reduce learning rate
- Check covariates don't have extreme values

**Problem**: Over-reliance on covariates (ignores data)

*Solution*:
- Reduce covariate weights (if supported)
- Use fewer covariates
- Increase training data

Next Steps
==========

- :doc:`ideal_points` - Estimate author positions with TBIP
- :doc:`../tutorials/index` - Advanced modeling techniques
- :doc:`../api/index` - Full API reference for CPF/CSPF
