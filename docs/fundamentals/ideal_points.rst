.. _ideal_points:

================================================================================
Ideal Points Models (TBIP)
================================================================================

**Text-Based Ideal Points (TBIP)** is a specialized model for estimating **latent positions**
(ideal points) of authors based on their language use. Commonly used in political science
and social media analysis.

What Are Ideal Points?
=======================

Ideal points are latent coordinates representing positions on abstract dimensions:

**Example**: Political polarization

- Left politicians use words: "equality", "justice", "workers", "government"
- Right politicians use words: "freedom", "liberty", "business", "market"
- Model estimates position on left-right spectrum from text

**Example**: Product stance

- Critics use words: "broken", "poor quality", "disappointed"
- Supporters use words: "amazing", "excellent", "recommend"
- Model estimates critic vs. supporter position

Model Intuition
===============

TBIP works by:

1. Discovering topics in text corpus
2. Analyzing word usage patterns within topics
3. Inferring author positions that explain language variation

Higher-dimensional spaces possible (not just 1D left-right):

- 2D: (left-right, authoritarian-libertarian)
- 3D+: Custom dimensions discovered from data

When to Use TBIP
================

Use TBIP when:

✓ You have author-attributed text (speeches, tweets, reviews)
✓ You assume polarization or position variation
✓ You want to estimate latent author positions
✓ You're interested in discourse analysis

Don't use if:

✗ Text is anonymous or unattributed
✗ No meaningful position variation expected
✗ You only care about topics, not author positions

Basic Usage
===========

.. code-block:: python

   from poisson_topicmodels import TBIP
   import numpy as np

   # Author IDs indicating who wrote each document
   author_ids = np.array([0, 1, 0, 2, 1, 0, ...])  # 3 authors

   model = TBIP(
       counts=counts,
       vocab=vocab,
       authors=author_ids,
       num_topics=10,
       batch_size=32,
   )

   params = model.train_step(num_steps=200, lr=0.01)

   # Extract results
   ideal_points_df = model.return_ideal_points()  # DataFrame: author, ideal_point, std
   print(ideal_points_df)

Interpreting Ideal Points
=========================

**1D Case** (single position axis):

.. code-block:: python

   ideal_points_df = model.return_ideal_points()
   print(ideal_points_df)
   #        author  ideal_point       std
   # 0    author_A        -2.30      0.15
   # 1    author_B         0.00      0.12
   # 2    author_C         1.50      0.18

**Visualization** (built-in):

.. code-block:: python

   # Publication-ready 1-D scatter with optional credible intervals
   fig, ax = model.plot_ideal_points(show_ci=True, ci=0.95)

   # Or manually:
   import matplotlib.pyplot as plt
   df = model.return_ideal_points()
   plt.scatter(df['ideal_point'], range(len(df)))
   for i, row in df.iterrows():
       plt.annotate(row['author'], (row['ideal_point'], i))
   plt.xlabel('Ideal Point (left ← → right)')
   plt.show()


Topic-Word-Author Relationships
===============================

TBIP discovers how words vary across author positions:

.. code-block:: python

   # Get word-topic associations
   beta = model.return_beta()  # DataFrame

   # Top words globally
   top_words = model.return_top_words_per_topic(n=10)

   # Ideological words per topic — shows which words load most on
   # the ideological dimension
   ideo_words = model.return_ideological_words(topic=0, n=10)
   print(ideo_words)
   # Columns: word, eta, direction
   # direction: 'positive' or 'negative' end of the axis

Practical Example: Political Speeches
=====================================

.. code-block:: python

   # Analyze legislative speeches
   # Documents: individual speeches
   # Authors: legislators
   # Goal: estimate left-right position from language

   from poisson_topicmodels import TBIP

   # Load speech dataset
   speeches = load_speeches()  # (num_speeches, num_documents)
   legislator_ids = speeches['legislator'].values  # who said each speech
   counts = speech_dtm  # document-term matrix

   model = TBIP(
       counts=counts,
       vocab=vocab,
       authors=legislator_ids,
       num_topics=20,
       batch_size=64,
   )

   model.train_step(num_steps=200, lr=0.01)

   # Get positions
   ideal_points_df = model.return_ideal_points()
   model.summary()

   # Built-in visualization with credible intervals
   model.plot_ideal_points(show_ci=True)

   # Ideological words for the most political topic
   print(model.return_ideological_words(topic=0, n=15))

   # Compare with known party affiliation
   parties = legislator_ids_to_parties(legislator_ids)

   import matplotlib.pyplot as plt
   for party_id, party in enumerate(['Democrat', 'Republican']):
       mask = parties == party_id
       plt.hist(ideal_points[mask], alpha=0.5, label=party)
   plt.xlabel('Ideal Point (left ← → right)')
   plt.legend()
   plt.show()
   # Expected: Democrats mostly negative, Republicans mostly positive

Validating Ideal Points
=======================

**Compare with known positions**:

.. code-block:: python

   # If ground truth available
   true_positions = get_known_positions()
   estimated = model.return_ideal_points()['ideal_point'].values

   # Correlation should be high
   correlation = np.corrcoef(true_positions, estimated)[0, 1]
   print(f"Correlation: {correlation:.3f}")  # Should be > 0.7 ideally

   # Spearman rank correlation (order matters)
   from scipy.stats import spearmanr
   rank_corr, p_value = spearmanr(true_positions, estimated)
   print(f"Rank correlation: {rank_corr:.3f}, p={p_value:.4f}")

**Qualitative inspection**:

.. code-block:: python

   # Read documents from extreme authors
   df = model.return_ideal_points()
   leftmost_author = df.iloc[0]['author']   # sorted by ideal_point
   rightmost_author = df.iloc[-1]['author']

   print(f"Leftmost author (ID {leftmost_author}):")
   print(f"Top documents: {get_top_docs(leftmost_author, n=3)}")
   print("\nRightmost author (ID {rightmost_author}):")
   print(f"Top documents: {get_top_docs(rightmost_author, n=3)}")

**Topic usage patterns**:

.. code-block:: python

   # Which words distinguish the extremes the most?
   for topic_id in range(min(3, model.num_topics)):
       ideo = model.return_ideological_words(topic=topic_id, n=5)
       print(f"\nTopic {topic_id} ideological words:")
       print(ideo)

Relationship to Other Models
=============================

**TBIP vs. PF**: Adds author position estimation

- PF: Discovers topics only
- TBIP: Discovers topics AND author positions

**TBIP vs. CPF**: Different covariate handling

- CPF: Document-level continuous covariates
- TBIP: Author-level latent positions

**Typical workflow**:

1. Start with PF or SPF to understand topics
2. If interested in author positions, add TBIP
3. Optional: compare with CPF using author dummies as covariates

Implementation Details
======================

**Identification**: Ideal points can be flipped in sign (both left and right position
work); only relative order is meaningful.

**Centering**: Model centers ideal points at 0 by default (mean = 0).

**Scaling**: Values are on arbitrary scale; interpret using relative differences.

**Multiple dimensions**: Discovered dimensions may not have clear interpretations.
This is normal—inspect word distributions to understand.

Troubleshooting
===============

**Problem**: Ideal points don't seem meaningful

*Solution*:
- Check author IDs are correct
- Ensure sufficient documents per author
- Inspect topics and words
- Try different num_topics or num_dimensions
- Increase training iterations

**Problem**: Positions don't match known affiliations

*Solution*:
- Known affiliations might not align with language patterns
- Try different num_dimensions
- Check if covariate (e.g., party) matches topic structure
- Language use might reveal different dimensions than official positions

**Problem**: Training is slow**

*Solution*:
- Reduce number of topics
- Increase batch size
- Reduce vocabulary (remove rare words)
- Use GPU: ``export JAX_PLATFORMS=gpu``

Next Steps
==========

- :doc:`embedded_models` - Exploring ETM with embeddings
- :doc:`../tutorials/index` - Advanced techniques
- :doc:`../api/index` - Complete TBIP API reference
