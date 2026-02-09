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
       author_ids=author_ids,
       num_topics=10,
       num_dimensions=1,  # 1D position (left-right)
       batch_size=32,
       random_seed=42
   )

   params = model.train(num_iterations=100, learning_rate=0.01)

   # Extract results
   ideal_points = model.get_ideal_points()  # (num_authors, num_dimensions)
   # e.g., shape (3, 1) with values like [[-1.2], [0.5], [0.8]]

Interpreting Ideal Points
=========================

**1D Case** (single position axis):

.. code-block:: python

   ideal_points = model.get_ideal_points()  # Shape: (num_authors, 1)

   # Author 0: position -2.3 → far left
   # Author 1: position 0.0 → center
   # Author 2: position 1.5 → somewhat right

**Visualization**:

.. code-block:: python

   import matplotlib.pyplot as plt
   positions = ideal_points[:, 0]
   author_names = ['author_0', 'author_1', 'author_2']
   plt.scatter(positions, range(len(positions)))
   for i, name in enumerate(author_names):
       plt.annotate(name, (positions[i], i))
   plt.xlabel('Ideal Point (left ← → right)')
   plt.show()

**2D Case** (two position dimensions):

.. code-block:: python

   ideal_points = model.get_ideal_points()  # Shape: (num_authors, 2)

   plt.scatter(ideal_points[:, 0], ideal_points[:, 1])
   for i, name in enumerate(author_names):
       plt.annotate(name, ideal_points[i])
   plt.xlabel('Dimension 1')
   plt.ylabel('Dimension 2')
   plt.show()

Topic-Word-Author Relationships
===============================

TBIP discovers how words vary across author positions:

.. code-block:: python

   # Get topic-word distributions
   topics = model.get_topics()  # Standard topic-word matrix

   # Get author-topic distributions (averaged across documents)
   author_topics = model.get_author_topics()  # (num_authors, num_topics)

   # Top words globally
   top_words = model.get_top_words(n=10)

   # Word probabilities may differ by author
   # This is implicitly captured in ideal points

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
       author_ids=legislator_ids,
       num_topics=20,
       num_dimensions=1,
       batch_size=64,
       random_seed=42
   )

   model.train(num_iterations=200, learning_rate=0.01)

   # Get positions
   ideal_points = model.get_ideal_points()

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

Advanced: Multi-dimensional Ideal Points
=========================================

Estimate author positions on multiple dimensions:

.. code-block:: python

   # 2D space: (left-right, libertarian-authoritarian)
   model = TBIP(
       counts=counts,
       vocab=vocab,
       author_ids=author_ids,
       num_topics=20,
       num_dimensions=2,
       batch_size=32
   )

   model.train(num_iterations=200)
   ideal_points = model.get_ideal_points()  # Shape: (num_authors, 2)

   # Visualization
   plt.scatter(ideal_points[:, 0], ideal_points[:, 1])
   for i, name in enumerate(author_names):
       plt.annotate(name, ideal_points[i])
   plt.xlabel('Economic (Left ← → Right)')
   plt.ylabel('Authoritarian (↓ ← → ↑)')
   plt.show()

Interpretation Notes:

- **Dimension selection**: Choose based on domain knowledge
- **Axis meaning**: Not automatically labeled; determined by language patterns
- **Stability**: Relative positions matter more than absolute values
- **Sign flip**: Dimensions may flip sign; only relative order matters

Validating Ideal Points
=======================

**Compare with known positions**:

.. code-block:: python

   # If ground truth available
   true_positions = get_known_positions()
   estimated = ideal_points[:, 0]

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
   leftmost_author = np.argmin(ideal_points[:, 0])
   rightmost_author = np.argmax(ideal_points[:, 0])

   print(f"Leftmost author (ID {leftmost_author}):")
   print(f"Top documents: {get_top_docs(leftmost_author, n=3)}")
   print("\nRightmost author (ID {rightmost_author}):")
   print(f"Top documents: {get_top_docs(rightmost_author, n=3)}")

**Topic usage patterns**:

.. code-block:: python

   # Do extreme authors differ in topic usage?
   author_topics = model.get_author_topics()

   leftmost_topics = author_topics[leftmost_author]
   rightmost_topics = author_topics[rightmost_author]

   # Find distinctive topics
   topic_diff = leftmost_topics - rightmost_topics
   distinctive_topics = np.argsort(np.abs(topic_diff))[-5:]
   print(f"Most distinctive topics: {distinctive_topics}")

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
