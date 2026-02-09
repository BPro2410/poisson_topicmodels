.. _seeded_models:

================================================================================
Seeded Models (SPF & Keywords)
================================================================================

**Seeded Poisson Factorization (SPF)** extends the basic PF model by incorporating
domain knowledge through **keyword priors**. If you have ideas about what topics should
look like, seeding guides the model toward discovering those topics.

When to Use Seeded Models
==========================

Use SPF when:

✓ You have prior knowledge about expected topics
✓ You can define a few keywords per expected topic
✓ You want to guide discovery without full supervision
✓ You need interpretable results aligned with expectations

Consider unsupervised PF if:

✗ You have no prior knowledge
✗ You want purely exploratory analysis
✗ You want to avoid bias from expectations

The Model
=========

**Extension of PF**:

SPF adds keyword guidance via `Dirichlet priors <https://en.wikipedia.org/wiki/Dirichlet_distribution>`_:

- For seeded topics: Place stronger prior on seed words
- For unseeded topics: Use standard prior (same as PF)

**Generative Process**:

Similar to PF, but topic-word distribution draws from informed priors:

.. code-block:: text

   For each topic k:
   - If topic k has seeds:
       β_k ~ Dirichlet(η_seed)  # η_seed has higher values at seed positions
   - Else:
       β_k ~ Dirichlet(η)       # Standard prior

This makes seed words more likely in their designated topics.

Basic Usage
===========

.. code-block:: python

   from poisson_topicmodels import SPF
   import numpy as np

   # Define seed words for each topic
   seeds = {
       0: ['research', 'data', 'experiment'],        # Science topic
       1: ['president', 'congress', 'vote'],         # Politics topic
       2: ['recipe', 'cooking', 'flavor'],           # Food topic
   }

   model = SPF(
       counts=counts,
       vocab=vocab,
       num_topics=3,
       seeds=seeds,
       batch_size=32,
       random_seed=42
   )

   params = model.train(num_iterations=100, learning_rate=0.01)

   # Results similar to PF
   top_words = model.get_top_words(n=10)

How Seeding Works
=================

**Step 1: Define Seeds**

Seeds are keywords you want associated with each topic:

.. code-block:: python

   seeds = {
       0: ['virus', 'vaccine', 'infection'],    # Medical
       1: ['climate', 'carbon', 'greenhouse'],   # Environment
       2: ['economy', 'trade', 'market'],        # Economics
   }

**Step 2: Seeds Influence Prior**

The model places higher prior probability on seed words:

.. code-block:: python

   # Without seeds: all words equally likely a priori
   # With seeds: seed words have boosted probability

**Step 3: Model Learns**

Training combines the informative prior with data:

- Data pulls topics toward observed word distributions
- Prior pulls topics toward seed words
- Result: Topics incorporate seeds + learned patterns

**Step 4: Interpret Results**

Top words typically include most seeds, plus additional related words:

.. code-block:: text

   Input seeds: ['virus', 'vaccine', 'infection']

   Learned top words: ['virus', 'vaccine', 'infection', 'disease',
                       'patients', 'treatment', 'symptoms', ...]

Advanced: Seed Strength
========================

Control how strongly seeds influence the model via `seed_strength`:

.. code-block:: python

   # Weak seeding: gentle guidance
   model = SPF(counts, vocab, num_topics=3, seeds=seeds, seed_strength=1.0)

   # Medium seeding: standard (default = 10.0)
   model = SPF(counts, vocab, num_topics=3, seeds=seeds, seed_strength=10.0)

   # Strong seeding: seeds dominate
   model = SPF(counts, vocab, num_topics=3, seeds=seeds, seed_strength=100.0)

Guidelines:

- **Lower values** (1-5): Seeds as gentle suggestions
- **Medium values** (10-50): Moderate influence (recommended)
- **High values** (100+): Seeds strongly constrain topics

Choose based on balance desired between prior knowledge and data.

Designing Good Seeds
====================

**Do's**:

✓ Use 3-10 words per topic (avoid too few or too many)
✓ Use words characteristic of the topic
✓ Use actual vocabulary words from your corpus
✓ Ensure seeds don't overlap across topics
✓ Choose frequent words (not rare/obscure)

**Don'ts**:

✗ Don't use generic/stopwords as seeds
✗ Don't use words not in your vocabulary
✗ Don't repeat seeds across topics
✗ Don't use too many seeds (>20 per topic)
✗ Don't seed every topic (leave some unseeded for discovery)

Example Good Seeds:

.. code-block:: python

   seeds = {
       0: ['neural', 'learning', 'network', 'algorithm'],
       1: ['legislation', 'congress', 'bill', 'committee'],
       2: ['earnings', 'profit', 'revenue', 'dividend'],
   }

Example Bad Seeds:

.. code-block:: python

   # Bad: Too generic
   seeds = {
       0: ['the', 'is', 'and'],  # Stopwords
       1: ['thing', 'stuff'],    # Too generic
   }

   # Bad: Not in vocabulary
   seeds = {
       0: ['xyz123', 'nonexistent_word'],  # Not in vocab
   }

   # Bad: Overlapping
   seeds = {
       0: ['research', 'data'],
       1: ['research', 'experiment'],  # 'research' in both!
   }

Mixing Seeded and Unseeded Topics
==================================

You can seed only some topics:

.. code-block:: python

   # Topic 0 and 1 are seeded, topic 2 is discovered freely
   seeds = {
       0: ['virus', 'vaccine', 'infection'],
       1: ['climate', 'carbon', 'warming'],
       # Topic 2 has no seeds - discovered from data
   }

   model = SPF(
       counts=counts,
       vocab=vocab,
       num_topics=3,
       seeds=seeds,
       random_seed=42
   )

**Use case**: When you have ideas about some topics but want other topics discovered.

Iterative Seeding
=================

1. Train unsupervised PF model
2. Inspect top words - identify coherent topics
3. Design seeds based on top words
4. Train SPF with those seeds
5. Compare results and refine seeds if needed

.. code-block:: python

   # Step 1: Unsupervised discovery
   pf_model = PF(counts, vocab, num_topics=5)
   pf_model.train(num_iterations=100, learning_rate=0.01)

   # Step 2: Inspect and design seeds
   top_words_pf = pf_model.get_top_words(n=10)
   print("Top words from unsupervised model:")
   for topic_id, words in enumerate(top_words_pf):
       print(f"Topic {topic_id}: {', '.join(words)}")

   # Step 3: Define seeds based on patterns
   seeds = {
       0: list(top_words_pf[0][:5]),  # Use top 5 from topic 0
       1: list(top_words_pf[1][:5]),
   }

   # Step 4: Train seeded model
   spf_model = SPF(counts, vocab, num_topics=5, seeds=seeds)
   spf_model.train(num_iterations=100, learning_rate=0.01)

   # Step 5: Compare and evaluate
   top_words_spf = spf_model.get_top_words(n=10)

Practical Example
=================

Seeding a corpus of news articles:

.. code-block:: python

   from poisson_topicmodels import SPF

   # Define themes you expect in news
   news_seeds = {
       0: ['election', 'vote', 'candidate', 'campaign'],  # Politics
       1: ['stock', 'market', 'trade', 'investment'],     # Business
       2: ['hurricane', 'flood', 'weather', 'storm'],     # Weather
       3: ['covid', 'virus', 'pandemic', 'vaccine'],      # Health
   }

   model = SPF(
       counts=counts,
       vocab=vocab,
       num_topics=4,
       seeds=news_seeds,
       seed_strength=10.0,
       batch_size=64,
       random_seed=42
   )

   params = model.train(num_iterations=150, learning_rate=0.01)

   # Expected: Topics strongly align with seed themes
   # but include additional related words from data
   top_words = model.get_top_words(n=15)
   for topic_id, words in enumerate(top_words):
       print(f"Topic {topic_id}: {', '.join(words)}")

Troubleshooting Seeds
=====================

**Problem**: Seeds don't appear in top words

*Solution*:
- Check seeds are in vocabulary: ``vocab in [word in seed for word in seeds]``
- Increase seed_strength
- Ensure seed words actually appear in documents
- Check seed words aren't too rare

**Problem**: Non-seeded topics disappear

*Solution*:
- Reduce seed strength
- Use fewer seeds per topic
- Ensure sufficient data per topic

**Problem**: Seeds make topics less coherent

*Solution*:
- Your seeds might not match data patterns
- Review actual top words from unsupervised PF
- Design seeds that align with data

Validation
==========

How to validate seeded models:

.. code-block:: python

   # 1. Check top words include seeds
   top_words = model.get_top_words(n=20)
   for topic_id, words in enumerate(top_words):
       topic_seeds = [s for s in news_seeds[topic_id] if s in words]
       coverage = len(topic_seeds) / len(news_seeds[topic_id])
       print(f"Topic {topic_id} seed coverage: {coverage:.1%}")

   # 2. Measure coherence
   coherence = model.compute_coherence()
   print(f"Average coherence: {coherence.mean():.3f}")

   # 3. Compare with unsupervised
   pf_model = PF(counts, vocab, num_topics=4, random_seed=42)
   pf_model.train(num_iterations=150)
   pf_coherence = pf_model.compute_coherence()
   print(f"PF coherence: {pf_coherence.mean():.3f} vs SPF: {coherence.mean():.3f}")

Comparison with Unsupervised
=============================

+-------------------+-------------------+-------------------+
| Aspect            | PF (Unsupervised) | SPF (Seeded)      |
+===================+===================+===================+
| Prior knowledge   | Not used          | Used as priors    |
| needed?           |                   |                   |
+-------------------+-------------------+-------------------+
| Bias?             | None              | Toward seeds      |
+-------------------+-------------------+-------------------+
| Interpretability  | Variable          | Usually better    |
+-------------------+-------------------+-------------------+
| Time to insights  | Requires reading  | Fast (seeds guide) |
|                   | top words         |                   |
+-------------------+-------------------+-------------------+
| Flexibility       | High              | Guided            |
+-------------------+-------------------+-------------------+

Next Steps
==========

- :doc:`covariate_models` - Add metadata to models
- :doc:`../how_to_guides/index` - Practical guides
- :doc:`../api/index` - SPF API reference
