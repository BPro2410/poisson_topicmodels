.. _embedded_models:

================================================================================
Embedded Topic Models (ETM)
================================================================================

**Embedded Topic Models (ETM)** integrate pre-trained word embeddings (e.g., Word2Vec,
FastText, GloVe) into topic modeling. This often produces more semantically coherent topics.

Why Use Embeddings?
===================

Traditional topic models (PF, LDA) treat words as independent. ETM uses embeddings
to leverage semantic relationships:

**Without embeddings (PF)**:

- Words "car" and "automobile" are completely separate
- No semantic similarity captured
- May discover both in different topics

**With embeddings (ETM)**:

- "car" and "automobile" are similar in embedding space
- Model encourages co-occurrence in topics
- Topics are more coherent and interpretable

Benefits:

✓ More coherent topics
✓ Better handling of synonyms
✓ Leverages external semantic knowledge
✓ Often faster convergence

When to Use ETM
===============

Use ETM when:

✓ You have pre-trained embeddings available
✓ You want highly coherent topics
✓ Semantically similar words should group together
✓ You have sufficient computing resources

Consider basic PF if:

✗ No pre-trained embeddings available (or corpus-specific)
✗ You want complete control over topic formation
✗ Computational resources are limited
✗ Embedding artifacts would introduce bias

Model Overview
==============

ETM extends PF by constraining topics in embedding space:

**Key idea**: Topics (distributions over words) are located in the word embedding space.

**Mechanism**:

1. Each word has an embedding vector (pre-trained)
2. Each topic is located at a point in embedding space
3. Word probability in topic depends on distance/similarity

**Formally**:

.. code-block:: text

   P(word w | topic z) ∝ exp(-||embedding_w - topic_center_z||²)

Close words in embedding space have higher probability in same topic.

Basic Usage
===========

.. code-block:: python

   from poisson_topicmodels import ETM
   import numpy as np

   # Pre-trained embeddings: (vocab_size, embedding_dim)
   embeddings = load_pretrained_embeddings('glove')  # Shape: (500, 300)

   model = ETM(
       counts=counts,
       vocab=vocab,
       embeddings=embeddings,
       num_topics=10,
       batch_size=32,
       random_seed=42
   )

   params = model.train(num_iterations=100, learning_rate=0.01)

   # Results similar to other models
   top_words = model.get_top_words(n=10)

Loading Pre-trained Embeddings
==============================

**From file (GloVe format)**:

.. code-block:: python

   import numpy as np

   def load_glove_embeddings(filepath, vocab, embedding_dim=300):
       """Load GloVe embeddings for given vocabulary."""
       embeddings = {}
       with open(filepath, 'r') as f:
           for line in f:
               parts = line.split()
               word = parts[0]
               if word in vocab:
                   embeddings[word] = np.array(parts[1:], dtype=np.float32)

       # Create matrix for vocab
       embedding_matrix = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
       for i, word in enumerate(vocab):
           if word in embeddings:
               embedding_matrix[i] = embeddings[word]
           else:
               # Random embedding for OOV words
               embedding_matrix[i] = np.random.randn(embedding_dim) * 0.1

       return embedding_matrix

   embeddings = load_glove_embeddings('glove.6B.300d.txt', vocab)

**From gensim**:

.. code-block:: python

   from gensim.models import Word2Vec

   # Load Word2Vec model
   w2v_model = Word2Vec.load('word2vec_model.bin')

   # Extract embeddings for vocabulary
   embedding_dim = w2v_model.vector_size
   embeddings = np.zeros((len(vocab), embedding_dim))

   for i, word in enumerate(vocab):
       if word in w2v_model.wv:
           embeddings[i] = w2v_model.wv[word]
       else:
           embeddings[i] = np.random.randn(embedding_dim) * 0.1

**From fastText**:

.. code-block:: python

   from fastText import load_model

   model = load_model('fasttext_model.bin')

   embeddings = np.array([
       model.get_word_vector(word) for word in vocab
   ]).astype(np.float32)

Practical Example: News Classification
======================================

.. code-block:: python

   from poisson_topicmodels import ETM
   from gensim.models import Word2Vec

   # Train or load Word2Vec on news corpus
   w2v = Word2Vec(sentences=tokenized_documents, vector_size=300, window=5)

   # Create embedding matrix
   embeddings = np.array([
       w2v.wv[word] if word in w2v.wv else np.random.randn(300) * 0.1
       for word in vocab
   ])

   # Train ETM
   model = ETM(
       counts=counts,
       vocab=vocab,
       embeddings=embeddings,
       num_topics=15,
       batch_size=64
   )

   model.train(num_iterations=150, learning_rate=0.01)

   # Inspect topics
   top_words = model.get_top_words(n=15)
   for topic_id, words in enumerate(top_words):
       print(f"Topic {topic_id}: {', '.join(words)}")

Comparing ETM vs Standard Models
=================================

**Quality comparison**:

.. code-block:: python

   # Train multiple models
   pf_model = PF(counts, vocab, num_topics=10)
   pf_model.train(num_iterations=100)

   etm_model = ETM(counts, vocab, embeddings, num_topics=10)
   etm_model.train(num_iterations=100)

   # Calculate coherence
   pf_coherence = pf_model.compute_coherence()
   etm_coherence = etm_model.compute_coherence()

   print(f"PF Coherence: {pf_coherence.mean():.3f}")
   print(f"ETM Coherence: {etm_coherence.mean():.3f}")
   # ETM usually has higher coherence

**Visual comparison**:

.. code-block:: python

   # Display topic evolution
   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(1, 2, figsize=(12, 5))

   # PF topics
   for topic_id in range(5):
       top_words_pf = pf_model.get_top_words(n=5)[topic_id]
       axes[0].text(0.1, 0.9 - topic_id * 0.15,
                    f"T{topic_id}: {', '.join(top_words_pf)}")
   axes[0].set_title('PF Topics')

   # ETM topics
   for topic_id in range(5):
       top_words_etm = etm_model.get_top_words(n=5)[topic_id]
       axes[1].text(0.1, 0.9 - topic_id * 0.15,
                    f"T{topic_id}: {', '.join(top_words_etm)}")
   axes[1].set_title('ETM Topics')

   plt.tight_layout()
   plt.show()

Advanced: Custom ETM Variants
=============================

**Combine with seeding**:

.. code-block:: python

   # ETM with keyword guidance
   seeds = {
       0: ['climate', 'carbon', 'greenhouse'],
       1: ['economy', 'market', 'trade'],
   }

   # If ETM supports seeding (check documentation)
   model = ETM(
       counts=counts,
       vocab=vocab,
       embeddings=embeddings,
       num_topics=10,
       seeds=seeds,
       seed_strength=10.0
   )

**Combine with covariates**:

.. code-block:: python

   # ETM with covariate effects
   covariates = np.random.randn(num_docs, 2)

   # If supported
   model = ETM(
       counts=counts,
       vocab=vocab,
       embeddings=embeddings,
       num_topics=10,
       covariates=covariates
   )

Embedding Quality Matters
==========================

**Good embeddings**:

- Trained on large, relevant corpus
- Capture domain-specific semantics
- Adequate dimensionality (usually 300+)
- Word coverage matches your vocabulary

**Issues with bad embeddings**:

- Random or poorly trained embeddings don't help
- May actually hurt performance
- Out-of-vocabulary words hurt coverage
- Mismatch between corpus domain and embedding domain

**Best practices**:

1. Use embeddings trained on similar corpus
2. Check coverage: ``coverage = sum(word in embeddings for word in vocab)``
3. Verify quality: do related words have similar embeddings?
4. Compare ETM vs PF on your data

Troubleshooting ETM
===================

**Problem**: ETM doesn't improve over basic PF

*Solution*:
- Check embedding quality (is coverage good?)
- Try different embedding model
- Ensure preprocessing matches embedding vocabulary
- ETM might not help for all datasets

**Problem**: Training is slow

*Solution*:
- Embeddings add computational cost
- Reduce num_topics
- Increase batch_size
- Reduce vocabulary size
- Use GPU

**Problem**: Topics look worse than PF

*Solution*:
- Bad embedding quality
- Domain mismatch (embeddings from different corpus)
- Embedding dimensionality too low (try higher-dim embeddings)
- Training not converged (more iterations)

**Problem**: Many OOV (out-of-vocabulary) words

*Solution*:
- Check embedding file covers words in vocab
- Preprocess to match embedding vocabulary
- Use subword embeddings (fastText) instead of word-level

Evaluation
==========

Metrics for ETM:

.. code-block:: python

   # Standard metrics still apply
   topics = etm_model.get_topics()
   doc_topics = etm_model.get_document_topics()

   # Coherence
   coherence = etm_model.compute_coherence()

   # Perplexity (if held-out data available)
   perplexity = etm_model.compute_perplexity(held_out_counts)

   # Compare with baseline
   pf_coherence = pf_model.compute_coherence()
   improvement = (coherence.mean() - pf_coherence.mean()) / pf_coherence.mean()
   print(f"ETM improves coherence by {improvement:.1%}")

Relationship to Other Models
=============================

**ETM vs PF**: Adds semantic constraints through embeddings

**ETM + SPF = Guided ETM**: Combine embedding quality with domain guidance

**ETM + CPF**: Use embeddings + metadata (if supported)

**ETM + TBIP**: Ideal points with better topic discovery

When to Stack Models:

- Use basic PF first to understand topics
- Add embeddings (ETM) if coherence is a concern
- Add seeds (SPF) if you have domain knowledge
- Add TBIP/CPF if you have additional structure

Next Steps
==========

- :doc:`../tutorials/index` - Advanced training techniques
- :doc:`../how_to_guides/index` - Practical recipes
- :doc:`../api/index` - Complete ETM API documentation
- Explore different embedding sources for your domain
