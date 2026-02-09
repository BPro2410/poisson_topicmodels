.. _poisson_factorization:

================================================================================
Poisson Factorization (PF)
================================================================================

The **Poisson Factorization (PF)** model is the foundational, unsupervised topic model
in poisson-topicmodels. It automatically discovers topics without any prior guidance.

When to Use PF
==============

Use Poisson Factorization when:

✓ You want to discover topics without prior knowledge
✓ You have document-term matrices as input
✓ You need interpretable word-topic associations
✓ You want a fast baseline model
✓ You're exploring new document collections

Consider other models if:

✗ You have prior knowledge about expected topics (→ use SPF)
✗ You have document-level metadata (→ use CPF/CSPF)
✗ You need to estimate author positions (→ use TBIP)
✗ You want to leverage pre-trained embeddings (→ use ETM)

The Model
=========

**Generative Process**:

For a corpus with D documents and V vocabulary terms:

1. For each document d:

   - Draw document-topic intensity: $\theta_d \sim \text{Gamma}(\alpha, \alpha)^K$
   - For each word position in document:

     - Draw topic: $z_n \sim \text{Discrete}(\text{softmax}(\theta_d))$
     - Draw word: $w_n \sim \text{Discrete}(\beta_{z_n})$

2. For each topic k:

   - Draw topic-word distribution: $\beta_k \sim \text{Dirichlet}(\eta)$

**Key Properties**:

- **Unsupervised**: No labels or guidance needed
- **Flexible**: Works with any document collection
- **Interpretable**: Topics are directly interpretable as word distributions
- **Scalable**: Mini-batch SVI enables large-scale inference

Example: Basic Usage
====================

.. code-block:: python

   import numpy as np
   from scipy.sparse import csr_matrix
   from poisson_topicmodels import PF

   # Prepare data
   counts = csr_matrix(np.random.poisson(2, (100, 500)).astype(np.float32))
   vocab = np.array([f'word_{i}' for i in range(500)])

   # Create model
   model = PF(
       counts=counts,
       vocab=vocab,
       num_topics=10,
       batch_size=32,
       random_seed=42
   )

   # Train
   params = model.train(
       num_iterations=100,
       learning_rate=0.01
   )

   # Extract results
   topics = model.get_topics()              # (vocab_size, num_topics)
   doc_topics = model.get_document_topics()  # (num_docs, num_topics)
   top_words = model.get_top_words(n=10)    # Top words per topic

Interpreting Results
====================

**Topics (β)**:

Access via ``model.get_topics()`` - shape (vocab_size, num_topics)

Each column k represents topic k:

.. code-block:: python

   topic_5 = topics[:, 5]  # Topic 5 probabilities over vocabulary

Interpretation:
- ``topic_5[0.15]`` means word_0 has 15% probability in topic 5
- Top words with highest probabilities characterize the topic

**Document Topics (θ)**:

Access via ``model.get_document_topics()`` - shape (num_docs, num_topics)

Each row d represents document d:

.. code-block:: python

   doc_3_topics = doc_topics[3, :]  # Topic mixture for document 3
   # e.g., [0.6, 0.2, 0.15, 0.05, ...] → mostly topic 0

Interpretation:
- ``doc_3_topics[0.6]`` means topic 0 has 60% intensity in document 3
- Dominance identifies primary topic(s)

**Top Words**:

Access via ``model.get_top_words(n=10)`` - Top n words per topic

.. code-block:: python

   top_words = model.get_top_words(n=10)
   # Shape: (num_topics, n)

   # Topic 2 top words
   print(top_words[2])
   # ['research', 'data', 'experiment', 'analysis', ...]

Human Interpretation:
- Read top ~10-20 words for each topic
- Does the topic make sense thematically?
- Can you give it a meaningful label?

Example Interpretation Workflow
==============================

.. code-block:: python

   model = PF(counts, vocab, num_topics=5, batch_size=32)
   model.train(num_iterations=100, learning_rate=0.01)

   # Examine each topic
   top_words = model.get_top_words(n=15)

   for topic_id in range(5):
       print(f"\n=== Topic {topic_id} ===")
       words = top_words[topic_id]
       print(f"Top words: {', '.join(words)}")

       # Find documents dominated by this topic
       doc_topics = model.get_document_topics()
       top_docs = np.argsort(doc_topics[:, topic_id])[-3:]
       print(f"Top documents: {top_docs}")

Hyperparameter Selection
========================

**Number of Topics (K)**

Start with 10-20 topics. Adjust based on:

- **Coherence**: Do top words form meaningful themes?
- **Interpretability**: Can you label each topic?
- **Downstream task**: Does it improve your application?

.. code-block:: python

   # Try different numbers of topics
   for k in [5, 10, 20, 50]:
       model = PF(counts, vocab, num_topics=k, batch_size=32)
       model.train(num_iterations=100, learning_rate=0.01)

       # Evaluate quality (e.g., via coherence, downstream task)
       # or inspect top words manually

**Learning Rate (lr)**

Controls optimization step size. Default: 0.01

- **0.001**: Very conservative, slow convergence
- **0.01**: Standard, good for most cases
- **0.1**: Aggressive, may overshoot
- **1.0+**: Usually too large, diverges

Recommended: Start with 0.01, adjust if needed

**Batch Size**

Controls documents per iteration. Default: 32

- **16/32**: Small, noisier gradients, fast iterations
- **64/128**: Medium, balanced gradients, standard
- **256/512**: Large, stable gradients, slower iterations

Recommended: 32-128 for balance

**Iterations**

How long to train. Default: 100

Monitor loss:

.. code-block:: python

   params = model.train(
       num_iterations=200,
       learning_rate=0.01,
       verbose=True  # Print loss every iteration
   )

Suggested: Train until loss plateaus (visual inspection)

Training Tips
=============

**Use GPU**: Set JAX to use GPU for 10-100x speedup

.. code-block:: bash

   # In Python, set before importing JAX
   export JAX_PLATFORMS=gpu
   python script.py

**Reproducibility**: Set random seed

.. code-block:: python

   model = PF(counts, vocab, num_topics=10, random_seed=42)
   # Same seed → same results (good for research)

**Progress Monitoring**: Check loss trajectory

.. code-block:: python

   params = model.train(
       num_iterations=100,
       learning_rate=0.01,
       verbose=True
   )

**Early stopping**: Stop if loss plateaus

.. code-block:: python

   # Manual: train in chunks, monitor loss
   for iteration in range(10):
       params = model.train(num_iterations=10, learning_rate=0.01)
       print(f"Loss: {params['loss']}")
       if loss_not_improving:
           break

Common Issues and Solutions
============================

**Problem**: Topics look similar or contain generic words

*Solution*:
- Could be too many topics - reduce K
- Improve preprocessing (better stopword removal)
- Look at more top words to find differences

**Problem**: Some topics are all garbage words

*Solution*:
- Preprocess better (remove URLs, unicode artifacts, numbers)
- Reduce number of topics
- Check document-term matrix for data issues

**Problem**: Training is slow

*Solution*:
- Use GPU: ``export JAX_PLATFORMS=gpu``
- Increase batch size (more docs per iteration)
- Reduce vocabulary size (remove rare words)

**Problem**: Loss not decreasing

*Solution*:
- Increase learning rate (try 0.05-0.1)
- Check data: ensure proper document-term matrix format
- Try different random seed

Evaluation Metrics
==================

See :doc:`../api/index` for available metrics:

- **Coherence**: Do top words of a topic correlate?
- **Perplexity**: How well does model explain held-out data?
- **Topic diversity**: Are topics distinct?

Example:

.. code-block:: python

   coherence = model.compute_coherence()
   print(f"Average coherence: {coherence.mean()}")

Next Steps
==========

- **Add guidance**: Use :doc:`seeded_models` to incorporate domain knowledge
- **Model metadata**: Try :doc:`covariate_models` if you have document attributes
- **Advanced**: Explore :doc:`../tutorials/index` for advanced topics
- **API details**: See :doc:`../api/index` for full documentation
