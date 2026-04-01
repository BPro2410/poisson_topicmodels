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
   params = model.train_step(
       num_steps=200,
       lr=0.01
   )

   # Extract results
   categories, e_theta = model.return_topics()       # dominant topic + proportions
   beta = model.return_beta()                          # word-topic DataFrame
   top_words = model.return_top_words_per_topic(n=10)  # top words per topic

Interpreting Results
====================

**Word-Topic Matrix (β)**:

Access via ``model.return_beta()`` — a ``pd.DataFrame`` (vocab_size × num_topics)

Each column k represents topic k:

.. code-block:: python

   beta = model.return_beta()
   topic_5 = beta.iloc[:, 5]  # Topic 5 weights over vocabulary

Interpretation:
- Higher weight means the word is more associated with the topic
- Top words with highest weights characterize the topic

**Document Topics (θ)**:

Access via ``model.return_topics()`` — returns ``(categories, E_theta)``

.. code-block:: python

   categories, e_theta = model.return_topics()
   # categories: dominant topic per document
   # e_theta: full document-topic matrix (num_docs × num_topics)

   doc_3_topics = e_theta[3, :]  # Topic mixture for document 3
   dominant = categories[3]       # Dominant topic for document 3

Interpretation:
- ``categories`` gives the argmax topic for each document
- ``e_theta[d, k]`` is the intensity of topic k in document d

**Top Words**:

Access via ``model.return_top_words_per_topic(n=10)``

.. code-block:: python

   top_words = model.return_top_words_per_topic(n=10)
   # dict: {topic_id: [word1, word2, ...]}

   print(top_words[2])
   # ['research', 'data', 'experiment', 'analysis', ...]

Human Interpretation:
- Read top ~10-20 words for each topic
- Does the topic make sense thematically?
- Can you give it a meaningful label?

Example Interpretation Workflow
===============================

.. code-block:: python

   model = PF(counts, vocab, num_topics=5, batch_size=32)
   model.train_step(num_steps=200, lr=0.01)

   # Examine each topic
   top_words = model.return_top_words_per_topic(n=15)

   for topic_id, words in top_words.items():
       print(f"\n=== Topic {topic_id} ===")
       print(f"Top words: {', '.join(words)}")

       # Find documents dominated by this topic
       categories, e_theta = model.return_topics()
       top_docs = np.argsort(e_theta[:, topic_id])[-3:]
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
       model.train_step(num_steps=200, lr=0.01)
       # Evaluate quality (e.g., via coherence or manual inspection)

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

   params = model.train_step(
       num_steps=200,
       lr=0.01,
   )
   # Then inspect: model.plot_model_loss()

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

   params = model.train_step(num_steps=200, lr=0.01, random_seed=42)
   # Same seed → same results (good for research)

**Progress Monitoring**: Check loss trajectory

.. code-block:: python

   model.train_step(num_steps=200, lr=0.01)
   model.plot_model_loss()  # visualize loss curve

**Early stopping**: Stop if loss plateaus

.. code-block:: python

   # Check loss after training
   model.train_step(num_steps=200, lr=0.01)
   model.plot_model_loss()  # visually inspect convergence
   # If loss hasn't plateaued, train more steps

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
- **Topic diversity**: Are topics distinct?

Example:

.. code-block:: python

   coherence_df = model.compute_topic_coherence()
   print(f"Average coherence: {coherence_df['coherence'].mean():.3f}")

   diversity = model.compute_topic_diversity()
   print(f"Topic diversity: {diversity:.3f}")

   # Built-in visualizations
   model.plot_model_loss()          # Training loss curve
   model.plot_topic_prevalence()    # Topic prevalence bar chart
   model.plot_topic_correlation()   # Topic similarity heatmap

Next Steps
==========

- **Add guidance**: Use :doc:`seeded_models` to incorporate domain knowledge
- **Model metadata**: Try :doc:`covariate_models` if you have document attributes
- **Advanced**: Explore :doc:`../tutorials/index` for advanced topics
- **API details**: See :doc:`../api/index` for full documentation
