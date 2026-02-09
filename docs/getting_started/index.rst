.. _getting_started:

================================================================================
Getting Started
================================================================================

Welcome to poisson-topicmodels! This guide will get you up and running in about 5 minutes.

Quickstart: Your First Topic Model
===================================

Let's walk through a complete example from data preparation to model interpretation.

Step 1: Import Required Libraries
----------------------------------

.. code-block:: python

   import numpy as np
   from scipy.sparse import csr_matrix
   from poisson_topicmodels import PF

Step 2: Prepare Your Data
--------------------------

Topic models work with a **document-term matrix** (documents × vocabulary terms) and
a **vocabulary** list.

.. code-block:: python

   # Create sample data: 100 documents, 500 vocabulary terms
   # In practice, you would load your own text data
   np.random.seed(42)
   counts = csr_matrix(np.random.poisson(2, (100, 500)).astype(np.float32))
   vocab = np.array([f'word_{i}' for i in range(500)])

   print(f"Document-term matrix shape: {counts.shape}")
   print(f"Vocabulary size: {len(vocab)}")

**Data format**: The document-term matrix should be a sparse matrix (rows = documents,
columns = vocabulary terms) with non-negative integer counts.

Step 3: Initialize and Train the Model
---------------------------------------

.. code-block:: python

   # Create a Poisson Factorization model with 10 topics
   model = PF(
       counts=counts,
       vocab=vocab,
       num_topics=10,
       batch_size=32,
       random_seed=42
   )

   # Train for 100 iterations with learning rate 0.01
   params = model.train(
       num_iterations=100,
       learning_rate=0.01
   )

Step 4: Extract and Interpret Results
--------------------------------------

.. code-block:: python

   # Get topic-word distributions
   topics = model.get_topics()  # Shape: (vocab_size, num_topics)
   print(f"Topics shape: {topics.shape}")

   # Get top words for each topic
   top_words = model.get_top_words(n=10)
   print("\nTop 10 words per topic:")
   for topic_id, words in enumerate(top_words):
       print(f"Topic {topic_id}: {', '.join(words)}")

   # Get document-topic distributions
   doc_topics = model.get_document_topics()  # Shape: (num_docs, num_topics)
   print(f"\nDocument-topic matrix shape: {doc_topics.shape}")

   # Get topics for a specific document
   first_doc_topics = doc_topics[0]
   print(f"First document topic distribution: {first_doc_topics}")

Understanding the Output
=========================

**Topics** (``get_topics()``)
   Shape: (vocabulary_size, num_topics)

   Each column is a topic: probabilities of each word appearing in that topic.

**Document Topics** (``get_document_topics()``)
   Shape: (num_documents, num_topics)

   Each row is a document: probabilities of each topic in that document.

**Top Words** (``get_top_words(n)``):
   The n most likely words for each topic, useful for interpretation.

Next Steps
==========

Now that you have a working model, explore:

1. **Data Preparation**: :doc:`../how_to_guides/load_data`
   Learn how to prepare your own text data for topic modeling.

2. **Model Variants**: :doc:`../fundamentals/index`
   Explore other models like seeded PF (SPF) for guided discovery.

3. **Training & Configuration**: :doc:`../tutorials/index`
   Understand training options, hyperparameters, and GPU acceleration.

4. **Advanced Usage**: :doc:`../how_to_guides/index`
   Extract results, customize inference, and integrate with your pipeline.

Complete Example with Real-ish Data
====================================

Here's a more realistic example with synthetic documents that have meaningful structure:

.. code-block:: python

   import numpy as np
   from scipy.sparse import csr_matrix
   from poisson_topicmodels import PF

   # Create synthetic documents with 3 underlying topics
   np.random.seed(42)
   num_docs = 200
   num_words = 1000
   num_topics = 3

   # Define some "topic-specific" words
   topic_words = {
       0: list(range(0, 100)),      # Words 0-99 for topic 0
       1: list(range(100, 200)),    # Words 100-199 for topic 1
       2: list(range(200, 300)),    # Words 200-299 for topic 2
   }

   # Generate documents with topic structure
   counts_list = []
   for doc_id in range(num_docs):
       # Each document is a mixture of topics
       topic_dist = np.random.dirichlet([1] * num_topics)
       word_counts = np.zeros(num_words)

       for topic_id in range(num_topics):
           topic_weight = topic_dist[topic_id]
           words_in_topic = topic_words[topic_id]
           for _ in range(int(50 * topic_weight)):
               word_id = np.random.choice(words_in_topic)
               word_counts[word_id] += 1

       counts_list.append(word_counts)

   counts = csr_matrix(np.array(counts_list).astype(np.float32))
   vocab = np.array([f'word_{i}' for i in range(num_words)])

   # Train model with matching number of topics
   model = PF(counts, vocab, num_topics=3, batch_size=32, random_seed=42)
   params = model.train(num_iterations=100, learning_rate=0.01)

   # The model should discover the 3 underlying topics
   print("Discovered topics:")
   top_words = model.get_top_words(n=20)
   for topic_id, words in enumerate(top_words):
       print(f"\nTopic {topic_id}:")
       print(f"  {', '.join(words[:10])}")

Key Concepts
============

**Document-Term Matrix**
   The core input format: a sparse matrix where rows are documents and columns are
   vocabulary terms, containing word counts.

**Topics**
   Latent variables representing abstract themes. Each topic is a distribution over words.

**Topic Modeling**
   Statistical technique to discover and analyze latent topics in text data.

**Stochastic Variational Inference (SVI)**
   Efficient training method that processes documents in small batches (mini-batch training).

**GPU Acceleration**
   Computations run on GPU (if available) for significant speedups on large datasets.

Common Parameters
=================

**num_topics**: Number of topics to discover (usually 10-50 for start)

**batch_size**: Documents processed per training step (32-256 typical)

**num_iterations**: Training iterations (100-1000 typical)

**learning_rate**: Step size for optimization (0.001-0.1 typical, try 0.01 first)

**random_seed**: For reproducibility

Tips for Best Results
=====================

1. **Tune the Number of Topics**: Start with 10-20 topics, adjust based on coherence
2. **Use a Good Batch Size**: Larger batches (256+) for stability, smaller (32) for faster iterations
3. **Monitor Training**: Check that loss decreases smoothly
4. **Validate Topics**: Read top words to verify topics make sense
5. **Use GPU**: If available, GPU acceleration provides 10-100x speedup
6. **Set Random Seed**: For reproducibility in research

What's Next?
============

- **Read more examples**: See :doc:`../examples_guide/index`
- **Explore other models**: Check :doc:`../fundamentals/index` for seeded, covariate, and embedded variants
- **Learn advanced techniques**: Visit :doc:`../tutorials/index`
- **Check the API**: Refer to :doc:`../api/index` for detailed documentation

Having Issues?
==============

- Check :doc:`../installation/index` for installation troubleshooting
- Read :doc:`../how_to_guides/index` for common tasks
- Explore the full :doc:`../api/index` reference
