.. _tutorial_training:

================================================================================
Tutorial: Training Your First Topic Model
================================================================================

This tutorial covers the complete workflow for training a topic model and interpreting results.

**Duration**: ~15 minutes
**Level**: Beginner
**Prerequisites**: :doc:`../getting_started/index`

Step 1: Prepare Your Data
==========================

First, organize your text data into a document-term matrix.

.. code-block:: python

   import numpy as np
   from scipy.sparse import csr_matrix
   from poisson_topicmodels import PF

   # Option A: Load pre-processed data
   # For this tutorial, create synthetic data
   np.random.seed(42)

   # Document-term matrix: 500 documents, 1000 vocabulary
   num_docs, num_words = 500, 1000
   # Synthetic: ~10 words per document on average
   counts = csr_matrix(np.random.poisson(1, (num_docs, num_words)).astype(np.float32))

   # Vocabulary is list of words
   vocab = np.array([f'word_{i}' for i in range(num_words)])

   print(f"Dataset shape: {counts.shape}")
   print(f"Sparsity: {1 - counts.nnz / (counts.shape[0] * counts.shape[1]):.1%}")

Step 2: Initialize the Model
=============================

Create a topic model with initial configuration.

.. code-block:: python

   # Start with a reasonable number of topics
   num_topics = 15  # Adjust based on your data and goals

   model = PF(
       counts=counts,
       vocab=vocab,
       num_topics=num_topics,
       batch_size=64,  # 64 documents per training batch
       random_seed=42  # for reproducibility
   )

   print(f"Initialized PF model with {num_topics} topics")

Step 3: Train the Model
=======================

Run inference to learn topic and document-topic distributions.

.. code-block:: python

   # Train for 200 iterations with moderate learning rate
   params = model.train(
       num_iterations=200,
       learning_rate=0.01,
       verbose=True  # Print progress
   )

   print("Training complete!")

**Monitor training**: Watch the loss values. Should decrease steadily then plateau.

**Time expectations**:
- 500 docs × 1000 words: ~30 seconds on CPU, ~5 seconds on GPU
- Larger datasets: scale accordingly

Step 4: Extract Results
=======================

Get the learned topics and document-topic distributions.

.. code-block:: python

   # 1. Get topic-word distributions
   topics = model.get_topics()  # Shape: (vocab_size, num_topics)
   print(f"Topics shape: {topics.shape}")

   # 2. Get document-topic distributions
   doc_topics = model.get_document_topics()  # Shape: (num_docs, num_topics)
   print(f"Document-topic shape: {doc_topics.shape}")

   # 3. Get top words per topic
   top_words = model.get_top_words(n=15)
   print(f"Top words shape: {top_words.shape}")

Step 5: Interpret Topics
=========================

Human interpretation is crucial for evaluating topic quality.

.. code-block:: python

   # Display top words for each topic
   print("=" * 60)
   print("DISCOVERED TOPICS")
   print("=" * 60)

   for topic_id in range(num_topics):
       words = top_words[topic_id]
       print(f"\nTopic {topic_id}:")
       print(f"  Top words: {', '.join(words[:10])}")

   # Ask yourself:
   # - Do the top words form a coherent theme?
   # - Can you give each topic a human-readable label?
   # - Do any topics look like garbage?
   # - Are any topics too similar?

Step 6: Analyze Document-Topic Distribution
=============================================

Understand how topics are distributed across documents.

.. code-block:: python

   import matplotlib.pyplot as plt

   # 1. Topic distribution in specific documents
   doc_0_topics = doc_topics[0]
   print(f"\nDocument 0 topic distribution:")
   sorted_topics = np.argsort(doc_0_topics)[::-1]
   for i, topic_id in enumerate(sorted_topics[:5]):
       intensity = doc_0_topics[topic_id]
       print(f"  Topic {topic_id}: {intensity:.3f}")

   # 2. Overall topic prevalence in corpus
   avg_topics = doc_topics.mean(axis=0)
   plt.figure(figsize=(12, 4))
   plt.bar(range(num_topics), avg_topics)
   plt.xlabel('Topic ID')
   plt.ylabel('Average Intensity')
   plt.title('Topic Prevalence in Corpus')
   plt.show()

   # 3. Find most interesting documents
   # (documents with strong topic concentration)
   doc_entropy = -np.sum(doc_topics * np.log(doc_topics + 1e-10), axis=1)
   focused_docs = np.argsort(doc_entropy)[:5]
   scattered_docs = np.argsort(doc_entropy)[-5:]

   print(f"\nMost focused documents (highest topic concentration): {focused_docs}")
   print(f"Most scattered documents (most mixed): {scattered_docs}")

Step 7: Advanced Analysis
=========================

Deeper exploration of results:

.. code-block:: python

   # 1. Topic similarity
   # Are any topics too similar? Compute pairwise similarity
   from sklearn.metrics.pairwise import cosine_similarity

   topic_similarity = cosine_similarity(topics.T)
   np.fill_diagonal(topic_similarity, 0)  # Remove self-similarity

   # Find most similar topic pairs
   for _ in range(3):
       i, j = np.unravel_index(topic_similarity.argmax(), topic_similarity.shape)
       if topic_similarity[i, j] > 0:
           print(f"Topic {i} and {j} are similar (sim={topic_similarity[i, j]:.3f})")
       topic_similarity[i, j] = 0

   # 2. Topic specialization
   # How many topics does each document use mainly?
   doc_dominance = (doc_topics.max(axis=1) / doc_topics.sum(axis=1))
   print(f"\nAverage document topic dominance: {doc_dominance.mean():.3f}")
   print(f"  → Values close to 1: documents focus on few topics")
   print(f"  → Values close to {1/num_topics:.3f}: documents spread across topics")

Step 8: Quality Metrics
=======================

Evaluate model quality programmatically.

.. code-block:: python

   # Coherence: do top words of a topic correlate?
   coherence = model.compute_coherence()
   print(f"Topic coherence (per topic):")
   print(f"  Mean: {coherence.mean():.3f}")
   print(f"  Std: {coherence.std():.3f}")
   print(f"  Range: [{coherence.min():.3f}, {coherence.max():.3f}]")

   # Which topics are most coherent?
   best_topics = np.argsort(coherence)[-5:]
   worst_topics = np.argsort(coherence)[:5]
   print(f"\nMost coherent topics: {best_topics}")
   print(f"Least coherent topics: {worst_topics}")

Next: Validation and Optimization
==================================

Your trained model is done! Now consider:

1. **Validate model quality** → :doc:`tutorial_validation`
2. **Optimize hyperparameters** → :doc:`tutorial_hyperparameters`
3. **Scale to bigger data** → :doc:`tutorial_gpu`
4. **Solve specific problems** → :doc:`../how_to_guides/index`

Quick Checklist
===============

✓ Data loaded and formatted as document-term matrix
✓ Model initialized with reasonable parameters
✓ Training completed and loss decreased
✓ Topics extracted and interpreted
✓ Document-topic distributions explored
✓ Quality metrics computed

What's Next?

- **Improve results**: Try :doc:`tutorial_validation` to assess quality
- **Lots of data?**: See :doc:`tutorial_gpu` for GPU acceleration
- **Fine-tune model**: Read :doc:`tutorial_hyperparameters`
- **Specific task?**: Browse :doc:`../how_to_guides/index`

Common Issues
=============

**Q: Loss isn't decreasing**
A: Try higher learning rate (0.05-0.1) or reduce batch size

**Q: Topics look random**
A: You may need more topics or more training iterations

**Q: Training is really slow**
A: Use GPU (see :doc:`tutorial_gpu`) or reduce vocabulary size

**Q: Memory error**
A: Reduce batch_size or use sparse matrix format

See :doc:`../fundamentals/index` for more details on each model.
