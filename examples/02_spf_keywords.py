"""
Example 2: Seeded Poisson Factorization (SPF)

This example demonstrates guided topic modeling using keywords (seed words).
SPF allows you to guide the model to discover topics related to specific concepts.

Requirements:
    - numpy
    - scipy
    - jax
    - numpyro
"""

import numpy as np
import scipy.sparse as sparse
from topicmodels import SPF

# ============================================================================
# STEP 1: Create Data
# ============================================================================

print("Step 1: Preparing data with domain-specific terms")
print("-" * 50)

D = 50  # Documents
V = 200  # Vocabulary size

# Create synthetic document-term matrix
counts = sparse.random(D, V, density=0.05, format="csr", dtype=np.float32)

# Create vocabulary with some recognizable terms
vocab = np.array([f"word_{i}" for i in range(V)])

# For this example, let's create some meaningful vocabulary
vocab[0:5] = ["climate", "weather", "temperature", "environment", "sustainability"]
vocab[10:15] = ["economy", "trade", "business", "market", "investment"]
vocab[20:25] = ["research", "science", "study", "experiment", "discovery"]
vocab[30:35] = ["technology", "software", "computer", "digital", "innovation"]

print(f"✓ Created {D} documents with {V} vocabulary terms")
print(f"✓ Vocabulary includes domain-specific terms")
print()

# ============================================================================
# STEP 2: Define Keywords (Seed Words)
# ============================================================================

print("Step 2: Defining seed words for guided topics")
print("-" * 50)

# Define keywords for guided topics
# These tell the model which words should be emphasized for each topic
keywords = {
    0: ["climate", "weather", "temperature", "environment"],
    1: ["economy", "trade", "business", "market"],
    2: ["research", "science", "study", "experiment"],
    3: ["technology", "software", "computer", "digital"],
}

print(f"✓ Defined {len(keywords)} guided topics:")
for topic_id, words in keywords.items():
    print(f"  Topic {topic_id}: {', '.join(words)}")

print()

# ============================================================================
# STEP 3: Initialize SPF Model
# ============================================================================

print("Step 3: Initializing SPF model")
print("-" * 50)

residual_topics = 1  # Additional unsupervised topics
batch_size = 10

model = SPF(
    counts=counts,
    vocab=vocab,
    keywords=keywords,
    residual_topics=residual_topics,
    batch_size=batch_size,
)

print(f"✓ Initialized Seeded Poisson Factorization model")
print(f"✓ Seeded topics: {len(keywords)}")
print(f"✓ Residual (unsupervised) topics: {residual_topics}")
print(f"✓ Total topics (K): {model.K}")
print()

# ============================================================================
# STEP 4: Train Model
# ============================================================================

print("Step 4: Training SPF model")
print("-" * 50)

num_steps = 50
learning_rate = 0.01
random_seed = 42

params = model.train_step(
    num_steps=num_steps,
    lr=learning_rate,
    random_seed=random_seed,
)

print(f"✓ Training completed")
print(f"✓ Final loss: {model.Metrics.loss[-1]:.4f}")
print()

# ============================================================================
# STEP 5: Extract and Display Results
# ============================================================================

print("Step 5: Extracting results")
print("-" * 50)

# Get topics
topics, E_theta = model.return_topics()

# Get beta matrix (topic-word distributions)
beta = model.return_beta()

# Get top words per topic
top_words = model.return_top_words_per_topic(n_words=10)

print(f"✓ Extracted results for {model.K} topics")
print()

# ============================================================================
# STEP 6: Display Top Words for Each Topic
# ============================================================================

print("Step 6: Topic Interpretations")
print("=" * 50)
print()

for topic_id in range(model.K):
    if topic_id < len(keywords):
        print(f"Guided Topic {topic_id}:")
        print(f"  Seed words: {', '.join(keywords[topic_id])}")
    else:
        print(f"Residual Topic {topic_id - len(keywords)}:")

    if topic_id in top_words:
        print(f"  Top words learned: {', '.join(top_words[topic_id][:5])}")
    print()

# ============================================================================
# STEP 7: Analyze Document-Topic Distributions
# ============================================================================

print("=" * 50)
print("Step 7: Document-Topic Analysis")
print("-" * 50)
print()

print(f"Document-topic matrix shape: {E_theta.shape}")
print(f"Number of documents: {E_theta.shape[0]}")
print(f"Number of topics: {E_theta.shape[1]}")
print()

# Show first 5 documents
print("Document-topic proportions (first 5 documents):")
for doc_id in range(min(5, D)):
    print(f"  Doc {doc_id}: {E_theta[doc_id]}")

print()

# ============================================================================
# STEP 8: Compare with Unsupervised PF
# ============================================================================

print("=" * 50)
print("Step 8: Comparison with Unsupervised Poisson Factorization")
print("-" * 50)
print()

from topicmodels import PF

# Train unsupervised model with same parameters
pf_model = PF(
    counts=counts,
    vocab=vocab,
    num_topics=model.K,  # Same number of topics as SPF
    batch_size=batch_size,
)

pf_params = pf_model.train_step(
    num_steps=num_steps,
    lr=learning_rate,
    random_seed=random_seed,
)

pf_top_words = pf_model.return_top_words_per_topic(n_words=10)

print("Guided SPF topics are influenced by seed words:")
for topic_id in range(min(3, len(keywords))):
    seed_words = set(keywords[topic_id])
    learned_words = set(top_words.get(topic_id, [])[:5])
    overlap = seed_words & learned_words
    print(f"  Topic {topic_id}: {len(overlap)} seed words in top 5 learned words")

print()
print("=" * 50)
print("✓ Seeded Poisson Factorization Example Complete!")
print("=" * 50)
