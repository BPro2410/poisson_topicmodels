"""
Example 1: Getting Started with Topic Modeling

This example demonstrates how to load data, train a basic Poisson Factorization
model, and extract results.

Requirements:
    - numpy
    - scipy
    - jax
    - numpyro
"""

import numpy as np
import scipy.sparse as sparse
from topicmodels import PF

# ============================================================================
# STEP 1: Create or Load Data
# ============================================================================

print("Step 1: Preparing data")
print("-" * 50)

# Create a synthetic document-term matrix
# In practice, you would load real data from a CSV or database
D = 50  # Number of documents
V = 200  # Vocabulary size

# Create sparse document-term matrix (D x V)
# density=0.05 means 5% of entries are non-zero
counts = sparse.random(D, V, density=0.05, format="csr", dtype=np.float32)

# Create vocabulary
vocab = np.array([f"word_{i}" for i in range(V)])

print(f"✓ Created {D} documents with {V} vocabulary terms")
print(f"✓ Sparsity: {(1 - counts.nnz / (D * V)) * 100:.1f}%")
print()

# ============================================================================
# STEP 2: Initialize Model
# ============================================================================

print("Step 2: Initializing model")
print("-" * 50)

num_topics = 5
batch_size = 10

model = PF(
    counts=counts,
    vocab=vocab,
    num_topics=num_topics,
    batch_size=batch_size,
)

print(f"✓ Initialized Poisson Factorization model")
print(f"✓ Number of topics: {num_topics}")
print(f"✓ Batch size: {batch_size}")
print(f"✓ Model dimensions: {model.D} documents, {model.V} vocabulary terms")
print()

# ============================================================================
# STEP 3: Train Model
# ============================================================================

print("Step 3: Training model")
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
print(f"✓ Trained for {num_steps} steps with learning rate {learning_rate}")
print(f"✓ Final loss: {model.Metrics.loss[-1]:.4f}")
print(f"✓ Loss history: {len(model.Metrics.loss)} values tracked")
print()

# ============================================================================
# STEP 4: Extract Topics
# ============================================================================

print("Step 4: Extracting results")
print("-" * 50)

# Get document-topic assignments
topics, E_theta = model.return_topics()
print(f"✓ Document-topic assignments extracted")
print(f"✓ Shape of document-topic matrix (E_theta): {E_theta.shape}")

# Get top words per topic
top_words_per_topic = model.return_top_words_per_topic(n_words=10)
print(f"✓ Top words extracted for {len(top_words_per_topic)} topics")
print()

# ============================================================================
# STEP 5: Display Results
# ============================================================================

print("Step 5: Viewing results")
print("=" * 50)
print()

for topic_id, words in top_words_per_topic.items():
    print(f"Topic {topic_id}: {', '.join(words)}")

print()
print("=" * 50)
print(f"Document-topic matrix (E_theta) shape: {E_theta.shape}")
print(f"First document topic proportions:")
print(f"  {E_theta[0]}")
print()

# ============================================================================
# STEP 6: Reproducibility
# ============================================================================

print("Step 6: Demonstrating reproducibility")
print("-" * 50)

# Train another model with the same seed
model2 = PF(
    counts=counts,
    vocab=vocab,
    num_topics=num_topics,
    batch_size=batch_size,
)

params2 = model2.train_step(
    num_steps=num_steps,
    lr=learning_rate,
    random_seed=random_seed,  # Same seed!
)

# Compare losses
loss_diff = np.abs(np.array(model.Metrics.loss) - np.array(model2.Metrics.loss)).max()
print(f"✓ Trained second model with same seed")
print(f"✓ Maximum difference in loss values: {loss_diff:.2e}")

if loss_diff < 1e-5:
    print("✓ Results are reproducible with fixed seed!")
else:
    print("⚠ Results show some variation (expected for stochastic optimization)")

print()
print("=" * 50)
print("✓ Getting Started Example Complete!")
print("=" * 50)
