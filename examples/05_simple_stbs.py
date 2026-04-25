"""
Example 5: Structured Topic Belief Scaling (STBS)

This example demonstrates topic modeling with author-level ideology estimation
and author-level covariates using the STBS model on randomly generated data.

Requirements:
    - numpy
    - scipy
    - pandas
    - jax
    - numpyro
"""

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import jax.numpy as jnp
import matplotlib.pyplot as plt

from poisson_topicmodels import STBS


# ============================================================================
# STEP 1: Generate Random Data
# ============================================================================

print("Step 1: Generating random data")
print("-" * 50)

D = 300   # Documents
V = 500   # Vocabulary size
A = 30    # Unique authors

# Create document-term matrix
counts = sparse.random(D, V, density=0.05, format="csr", dtype=np.float32)

# Create vocabulary
vocab = np.array([f"word_{i}" for i in range(V)])

# Create author names (MUST be alphabetically sorted)
author_names = np.array(sorted([f"Author_{i:02d}" for i in range(A)]))

# Assign each document to a random author (document-level, length D)
authors_doc = np.random.choice(author_names, size=D)

print(f"✓ Document-term matrix: {counts.shape}")
print(f"✓ Vocabulary: {len(vocab)} terms")
print(f"✓ Documents: {D}, Authors: {A}")
print()


# ============================================================================
# STEP 2: Generate Author-Level Covariates
# ============================================================================

print("Step 2: Generating author-level covariates")
print("-" * 50)

# Create political party covariates: R, D, or I (where I is rare)
party   = np.random.choice(["R", "D", "I"], size=A, p=[0.45, 0.45, 0.10])
party_r = (party == "R").astype(int)
party_i = (party == "I").astype(int)

# Create a gender covariate
gender   = np.random.choice(["M", "F"], size=A, p=[0.5, 0.5])
gender_f = (gender == "F").astype(int)

# Create an experience covariate
experience = np.random.randint(1, 30, size=A)

# Create the covariate dataFrame: one row per author, indexed by name, and alphabetically sorted
covariate_df = pd.DataFrame({
    "party_r":    party_r,
    "party_i":    party_i,
    "gender_f":   gender_f,
    "experience": experience,}, 
    index=author_names)

print(f"✓ Covariate matrix shape: {covariate_df.shape}")
print(f"✓ Covariates: {covariate_df.columns.tolist()}")
print(f"✓ Party distribution: R={party_r.sum()}, D={(party_r+party_i==0).sum()}, I={party_i.sum()}")
print(f"✓ Gender distribution: F={gender_f.sum()}, M={(gender_f==0).sum()}")
print(covariate_df.head())
print()


# ============================================================================
# STEP 3: Initialise Ideology Prior (i_mu_init)
# ============================================================================

print("Step 3: Initialising ideology prior")
print("-" * 50)

# If there exists a prior assumption of the ideology, initialize accordingly. 
i_mu_init = np.select(
    [(covariate_df["party_i"] == 0) & (covariate_df["party_r"] == 1),   # Republican
     (covariate_df["party_i"] == 1) & (covariate_df["party_r"] == 0),   # Independent
     (covariate_df["party_i"] == 0) & (covariate_df["party_r"] == 0),],   # Democrat
    [1, 0, -1],
    default=0,)

i_mu_init = jnp.array(i_mu_init, dtype=jnp.float32)

print(f"✓ i_mu_init shape: {i_mu_init.shape}")
print()


# ============================================================================
# STEP 4: Initialise STBS Model
# ============================================================================

print("Step 4: Initialising STBS model")
print("-" * 50)

num_topics = 10
batch_size = 64

model = STBS(
    counts=counts,
    vocab=vocab,
    num_topics=num_topics,
    batch_size=batch_size,
    authors=authors_doc,  
    X_design_matrix=covariate_df,  
    i_mu_init=i_mu_init,     
)

print(f"✓ Initialised STBS model")
print(f"✓ Topics: {num_topics}")
print(f"✓ Documents: {model.D}, Vocabulary: {model.V}")
print(f"✓ Authors: {model.N}, Covariates: {len(model.covariates)}")
print()


# ============================================================================
# STEP 5: Train Model
# ============================================================================

print("Step 5: Training STBS model")
print("-" * 50)

params = model.train_step(
    num_steps=500,
    lr=0.01,
)

print(f"✓ Training completed")
print(f"✓ Final loss: {model.Metrics.loss[-1]:.4f}")
print()


# ============================================================================
# STEP 5: Extract Results
# ============================================================================

print("Step 5: Extracting results")
print("-" * 50)

# Get document-topic assignments
topics, E_theta = model.return_topics()
print("✓ Document-topic assignments extracted")
print(f"✓ Shape of document-topic matrix: {E_theta.shape}")

# Get beta matrix (topic-word distributions)
beta = model.return_beta()
print("✓ Beta matrix extracted")
print(f"✓ Beta shape: {beta.shape}")

# Get ideological points
ideal_points = model.return_ideal_points()
print("✓ Ideal points extracted")
print(f"✓ Ideal points shape: {ideal_points.shape}")
print()

# Get covariate effects (on ideology)
covariate_effects = model.return_ideal_covariates()
print("✓ Covariate effects extracted")
print(f"✓ Covariate effects shape: {covariate_effects.shape}")
print()


# ============================================================================
# STEP 6: Top Words Per Topic
# ============================================================================

print("Step 6: Topic interpretations")
print("-" * 50)

top_words = model.return_top_words_per_topic(n=10)
for topic_id, words in top_words.items():
    print(f"Topic {topic_id}: {', '.join(words)}")
print()


# ============================================================================
# STEP 7: Visualise Topic Prevalence
# ============================================================================

print("Step 7: Topic prevalence")
print("-" * 50)

fig, ax = model.plot_topic_prevalence(sort=True)
print("✓ Topic prevalence bar chart generated")
print()


# ============================================================================
# STEP 8: Visualise Topic Word Clouds
# ============================================================================

print("Step 7: Topic Word Clouds")
print("-" * 50)

fig, ax = model.plot_topic_wordclouds(ideology_values=None)
print("✓ Topic word clouds generated")
print()

fig, ax = model.plot_topic_wordclouds(topics=[0,2,4], ideology_values=(-1, 0, 1), log_corrected=True)
print("✓ Topic (subsetted) word clouds (incl. ideology) generated")
print()


# ============================================================================
# STEP 9: Visualise Author Ideology
# ============================================================================

print("Step 8: Author ideology dot plot")
print("-" * 50)

group_labels  = {-1: "D", 0: "I", 1: "R"}
group_palette = {"D": "dodgerblue", "I": "gray", "R": "red"}

fig, ax = model.plot_ideol_points(
    group_var=np.array(i_mu_init),
    group_labels=group_labels,
    group_palette=group_palette,
)
print("✓ Ideology dot plot generated")
print()


# ============================================================================
# STEP 10: Author-Topic Heatmap
# ============================================================================

print("Step 9: Author-topic heatmap")
print("-" * 50)

fig, ax = model.plot_author_topic_heatmap()
print("✓ Author-topic heatmap generated")
print()


# ============================================================================
# STEP 11: Covariate Ideology Coefficients (iota)
# ============================================================================

print("Step 11: Covariate ideology coefficients")
print("-" * 50)

fig, ax = model.plot_iota_credible_intervals()
print("✓ Iota credible interval plot generated")
print()


# ============================================================================
# STEP 12: Model Summary
# ============================================================================

print("=" * 50)
print("Step 9: Model Summary")
print("-" * 50)
print()

print(model.summary(n_top_words=5))
print()