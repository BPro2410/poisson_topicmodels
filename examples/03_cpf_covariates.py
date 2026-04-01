"""
Example 3: Covariate Poisson Factorization (CPF)

This example demonstrates topic modeling with document-level covariates.
CPF allows you to incorporate external variables that influence topic distributions.

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

from poisson_topicmodels import CPF

# ============================================================================
# STEP 1: Create Data with Covariates
# ============================================================================

print("Step 1: Preparing data with document covariates")
print("-" * 50)

D = 50  # Documents
V = 200  # Vocabulary size
C = 3  # Number of covariates

# Create document-term matrix
counts = sparse.random(D, V, density=0.05, format="csr", dtype=np.float32)

# Create vocabulary
vocab = np.array([f"word_{i}" for i in range(V)])

# Create covariates (e.g., author demographics, temporal features, etc.)
# Could represent:
# - Author age, education level, expertise
# - Document publication time, medium
# - Geographic location
covariates = np.random.randn(D, C)

# Normalize covariates to [-1, 1] for interpretability
covariates = (
    2 * (covariates - covariates.min(axis=0)) / (covariates.max(axis=0) - covariates.min(axis=0))
    - 1
)

print(f"✓ Created {D} documents with {V} vocabulary terms")
print(f"✓ Added {C} document-level covariates")
print(f"✓ Covariate matrix shape: {covariates.shape}")
print()

# ============================================================================
# STEP 2: Use DataFrame for Covariates (Optional)
# ============================================================================

print("Step 2: Creating covariate DataFrame")
print("-" * 50)

# Create a DataFrame with meaningful covariate names
covariate_df = pd.DataFrame(
    covariates,
    columns=["author_expertise", "document_recency", "topic_specificity"],
)

print("✓ Created covariate DataFrame:")
print(covariate_df.head())
print()

# ============================================================================
# STEP 3: Initialize CPF Model
# ============================================================================

print("Step 3: Initializing CPF model")
print("-" * 50)

num_topics = 5
batch_size = 10

# You can pass either numpy array or DataFrame
model = CPF(
    counts=counts,
    vocab=vocab,
    num_topics=num_topics,
    batch_size=batch_size,
    X_design_matrix=covariate_df,  # Pass DataFrame with column names
)

print("✓ Initialized Covariate Poisson Factorization model")
print(f"✓ Number of topics: {num_topics}")
print(f"✓ Number of covariates: {model.C}")
print(f"✓ Model dimensions: {model.D} documents, {model.V} vocabulary terms")
print()

# ============================================================================
# STEP 4: Train Model
# ============================================================================

print("Step 4: Training CPF model")
print("-" * 50)

num_steps = 50
learning_rate = 0.01
random_seed = 42

params = model.train_step(
    num_steps=num_steps,
    lr=learning_rate,
    random_seed=random_seed,
)

print("✓ Training completed")
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

# Get covariate effects
covariate_effects = model.return_covariate_effects()
print("✓ Covariate effects extracted")
print(f"✓ Covariate effects shape: {covariate_effects.shape}")
print()

# ============================================================================
# STEP 6: Display Top Words per Topic
# ============================================================================

print("Step 6: Topic Interpretations")
print("=" * 50)
print()

# CPF inherits return_top_words_per_topic() from the base class
top_words = model.return_top_words_per_topic(n=10)
for topic_id, words in top_words.items():
    print(f"Topic {topic_id}: {', '.join(words)}")

print()

# ============================================================================
# STEP 7: Analyze Covariate Effects
# ============================================================================

print("=" * 50)
print("Step 7: Covariate Effects Analysis")
print("-" * 50)
print()

print("Covariate effects on topics (lambda matrix):")
covariate_effects = model.return_covariate_effects()
print(covariate_effects)
print()

# Get covariate effects with 95% credible intervals
print("Covariate effects with 95% credible intervals:")
ci_table = model.return_covariate_effects_ci(ci=0.95)
print(ci_table.to_string(index=False))
print()

print("Interpretation:")
print("  - Positive mean: covariate increases topic prevalence")
print("  - Negative mean: covariate decreases topic prevalence")
print("  - CI not crossing 0 suggests a credible effect")
print()

# ============================================================================
# STEP 8: Covariate Effect Forest Plot
# ============================================================================

print("=" * 50)
print("Step 8: Covariate Effect Forest Plot")
print("-" * 50)
print()

# Forest plot shows point estimates with credible intervals per topic
fig_cov, _ = model.plot_cov_effects(ci=0.95, save_path=None)
print("✓ Covariate effects forest plot generated")
print()

# ============================================================================
# STEP 9: Model Summary
# ============================================================================

print("=" * 50)
print("Step 9: Model Summary")
print("-" * 50)
print()

model.summary(n_top_words=5)
print()

# ============================================================================
# STEP 10: Analyze Document-Topic Distributions
# ============================================================================

print("=" * 50)
print("Step 10: Document-Topic Analysis")
print("-" * 50)
print()

print("Document-topic proportions influenced by covariates:")
print()

# Show correlation between covariate and topic prevalence
print("Example: First document's covariate values and topic proportions:")
print(f"  Covariates: {covariate_df.iloc[0].to_dict()}")
print(f"  Topic proportions: {E_theta[0]}")
print()

# ============================================================================
# STEP 11: Compare Different Covariate Scenarios
# ============================================================================

print("=" * 50)
print("Step 11: Simulating Different Covariate Scenarios")
print("-" * 50)
print()

# Create hypothetical documents with different covariate values
print("Expected topic distributions for different scenarios:")
print()

scenarios = {
    "High expertise, recent, specific": [1.0, 1.0, 1.0],
    "Low expertise, old, general": [-1.0, -1.0, -1.0],
    "Medium expertise, recent, general": [0.0, 1.0, -1.0],
}

for scenario_name, cov_values in scenarios.items():
    print(f"{scenario_name}: {cov_values}")

print()
print("Note: In practice, you would use the trained lambda matrix to predict")
print("topic distributions for new documents with specific covariate values.")
print()

print("=" * 50)
print("✓ Covariate Poisson Factorization Example Complete!")
print("=" * 50)
