"""
Example 4: Advanced - Covariate Seeded PF (CSPF)

This example demonstrates the most advanced model: combining both
guided topics (keywords) and document covariates.

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

# Try to import CSPF, fall back gracefully if not available
from poisson_topicmodels import CPF, CSPF, PF, SPF

# ============================================================================
# STEP 1: Create Data
# ============================================================================

print("Step 1: Preparing comprehensive dataset")
print("-" * 50)

D = 50  # Documents
V = 200  # Vocabulary size
C = 2  # Number of covariates

# Create document-term matrix
counts = sparse.random(D, V, density=0.05, format="csr", dtype=np.float32)

# Create vocabulary
vocab = np.array([f"word_{i}" for i in range(V)])

# Add some meaningful terms
vocab[0:5] = ["climate", "weather", "environment", "sustainability", "carbon"]
vocab[10:15] = ["economy", "trade", "market", "business", "profit"]
vocab[20:25] = ["technology", "software", "computer", "digital", "innovation"]

# Create covariates
covariates = np.random.randn(D, C)
covariates = (
    2 * (covariates - covariates.min(axis=0)) / (covariates.max(axis=0) - covariates.min(axis=0))
    - 1
)

covariate_df = pd.DataFrame(
    covariates,
    columns=["author_expertise", "document_recency"],
)

print(f"✓ Created {D} documents with {V} vocabulary terms")
print(f"✓ Added {C} document-level covariates")
print()

# ============================================================================
# STEP 2: Define Guided Topics
# ============================================================================

print("Step 2: Defining guided topics")
print("-" * 50)

keywords = {
    0: ["climate", "weather", "environment"],
    1: ["economy", "trade", "market"],
    2: ["technology", "software", "computer"],
}

print(f"✓ Defined {len(keywords)} guided topics")
print()

# ============================================================================
# STEP 3: Model Comparison
# ============================================================================

print("Step 3: Training and comparing models")
print("-" * 50)
print()

# Common parameters
num_steps = 50
learning_rate = 0.01
random_seed = 42

# --- Model 1: Unsupervised PF ---
print("Training Model 1: Unsupervised PF (baseline)")
print("  No guidance, no covariates")

pf_model = PF(
    counts=counts,
    vocab=vocab,
    num_topics=6,
    batch_size=10,
)

pf_params = pf_model.train_step(
    num_steps=num_steps,
    lr=learning_rate,
    random_seed=random_seed,
)

print(f"  ✓ Final loss: {pf_model.Metrics.loss[-1]:.4f}")
print()

# --- Model 2: Seeded SPF ---
print("Training Model 2: Seeded PF (with keywords)")
print("  Guided topics, no covariates")

spf_model = SPF(
    counts=counts,
    vocab=vocab,
    keywords=keywords,
    residual_topics=3,
    batch_size=10,
)

spf_params = spf_model.train_step(
    num_steps=num_steps,
    lr=learning_rate,
    random_seed=random_seed,
)

print(f"  ✓ Final loss: {spf_model.Metrics.loss[-1]:.4f}")
print()

# --- Model 3: Covariate CPF ---
print("Training Model 3: Covariate PF (with covariates)")
print("  No guidance, covariate-aware")

cpf_model = CPF(
    counts=counts,
    vocab=vocab,
    num_topics=6,
    batch_size=10,
    X_design_matrix=covariate_df,
)

cpf_params = cpf_model.train_step(
    num_steps=num_steps,
    lr=learning_rate,
    random_seed=random_seed,
)

print(f"  ✓ Final loss: {cpf_model.Metrics.loss[-1]:.4f}")
print()

# --- Model 4: Combined CSPF ---
print("Training Model 4: Combined CSPF (guided + covariate)")
print("  Guided topics AND covariate effects")

cspf_model = CSPF(
    counts=counts,
    vocab=vocab,
    keywords=keywords,
    residual_topics=3,
    batch_size=10,
    X_design_matrix=covariate_df,
)

cspf_params = cspf_model.train_step(
    num_steps=num_steps,
    lr=learning_rate,
    random_seed=random_seed,
)

print(f"  ✓ Final loss: {cspf_model.Metrics.loss[-1]:.4f}")
print()

# ============================================================================
# STEP 4: Loss Comparison
# ============================================================================

print("=" * 50)
print("Step 4: Model Performance Comparison")
print("-" * 50)
print()

print("Final Loss Values:")
print(f"  PF (baseline):        {pf_model.Metrics.loss[-1]:.4f}")
print(f"  SPF (with keywords):  {spf_model.Metrics.loss[-1]:.4f}")
print(f"  CPF (with covariates): {cpf_model.Metrics.loss[-1]:.4f}")
print(f"  CSPF (combined):      {cspf_model.Metrics.loss[-1]:.4f}")

print()
print("Interpretation:")
print("  - Lower loss indicates better model fit")
print("  - SPF/CPF/CSPF may have different losses due to different architectures")
print("  - Use domain knowledge to choose the right model for your task")
print()

# ============================================================================
# STEP 5: Extract and Compare Top Words
# ============================================================================

print("=" * 50)
print("Step 5: Topic Comparison")
print("-" * 50)
print()

print("Top words for first 3 topics (PF baseline):")
pf_top_words = pf_model.return_top_words_per_topic(n=5)
for topic_id in range(min(3, len(pf_top_words))):
    print(f"  Topic {topic_id}: {', '.join(pf_top_words[topic_id])}")

print()
print("Top words for first 3 topics (SPF with keywords):")
spf_top_words = spf_model.return_top_words_per_topic(n=5)
for topic_id in range(min(3, len(spf_top_words))):
    print(f"  Topic {topic_id}: {', '.join(spf_top_words[topic_id])}")

print()

# ============================================================================
# STEP 6: Advanced Analysis
# ============================================================================

print("=" * 50)
print("Step 6: Advanced Model Insights")
print("-" * 50)
print()

# CPF covariate effects
print("CPF Covariate Effects on Topics:")
cpf_covariate_effects = cpf_model.return_covariate_effects()
print(cpf_covariate_effects)
print()

# CPF covariate effects with credible intervals
print("CPF Covariate Effects with 90% Credible Intervals:")
cpf_effects_ci = cpf_model.return_covariate_effects_ci(ci=0.90)
print(cpf_effects_ci.head(10))  # Show first 10 rows
print()

# Forest plot of CPF covariate effects
cpf_fig, cpf_axes = cpf_model.plot_cov_effects(ci=0.90)
print("  ✓ Generated CPF covariate effects forest plot")
print()

# CSPF covariate effects (guided + covariate model)
print("CSPF Covariate Effects on Topics:")
cspf_covariate_effects = cspf_model.return_covariate_effects()
print(cspf_covariate_effects)
print()

print("CSPF Covariate Effects with 90% Credible Intervals:")
cspf_effects_ci = cspf_model.return_covariate_effects_ci(ci=0.90)
print(cspf_effects_ci.head(10))
print()

# Forest plot of CSPF covariate effects
cspf_fig, cspf_axes = cspf_model.plot_cov_effects(ci=0.90)
print("  ✓ Generated CSPF covariate effects forest plot")
print()

print("Interpretation:")
print("  - Rows: covariates (author_expertise, document_recency)")
print("  - Columns: topics")
print("  - Positive/negative values indicate effect direction")
print("  - Credible intervals that exclude zero suggest significant effects")
print()

# ============================================================================
# STEP 7: Model Summaries
# ============================================================================

print("=" * 50)
print("Step 7: Model Summaries")
print("-" * 50)
print()

print("--- PF Summary ---")
pf_model.summary()
print()

print("--- SPF Summary ---")
spf_model.summary()
print()

print("--- CPF Summary ---")
cpf_model.summary()
print()

print("--- CSPF Summary ---")
cspf_model.summary()
print()

# ============================================================================
# STEP 8: Best Practices
# ============================================================================

print("=" * 50)
print("Step 7: Model Selection Guide")
print("-" * 50)
print()

print("Choose PF (Poisson Factorization) when:")
print("  ✓ You want unsupervised topic discovery")
print("  ✓ You have no prior knowledge about topics")
print("  ✓ Computational efficiency is important")
print()

print("Choose SPF (Seeded PF) when:")
print("  ✓ You have domain knowledge about topics")
print("  ✓ You can define seed words for guidance")
print("  ✓ You want to encourage specific topic interpretations")
print()

print("Choose CPF (Covariate PF) when:")
print("  ✓ Topics vary with external variables")
print("  ✓ You have document-level metadata")
print("  ✓ You want to model covariate effects")
print()

print("Choose CSPF (Combined) when:")
print("  ✓ You have both domain knowledge AND metadata")
print("  ✓ You want maximum model flexibility")
print("  ✓ You have sufficient computational resources")
print()

# ============================================================================
# STEP 9: Reproducibility and Workflows
# ============================================================================

print("=" * 50)
print("Step 9: Reproducibility Best Practices")
print("-" * 50)
print()

print("Key Points:")
print("  ✓ Always use random_seed for reproducibility")
print("  ✓ Document all preprocessing steps")
print("  ✓ Keep track of hyperparameters (num_steps, learning_rate, etc.)")
print("  ✓ Validate results on held-out data")
print("  ✓ Compare multiple models before final selection")
print()

print("=" * 50)
print("✓ Advanced CSPF Example Complete!")
print("=" * 50)
