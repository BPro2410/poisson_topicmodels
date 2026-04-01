"""
CSPF2 test with two grouped categorical covariates on 10k Amazon data.

Covariate construction:
    animal::dog / animal::cat
        – assigned to 80% of "pet supplies" docs and 10% of "toys games" docs
        – remaining docs get all-zero (no animal category)
    eating::ice cream / eating::banana
        – assigned randomly to 50% of all docs each

This way we have 4 categories in 2 groups, with some correlation to the underlying topics (e.g. "pet supplies" should correlate with animal covariates).

Usage:
    poetry run python examples/test_amazon_grouped.py
"""

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.feature_extraction.text import CountVectorizer

from poisson_topicmodels.models.CSPF2 import CSPF2


def _build_keywords(vocab_set: set[str]) -> dict[str, list[str]]:
    pets = ["dog", "cat", "litter", "cats", "dogs", "food", "box", "collar", "water", "pet"]
    toys = ["toy", "game", "play", "fun", "old", "son", "year", "loves", "kids", "daughter"]
    beauty = [
        "hair",
        "skin",
        "product",
        "color",
        "scent",
        "smell",
        "used",
        "dry",
        "using",
        "products",
    ]
    baby = [
        "baby",
        "seat",
        "diaper",
        "diapers",
        "stroller",
        "bottles",
        "son",
        "pump",
        "gate",
        "months",
    ]
    health = [
        "product",
        "like",
        "razor",
        "shave",
        "time",
        "day",
        "shaver",
        "better",
        "work",
        "years",
    ]
    grocery = [
        "tea",
        "taste",
        "flavor",
        "coffee",
        "sauce",
        "chocolate",
        "sugar",
        "eat",
        "sweet",
        "delicious",
    ]

    keyword_candidates = {
        "pet supplies": pets,
        "toys games": toys,
        "beauty": beauty,
        "baby products": baby,
        "health personal care": health,
        "grocery gourmet food": grocery,
    }

    keywords: dict[str, list[str]] = {}
    for topic, words in keyword_candidates.items():
        in_vocab = [w for w in words if w in vocab_set]
        if len(in_vocab) >= 2:
            keywords[topic] = in_vocab

    if len(keywords) < 3:
        raise ValueError(
            "Not enough keyword groups found in vocabulary. Check vectorizer settings."
        )

    return keywords


# ============================================================================
# Load and vectorize
# ============================================================================

np.random.seed(42)
rng = np.random.RandomState(42)

print("Loading ./data/10k_amazon.csv")
df = pd.read_csv(
    "/Users/Q610748/Documents/01_Coding/02_self/poisson_topicmodels/data/10k_amazon.csv"
)
df = df.dropna(subset=["Text", "Cat1"]).copy()

n_docs = min(10000, len(df))
df = df.sample(n=n_docs, random_state=42).reset_index(drop=True)

vectorizer = CountVectorizer(stop_words="english", min_df=5, max_features=5000)
counts = sparse.csr_matrix(vectorizer.fit_transform(df["Text"]), dtype=np.float32)
vocab = vectorizer.get_feature_names_out()
keywords = _build_keywords(set(vocab))

print(f"Documents: {counts.shape[0]}, Vocabulary: {counts.shape[1]}")
print()

# ============================================================================
# Build covariates
# ============================================================================

cat1_lower = df["Cat1"].astype(str).str.lower()
n = len(df)

# --- Covariate group 1: animal (dog / cat) ---
# 80% of "pet supplies" docs get a random animal label, 10% of "toys games".
animal = pd.Series([""] * n, dtype=str)

for i in range(n):
    if cat1_lower.iloc[i] == "pet supplies" and rng.rand() < 0.80:
        animal.iloc[i] = rng.choice(["dog", "cat"])
    elif cat1_lower.iloc[i] == "toys games" and rng.rand() < 0.10:
        animal.iloc[i] = rng.choice(["dog", "cat"])

# One-hot encode with group prefix "animal::".
animal_dog = (animal == "dog").astype(np.float32)
animal_cat = (animal == "cat").astype(np.float32)

# --- Covariate group 2: eating (ice cream / banana) ---
# Each observation gets exactly one of the two labels with 50/50 chance.
eating_choice = rng.choice(["ice cream", "banana"], size=n)
eating_icecream = (eating_choice == "ice cream").astype(np.float32)
eating_banana = (eating_choice == "banana").astype(np.float32)

x_design = pd.DataFrame(
    {
        "animal::dog": animal_dog,
        "animal::cat": animal_cat,
        "eating::ice cream": eating_icecream,
        "eating::banana": eating_banana,
    }
)

print("Covariate summary")
print("-" * 50)
print(f"  Covariate matrix shape: {x_design.shape}")
print(f"  animal::dog  mean: {animal_dog.mean():.4f}")
print(f"  animal::cat  mean: {animal_cat.mean():.4f}")
print(f"  eating::ice cream mean: {eating_icecream.mean():.4f}")
print(f"  eating::banana    mean: {eating_banana.mean():.4f}")
print()

# ============================================================================
# Train CSPF2
# ============================================================================

print("Initializing CSPF2 model")
print("-" * 50)

model = CSPF2(
    counts=counts,
    vocab=vocab,
    keywords=keywords,
    residual_topics=0,
    batch_size=min(1024, counts.shape[0]),
    X_design_matrix=x_design,
)

print(f"  Topics (K): {model.K}")
print(f"  Covariates (C): {model.C}")
print(f"  Covariate groups (G): {model.G}")
print()

print("Training ...")
params = model.train_step(num_steps=1000, lr=0.1, random_seed=7)

# ============================================================================
# Evaluate
# ============================================================================

topics, e_theta = model.return_topics()
np.sum(df.Cat1 == topics) / len(topics)
beta_df = model.return_beta()
effects_df = model.return_covariate_effects()

accuracy = np.sum(topics == df["Cat1"].values) / len(topics)

assert "lambda_location" in params, "Missing lambda_location in learned params"
assert len(topics) == counts.shape[0], "Topic assignment length mismatch"
assert e_theta.shape == (counts.shape[0], model.K), "Theta shape mismatch"
assert beta_df.shape == (counts.shape[1], model.K), "Beta shape mismatch"
assert effects_df.shape == (4, model.K), "Expected 4 covariate rows"
assert np.isfinite(model.Metrics.loss[-1]), "Final loss is not finite"
assert np.isfinite(e_theta).all(), "Theta contains non-finite values"
assert np.isfinite(beta_df.values).all(), "Beta contains non-finite values"
assert np.isfinite(effects_df.values).all(), "Covariate effects contain non-finite values"

print()
print("Results")
print("=" * 50)
print(f"  Final loss: {model.Metrics.loss[-1]:.4f}")
print(f"  Topic accuracy (vs Cat1): {accuracy:.2%}")
print(f"  Theta shape: {e_theta.shape}")
print(f"  Beta shape: {beta_df.shape}")
print(f"  Covariate effects shape: {effects_df.shape}")
print()

print("Covariate effects (lambda):")
print(effects_df.to_string())
print()

# ============================================================================
# Group importance via shrinkage parameters
# ============================================================================
# The model uses a global-local shrinkage prior (equations 8-9 in the spec):
#   tau_k^2     -- global (per-topic) shrinkage
#   delta_{gk}^2 -- local (per-group, per-topic) shrinkage
#
# Large E[delta_{gk}^2] = shape/rate  means group g is IMPORTANT for topic k
# (the prior lets the coefficients remain large).
# Small E[delta_{gk}^2] means the group is shrunk toward zero = NOT important.

topic_names = list(keywords.keys())
group_names = ["animal", "eating"]

# E[tau_k^2] -- global per-topic scale
E_tau2 = params["tau2_shape"] / params["tau2_rate"]  # shape (K,)
tau2_df = pd.DataFrame(np.array(E_tau2)[None, :], index=["E[tau²]"], columns=topic_names)

# E[delta_{gk}^2] -- local group-topic scale
E_delta2 = params["delta2_shape"] / params["delta2_rate"]  # shape (G, K)
delta2_df = pd.DataFrame(np.array(E_delta2), index=group_names, columns=topic_names)

# Effective group variance = tau_k^2 * delta_{gk}^2  (higher = more important)
effective_var = np.array(E_tau2)[None, :] * np.array(E_delta2)
eff_var_df = pd.DataFrame(effective_var, index=group_names, columns=topic_names)

print("Global shrinkage E[tau²_k] (per topic):")
print(tau2_df.to_string())
print()
print("Local group shrinkage E[delta²_{gk}] (per group × topic):")
print(delta2_df.to_string())
print()
print("Effective group variance tau²_k × delta²_{gk} (higher = more important):")
print(eff_var_df.to_string())
print()

# Quick summary: which group matters most per topic?
print("Most important group per topic (by effective variance):")
for topic in topic_names:
    best_group = eff_var_df[topic].idxmax()
    val = eff_var_df[topic].max()
    print(f"  {topic}: {best_group} ({val:.4f})")
print()

print("Top 10 words per topic:")
top_words = model.return_top_words_per_topic(n=10)
for topic_id, words in top_words.items():
    print(f"  {topic_id}: {', '.join(words)}")

print()
print("PASS")
