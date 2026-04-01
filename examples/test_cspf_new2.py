"""
Focused CSPF test with a single binary dummy covariate from Cat1.

Dummy rule:
    cat1::is_toys_games = 1 if Cat1 == "toys games", else 0

Usage:
    poetry run python examples/test_cspf_new2.py
"""

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.feature_extraction.text import CountVectorizer

from poisson_topicmodels import CSPF


def _build_keywords(vocab_set: set[str]) -> dict[str, list[str]]:
    keyword_candidates = {
        "pet supplies": ["dog", "cat", "litter", "dogs", "food"],
        "toys games": ["toy", "game", "play", "kids", "fun"],
        "beauty": ["hair", "skin", "scent", "dry", "products"],
        "baby products": ["baby", "diaper", "stroller", "bottles", "months"],
        "health personal care": ["razor", "shave", "shaver", "better", "work"],
        "grocery gourmet food": ["tea", "taste", "flavor", "coffee", "chocolate"],
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


# def main() -> None:
np.random.seed(42)

print("Loading ./data/10k_amazon.csv")
df = pd.read_csv(
    "/Users/bernd/Documents/01_Coding/02_GitHub/poisson_topicmodels/data/10k_amazon.csv"
)
df = df.dropna(subset=["Text", "Cat1"]).copy()

n_docs = min(2000, len(df))
df = df.sample(n=n_docs, random_state=42).reset_index(drop=True)

vectorizer = CountVectorizer(stop_words="english", min_df=5, max_features=5000)
counts = sparse.csr_matrix(vectorizer.fit_transform(df["Text"]), dtype=np.float32)
vocab = vectorizer.get_feature_names_out()
keywords = _build_keywords(set(vocab))

# Requested dummy coding:
# 1 for Cat1 == "toys games", 0 for all other categories.
x_design = pd.DataFrame(
    {"cat1::is_toys_games": (df["Cat1"].astype(str).str.lower() == "toys games").astype(np.float32)}
)

print(f"Documents: {counts.shape[0]}, Vocabulary: {counts.shape[1]}")
print(f"Dummy covariate mean (share of toys games): {x_design.iloc[:, 0].mean():.4f}")

model = CSPF(
    counts=counts,
    vocab=vocab,
    keywords=keywords,
    residual_topics=0,
    batch_size=min(256, counts.shape[0]),
    X_design_matrix=x_design,
)

params = model.train_step(num_steps=1200, lr=0.01, random_seed=7)

topics, e_theta = model.return_topics()
np.sum(topics == df.Cat1)
beta_df = model.return_beta()
effects_df = model.return_covariate_effects()

assert "lambda_location" in params, "Missing lambda_location in learned params"
assert len(topics) == counts.shape[0], "Topic assignment length mismatch"
assert e_theta.shape == (counts.shape[0], model.K), "Theta shape mismatch"
assert beta_df.shape == (counts.shape[1], model.K), "Beta shape mismatch"
assert effects_df.shape == (1, model.K), "Expected exactly one dummy covariate row"
assert np.isfinite(model.Metrics.loss[-1]), "Final loss is not finite"
assert np.isfinite(e_theta).all(), "Theta contains non-finite values"
assert np.isfinite(beta_df.values).all(), "Beta contains non-finite values"
assert np.isfinite(effects_df.values).all(), "Covariate effects contain non-finite values"

print(f"Final loss: {model.Metrics.loss[-1]:.4f}")
print(f"Theta shape: {e_theta.shape}")
print(f"Beta shape: {beta_df.shape}")
print(f"Covariate effects shape: {effects_df.shape}")
print("PASS")


# if __name__ == "__main__":
#     main()
