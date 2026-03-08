"""
Integration test script for the new CSPF2 implementation.

Runs two scenarios on ./data/10k_amazon.csv:
1. Grouped one-hot covariates only.
2. Grouped one-hot covariates + multiple numeric covariates.

Usage:
    python3 examples/test_cspf_new.py
"""

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.feature_extraction.text import CountVectorizer

from poisson_topicmodels.models.CSPF2 import CSPF2


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


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def _prepare_data():
    print("Loading ./data/10k_amazon.csv")
    df = pd.read_csv("./data/10k_amazon.csv")
    df = df.dropna(subset=["Text"]).copy()

    # Keep runtime low while still testing grouped design and SVI behavior.
    n_docs = min(2000, len(df))
    df = df.sample(n=n_docs, random_state=42).reset_index(drop=True)

    vectorizer = CountVectorizer(stop_words="english", min_df=5, max_features=5000)
    counts = sparse.csr_matrix(vectorizer.fit_transform(df["Text"]), dtype=np.float32)
    vocab = vectorizer.get_feature_names_out()

    print(f"Documents: {counts.shape[0]}, Vocabulary: {counts.shape[1]}")
    return df, counts, vocab


def _grouped_onehot_covariates(df: pd.DataFrame) -> pd.DataFrame:
    # Use explicit group separators ("::") so CSPF2 can infer group membership.
    cat1 = pd.get_dummies(df["Cat1"].fillna("unknown"), prefix="cat1", prefix_sep="::")
    cat2 = pd.get_dummies(df["Cat2"].fillna("unknown"), prefix="cat2", prefix_sep="::")
    x = pd.concat([cat1, cat2], axis=1).astype(np.float32)
    return x


def _mixed_covariates(df: pd.DataFrame, onehot: pd.DataFrame) -> pd.DataFrame:
    score_num = pd.to_numeric(df["Score"], errors="coerce").fillna(df["Score"].mean())
    tokens_num = pd.to_numeric(df["tokens"], errors="coerce").fillna(df["tokens"].median())
    text_len_num = df["Text"].str.len().astype(np.float32)

    numeric = pd.DataFrame(
        {
            "num::score_z": _zscore(score_num).astype(np.float32),
            "num::tokens_z": _zscore(tokens_num).astype(np.float32),
            "num::text_len_z": _zscore(text_len_num).astype(np.float32),
        }
    )
    # Limit extreme values to improve numerical stability during SVI.
    numeric = numeric.clip(lower=-3.0, upper=3.0)
    return pd.concat([onehot, numeric], axis=1).astype(np.float32)


def _train_and_check(
    counts: sparse.csr_matrix,
    vocab: np.ndarray,
    keywords: dict[str, list[str]],
    x_design: pd.DataFrame,
    run_name: str,
    lr: float = 0.01,
) -> None:
    print(f"\n=== {run_name} ===")
    print(f"Design matrix shape: {x_design.shape}")

    model = CSPF2(
        counts=counts,
        vocab=vocab,
        keywords=keywords,
        residual_topics=2,
        batch_size=min(256, counts.shape[0]),
        X_design_matrix=x_design,
    )
    params = model.train_step(
        num_steps=25,
        lr=lr,
        random_seed=7,
    )

    required = [
        "beta_shape",
        "beta_rate",
        "theta_shape",
        "theta_rate",
        "beta_tilde_shape",
        "beta_tilde_rate",
        "lambda_location",
        "lambda_scale",
    ]
    missing = [k for k in required if k not in params]
    assert not missing, f"Missing expected variational params: {missing}"

    topics, e_theta = model.return_topics()
    beta_df = model.return_beta()
    effects_df = model.return_covariate_effects()

    assert len(topics) == counts.shape[0], "Topic assignment length mismatch"
    assert e_theta.shape == (counts.shape[0], model.K), "Theta shape mismatch"
    assert beta_df.shape == (counts.shape[1], model.K), "Beta shape mismatch"
    assert effects_df.shape == (x_design.shape[1], model.K), "Covariate effects shape mismatch"
    assert np.isfinite(model.Metrics.loss[-1]), "Final loss is not finite"
    assert np.isfinite(e_theta).all(), "Theta contains non-finite values"
    assert np.isfinite(beta_df.values).all(), "Beta contains non-finite values"
    assert np.isfinite(effects_df.values).all(), "Covariate effects contain non-finite values"

    print(f"Final loss: {model.Metrics.loss[-1]:.4f}")
    print(f"Topics shape: {e_theta.shape}")
    print(f"Beta shape: {beta_df.shape}")
    print(f"Covariate effects shape: {effects_df.shape}")
    print("PASS")


def main() -> None:
    np.random.seed(42)
    df, counts, vocab = _prepare_data()
    keywords = _build_keywords(set(vocab))

    x_onehot = _grouped_onehot_covariates(df)
    _train_and_check(
        counts,
        vocab,
        keywords,
        x_onehot,
        "Test 1: grouped one-hot covariates",
        lr=0.01,
    )

    x_mixed = _mixed_covariates(df, x_onehot)
    _train_and_check(
        counts,
        vocab,
        keywords,
        x_mixed,
        "Test 2: grouped one-hot + numeric covariates",
        lr=0.005,
    )

    print("\nAll CSPF2 checks passed.")


# if __name__ == "__main__":
#     main()


print("Loading ./data/10k_amazon.csv")
df = pd.read_csv(
    "/Users/bernd/Documents/01_Coding/02_GitHub/poisson_topicmodels/data/10k_amazon.csv"
)
df = df.dropna(subset=["Text"]).copy()

# Keep runtime low while still testing grouped design and SVI behavior.
n_docs = min(2000, len(df))
df = df.sample(n=n_docs, random_state=42).reset_index(drop=True)

vectorizer = CountVectorizer(stop_words="english", min_df=5, max_features=5000)
counts = sparse.csr_matrix(vectorizer.fit_transform(df["Text"]), dtype=np.float32)
vocab = vectorizer.get_feature_names_out()

print(f"Documents: {counts.shape[0]}, Vocabulary: {counts.shape[1]}")


np.random.seed(42)
# df, counts, vocab = _prepare_data()
keywords = _build_keywords(set(vocab))


x_onehot = _grouped_onehot_covariates(df)
_train_and_check(
    counts,
    vocab,
    keywords,
    x_onehot,
    "Test 1: grouped one-hot covariates",
    lr=0.01,
)

x_mixed = _mixed_covariates(df, x_onehot)
_train_and_check(
    counts,
    vocab,
    keywords,
    x_mixed,
    "Test 2: grouped one-hot + numeric covariates",
    lr=0.005,
)

print("\nAll CSPF2 checks passed.")
