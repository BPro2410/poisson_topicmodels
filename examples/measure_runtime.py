# --- JAX Configuration for Metal GPU ---
# If you use Mac and want to enable Metal GPU for JAX,
# make sure to call the appropriate functions in jax_config.py
# --- Import topicmodels package ---
import os

import jax
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.feature_extraction.text import CountVectorizer

from poisson_topicmodels import SPF

# Reuse compiled executables across script runs to reduce startup overhead.
jax.config.update(
    "jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_poisson_topicmodels")
)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

# import jax_config
# from poisson_topicmodels import topicmodels

# ---- Load data ----
df1 = pd.read_csv("data/10k_amazon.csv")

# --- Create corpus ---
cv = CountVectorizer(stop_words="english", min_df=2)
cv.fit(df1["Text"])
counts = sparse.csr_matrix(cv.transform(df1["Text"]), dtype=np.float32)
vocab = cv.get_feature_names_out()


# ####################
# ##### SPF TEST #####
# ####################

# ---- Define keywords ----
pets = ["dog", "cat", "litter", "cats", "dogs", "food", "box", "collar", "water", "pet"]
toys = ["toy", "game", "play", "fun", "old", "son", "year", "loves", "kids", "daughter"]
beauty = ["hair", "skin", "product", "color", "scent", "smell", "used", "dry", "using", "products"]
baby = ["baby", "seat", "diaper", "diapers", "stroller", "bottles", "son", "pump", "gate", "months"]
health = ["product", "like", "razor", "shave", "time", "day", "shaver", "better", "work", "years"]
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

keywords = {
    "pet supplies": pets,
    "toys games": toys,
    "beauty": beauty,
    "baby products": baby,
    "health personal care": health,
    "grocery gourmet food": grocery,
}


# ---- Initialize TM package ----

tm1 = SPF(counts, vocab, keywords, residual_topics=0, batch_size=1024)
print(tm1)

# ---- Run inference -----
estimated_params = tm1.train_step(
    num_steps=1350,
    lr=0.1,
    jit_compile=True,
    cache_dense_counts=True,
)


topics, e_theta = tm1.return_topics()

print(topics)

print("ACCURACY")
print("================" * 5)
print(np.sum(topics == df1.Cat1))
print("================" * 5)
