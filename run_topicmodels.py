# --- JAX Configuration for Metal GPU ---
# If you use Mac and want to enable Metal GPU for JAX,
# make sure to call the appropriate functions in jax_config.py
# --- Import topicmodels package ---
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.feature_extraction.text import CountVectorizer

from poisson_topicmodels import CSPF, ETM, PF, SPF, TBIP
from poisson_topicmodels.utils.utils import load_embeds

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
estimated_params = tm1.train_step(num_steps=500, lr=0.1)

# ---- Inspect results ----
print(estimated_params)
print(estimated_params.keys())
topics, e_theta = tm1.return_topics()
beta = tm1.return_beta()
top_words = tm1.return_top_words_per_topic(n=10)

# --- See loss within inherited metrics object ---
tm1.Metrics.loss


# ###############
# ### PF Test ###
# ###############


tm2 = PF(counts, vocab, num_topics=10, batch_size=1024)
estimated_params = tm2.train_step(num_steps=100, lr=0.01)
topics, e_theta = tm2.return_topics()
betas = tm2.return_beta()


# #################
# ### CSPF Test ###
# #################


category0 = ["grocery gourmet food", "toys games"]
covariable = df1["Cat1"].apply(lambda x: 0 if x in category0 else 1)
print(covariable[0:10])
print(df1["Cat1"].head(10))

X_design_matrix = pd.DataFrame({"intercept": np.repeat(1, len(df1)), "var_infromative": covariable})

tm3 = CSPF(
    counts,
    vocab,
    keywords,
    residual_topics=2,
    batch_size=1024,
    X_design_matrix=X_design_matrix,
)
estimated_params = tm3.train_step(num_steps=1000, lr=0.01)
topics, e_theta = tm3.return_topics()
betas = tm3.return_beta()


# ##############
# ## TBIP Test #
# ##############

df1["speaker"] = np.random.choice(
    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"], size=len(df1), replace=True
)
tm4 = TBIP(counts, vocab, num_topics=10, authors=df1.speaker, batch_size=1024)
estimated_params = tm4.train_step(num_steps=1000, lr=0.01)


# ##############
# ## CPF Test ##
# ##############


# tm3 = CPF(counts, vocab, num_topics = 5, batch_size = 1024, X_design_matrix = X_design_matrix)
# svi_batch, svi_state = tm3.train_step(num_steps = 100, lr = 0.01)
# estimated_params = svi_batch.get_params(svi_state)


df1["speaker"] = np.random.choice(
    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"], size=len(df1), replace=True
)
tm4 = TBIP(counts, vocab, num_topics=10, authors=df1.speaker, batch_size=1024)
estimated_params = tm4.train_step(num_steps=1000, lr=0.01)


# ##############
# ### ETM TEST #
# ##############


# -- Create embeddings --
# embeds = create_word2vec_embedding_from_dataset(list(df1["Text"]))
# save_embeds(embeds, "data/embeds.bin")

# -- Load embeddings --
# path to embeddings
path_to_embeddings = "data/embeds.bin"
embeddings_mapping = load_embeds(path_to_embeddings)

# -- RUN ETM --

tm5 = ETM(
    counts,
    vocab,
    num_topics=5,
    batch_size=1024,
    embeddings_mapping=embeddings_mapping,
    embed_size=300,
)
estimated_params = tm5.train_step(num_steps=100, lr=0.01)


# --- TVTBIP, STBS and ETM to be added --- #
# tvtbip already implemented in models/tbip.py #

# e.g.,
# tm5 = topicmodels("TBIP", counts, vocab, num_topics, authors, batchsize, time_varying = True, initial_beta_shape, initial_beta_rate)
