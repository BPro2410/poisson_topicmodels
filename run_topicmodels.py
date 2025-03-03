# --- Import topicmodels package ---
from packages.topicmodels import topicmodels



#######################
#### - ANWENDUNG - ####
#######################

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sparse

# ---- Load data ----
df1 = pd.read_csv("./data/10k_amazon.csv")

# --- Create corpus ---
cv = CountVectorizer(stop_words='english', min_df = 2)
counts = sparse.csr_matrix(cv.fit_transform(df1["Text"]), dtype = np.float32)
vocab = cv.get_feature_names_out()




# ####################
# ##### SPF TEST #####
# ####################

# ---- Define keywords ----
pets = ["dog","cat", "litter", "cats", "dogs", "food", "box", "collar", "water", "pet"]
toys = ["toy", "game", "play", "fun", "old", "son", "year", "loves", "kids", "daughter"]
beauty = ["hair", "skin", "product", "color", "scent", "smell", "used", "dry", "using", "products"]
baby = ["baby", "seat", "diaper", "diapers", "stroller", "bottles", "son", "pump", "gate", "months"]
health = ["product", "like", "razor", "shave", "time", "day", "shaver", "better", "work", "years"]
grocery = ["tea", "taste", "flavor", "coffee", "sauce", "chocolate", "sugar", "eat", "sweet", "delicious"]

keywords = {"pet supplies": pets, "toys games": toys, "beauty": beauty, "baby products": baby,
            "health personal care": health, "grocery gourmet food": grocery}


# ---- Initialize TM package ----
tm1 = topicmodels("SPF", counts, vocab, keywords, residual_topics = 2, batch_size = 1024)
# ---- Run inference -----
svi_batch, svi_state = tm1.train_step(num_steps = 100, lr = 0.01)
# ---- Inspect results ----
estimated_params = svi_batch.get_params(svi_state)

# - model metrics -
tm1.Metrics



# ###############
# ### PF Test ###
# ###############

# tm2 = topicmodels("PF", counts, vocab, num_topics = 10, batch_size = 1024)
# svi_batch, svi_state = tm2.train_step(num_steps = 100, lr =0.01)
# estimated_params = svi_batch.get_params(svi_state)


# #################
# ### CSPF Test ###
# #################

# category0 = ["grocery gourmet food", "toys games", "baby products"]
# covariable = df1['Cat1'].apply(lambda x: 0 if x in category0 else 1)

# print(covariable[0:10])
# print(df1['Cat1'].head(10))

# X_design_matrix = pd.DataFrame({'intercept' : np.repeat(1, len(df1)), 'var_infromative' : covariable})


# tm3 = topicmodels("CSPF", counts, vocab, keywords, residual_topics = 2, batch_size = 1024, X_design_matrix = X_design_matrix)
# svi_batch, svi_state = tm3.train_step(num_steps = 100, lr = 0.01)
# estimated_params = svi_batch.get_params(svi_state)


# ##############
# ## CPF Test ##
# ##############

# tm3 = topicmodels("CPF", counts, vocab, num_topics = 5, batch_size = 1024, X_design_matrix = X_design_matrix)
# svi_batch, svi_state = tm3.train_step(num_steps = 100, lr = 0.01)
# estimated_params = svi_batch.get_params(svi_state)




##############
## TBIP Test #
##############

# --- create authors ---
# df1['speaker'] = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'], size=len(df1), replace=True)

# --- run tbip ---
# tm4 = topicmodels("TBIP", counts, vocab, num_topics = 10, authors = df1.speaker, batch_size = 1024)
# svi_batch, svi_sate, losses = tm4.train_step(num_steps = 1000, lr = 0.01)
