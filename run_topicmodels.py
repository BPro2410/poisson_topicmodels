# --- Import topicmodels package ---
from packages.models.topicmodels import topicmodels

 

#######################

#### - ANWENDUNG - ####

#######################

 

"""

To Dos:

- Schauen welche klassenattribute wir in die abstrakte klasse bauen können

- vielleicht ein paar attribute in eine self.model_settings dictionary packen, damit es weniger ueberladen ist?!

- TBIP implementierung anpassen, variational variables und parameter ggf umbenennen

"""

 

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sparse

 

#######

# SPF #

#######

 

# SPF Versuch1

 
# ---- Load data ----
df1 = pd.read_csv("/home/sagemaker-user/Bernd-PHD/Data/10k_amazon.csv")

 

# ---- Define keywords ----
pets = ["dog","cat", "litter", "cats", "dogs", "food", "box", "collar", "water", "pet"]
toys = ["toy", "game", "play", "fun", "old", "son", "year", "loves", "kids", "daughter"]
beauty = ["hair", "skin", "product", "color", "scent", "smell", "used", "dry", "using", "products"]
baby = ["baby", "seat", "diaper", "diapers", "stroller", "bottles", "son", "pump", "gate", "months"]
health = ["product", "like", "razor", "shave", "time", "day", "shaver", "better", "work", "years"]
grocery = ["tea", "taste", "flavor", "coffee", "sauce", "chocolate", "sugar", "eat", "sweet", "delicious"]

keywords = {"pet supplies": pets, "toys games": toys, "beauty": beauty, "baby products": baby,

            "health personal care": health, "grocery gourmet food": grocery}

 

# --- Create corpus ---
cv = CountVectorizer(stop_words='english', min_df = 2)
cv.fit(df1["Text"])
counts = sparse.csr_matrix(cv.transform(df1["Text"]), dtype = np.float32)
vocab = cv.get_feature_names_out()





# ####################

# ##### SPF TEST #####

# ####################

 

# ---- Initialize TM package ----
tm1 = topicmodels("SPF", counts, vocab, keywords, residual_topics = 2, batch_size = 1024)

# ---- Run inference -----
estimated_params = tm1.train_step(num_steps = 100, lr = 0.01)

# ---- Inspect results ----
estimated_params
topics, e_theta = tm1.return_topics()
beta = tm1.return_beta()


# --- See loss within inherited metrics object ---
tm1.Metrics.loss




# ###############

# ### PF Test ###

# ###############

 

tm2 = topicmodels("PF", counts, vocab, num_topics = 10, batch_size = 1024)
estimated_params = tm2.train_step(num_steps = 100, lr =0.01)
topics, e_theta = tm2.return_topics()
betas = tm2.return_beta()

 

# #################

# ### CSPF Test ###

# #################

 

category0 = ["grocery gourmet food", "toys games", "baby products"]
covariable = df1['Cat1'].apply(lambda x: 0 if x in category0 else 1)
print(covariable[0:10])
print(df1['Cat1'].head(10))

X_design_matrix = pd.DataFrame({'intercept' : np.repeat(1, len(df1)), 'var_infromative' : covariable})

tm3 = topicmodels("CSPF", counts, vocab, keywords, residual_topics = 2, batch_size = 1024, X_design_matrix = X_design_matrix)
estimated_params = tm3.train_step(num_steps = 100, lr = 0.01)
topics, e_theta = tm3.return_topics()
betas = tm3.return_beta()

 

# ##############

# ## CPF Test ##

# ##############

 

# tm3 = topicmodels("CPF", counts, vocab, num_topics = 5, batch_size = 1024, X_design_matrix = X_design_matrix)
# svi_batch, svi_state = tm3.train_step(num_steps = 100, lr = 0.01)
# estimated_params = svi_batch.get_params(svi_state)





##############

## TBIP Test #

##############

 

df1['speaker'] = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'], size=len(df1), replace=True)
tm4 = topicmodels("TBIP", counts, vocab, num_topics = 10, authors = df1.speaker, batch_size = 1024)
print("HELLO")
svi_batch, svi_sate, losses = tm4.train_step(num_steps = 1000, lr = 0.01)
import matplotlib.pyplot as plt
plt.plot(losses[800:])

# estimated_params

# import jax.numpy as jnp

#  self.author_map = jnp.unique(authors)

#         self.author_indices = jnp.array(list(range(len(self.author_map))))

      