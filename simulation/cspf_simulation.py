####################################
#### ----- Simulation ----- ########
####################################

import numpy as np
import random
from poisson_topicmodels import topicmodels
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)

# Parameters
num_documents = D = 10000
vocab_size = V = 8000
num_topics = K = 5
words_per_doc = 100  # Adjustable: average number of words per document
p_covariate = 0.5  # Probability of x_i = 1

# Step 1: Create a vocabulary
vocabulary = [f"word_{i}" for i in range(vocab_size)]

# Step 2: Define topic-specific word distributions
topic_word_distributions = []
for _ in range(num_topics):
    # Each topic has a probability distribution over words
    probs = np.random.dirichlet(alpha=np.ones(vocab_size) * 0.1)
    topic_word_distributions.append(probs)

# Step 3: Assign covariate values (0 or 1)
covariates = np.random.binomial(1, p_covariate, size=num_documents)

# Step 4: Define topic probability vectors conditioned on covariate
pi_x0 = [0.1, 0.2, 0.3, 0.2, 0.2]  # Topic distribution when x_i = 0
pi_x1 = [0.7, 0.3, 0.0, 0.0, 0.0]  # Topic distribution when x_i = 1

# Step 5: Assign topics to documents based on covariate
document_topics = []
for x in covariates:
    topic_probs = pi_x1 if x == 1 else pi_x0
    topic = np.random.choice(num_topics, p=topic_probs)
    document_topics.append(topic)

document_topics = np.array(document_topics)

# Step 6: Generate documents
documents = []
for doc_id in range(num_documents):
    topic = document_topics[doc_id]
    word_probs = topic_word_distributions[topic]
    words = np.random.choice(vocabulary, size=words_per_doc, p=word_probs)
    documents.append(" ".join(words))

# Output: documents contains generated texts, document_topics holds ground-truth topic assignments,
# and covariates stores the binary covariate values


#####################
# Create seed words #
#####################

keywords = dict()
vocab = vocabulary

def print_topics(E_beta, num_words: int = 50):
    top_words = np.argsort(-E_beta, axis = 1)

    hot_words = dict()
    for topic_idx in range(K):
        if topic_idx in list(range(len(keywords.keys()))):
            topic_name = "{}".format(list(keywords.keys())[topic_idx])
            words_per_topic = num_words
            hot_words_topic = [vocab[word] for word in top_words[topic_idx, :words_per_topic]]
            hot_words[topic_name] = hot_words_topic
        else:
            words_per_topic = num_words
            hot_words[f"Topic_{topic_idx - len(keywords) + 1}"] = \
                    [vocab[word] for word in top_words[topic_idx, :words_per_topic]]

    return hot_words


# -- Top 100 words per topic
hot_words = print_topics(np.array(topic_word_distributions), num_words = 100)

# -- Sample of 10 words per topic for seed words
keywords_selected = {k: np.random.choice(v, 10, replace = False) for k, v in hot_words.items()}
keywords = keywords_selected



##################
# Estimate model #
##################

from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sparse

# --- hyperparameter ---
model_hyperparameter = dict(
    residual_topics = 0,
    epochs = 150,
    batch_size = 1024,
    lr = 0.01
)
seeded_topics = len(list(keywords.keys()))
model_hyperparameter["num_topics"] = model_hyperparameter["residual_topics"] + seeded_topics

# --- create corpus ---
cv = CountVectorizer(vocabulary = vocab)
cv.fit(documents)
counts = sparse.csr_matrix(cv.transform(documents), dtype=np.int8)

# --- Create design matrix ---
X_design_matrix = pd.DataFrame({"intercept": np.repeat(1, len(covariates)), "simulated_cov": covariates})

# --- Run topicmodels package ---
cspf = topicmodels("CSPF", counts, vocab, keywords, residual_topics = 0, batch_size = 1024, X_design_matrix = X_design_matrix)
num_steps = model_hyperparameter["epochs"] * int(counts.shape[0] / model_hyperparameter["batch_size"])
estimates = cspf.train_step(num_steps = num_steps, lr = 0.01)
topic_prediction, e_theta = cspf.return_topics()

# --- Analyze results ---
estimated_categories = estimates["theta_shape"] / estimates["theta_rate"]
estimated_categories = np.argmax(estimated_categories, axis = 1)

np.sum(document_topics == estimated_categories )/len(document_topics)

cov_effects = cspf.return_covariate_effects()
print(cov_effects)



z = 1.95
C = estimates["lambda_location"].shape[0]
K = estimates["lambda_location"].shape[1]
cov_effects = estimates["lambda_location"]
cov_scales = estimates["lambda_scale"]
# Format LaTeX entries
latex_matrix = []
for i in range(C):
    row = []
    for j in range(K):
        mu = cov_effects[i, j]
        sigma = cov_scales[i, j]
        lower = mu - z * sigma
        upper = mu + z * sigma

        # Format the inner mini-table
        if lower > 0 or upper < 0:
            mu_str = f"$\\mathbf{{{mu:.2f}}}$"
        else:
            mu_str = f"${mu:.2f}$"

        ci_str = f"{{\\footnotesize [{lower:.2f}, {upper:.2f}]}}"
        entry = f"\\begin{{tabular}}{{@{{}}c@{{}}}}\n{mu_str} \\\\\n{ci_str}\n\\end{{tabular}}"
        row.append(entry)
    latex_matrix.append(row)

# Print the LaTeX table
print("\\begin{table}[ht]")
print("\\centering")
print("\\begin{tabular}{l|" + "c" * K + "}")
print("\\toprule")
print("Covariate & " + " & ".join([f"Topic {j + 1}" for j in range(K)]) + " \\\\")
print("\\midrule")
for i, row in enumerate(latex_matrix):
    print(f"Cov {i + 1} & " + " & ".join(row) + " \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("""\\caption{
    Posterior mean estimates of covariate effects on topic proportions with 95\% credible intervals. 
    Values shown are the posterior means with corresponding 95\% credible intervals below each estimate, 
    based on a normal approximation. Boldface indicates effects whose intervals do not include zero, 
    suggesting strong evidence for a positive or negative association.
    }""")
print("\\label{tab:simulation_cov_effects2}")
print("\\end{table}")
