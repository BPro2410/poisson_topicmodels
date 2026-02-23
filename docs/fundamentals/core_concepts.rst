.. _core_concepts:

================================================================================
Core Concepts
================================================================================

This page introduces the fundamental concepts underlying probabilistic topic modeling
and poisson-topicmodels.

What is Topic Modeling?
=======================

**Topic modeling** is a statistical technique for discovering abstract "topics" that
occur in a collection of documents.

Key Idea:

- Each **topic** is a distribution over words (some words more likely than others)
- Each **document** is a mixture of topics (can contain multiple topics)
- We observe word counts in documents and infer the hidden topic structure

Example:

Imagine 3 documents about science, politics, and cuisine. The model might discover:

- **Topic 1** (Science): "research", "experiment", "data", "variable"...
- **Topic 2** (Politics): "government", "vote", "policy", "election"...
- **Topic 3** (Cuisine): "recipe", "cook", "ingredient", "flavor"...

Document 1 (Science paper): 80% Topic 1, 10% Topic 2, 10% Topic 3
Document 2 (Political cookbook): 30% Topic 1, 50% Topic 2, 20% Topic 3
Document 3 (Cooking blog): 5% Topic 1, 5% Topic 2, 90% Topic 3

Document-Term Matrix
====================

The fundamental input to all topic models is a **document-term matrix** (DTM):

- **Rows**: Documents
- **Columns**: Vocabulary terms (words)
- **Values**: Word counts in each document

Example (5 documents × 10 vocabulary):

.. code-block:: text

   Document  | word_1 | word_2 | word_3 | ... | word_10
   ----------+--------+--------+--------+-----+---------
   doc_1     |   3    |   0    |   5    | ... |    1
   doc_2     |   1    |   2    |   0    | ... |    4
   doc_3     |   0    |   7    |   2    | ... |    0
   ...       |   ..   |   ..   |   ..   | ... |   ..
   doc_5     |   2    |   1    |   3    | ... |    6

**Sparse Format**:
In practice, DTMs are **very sparse** (mostly zeros) because documents use only a
small fraction of vocabulary. We use sparse matrix formats (CSR, CSC) for efficiency.

Vocabulary
==========

The **vocabulary** is the complete list of unique words (terms) in your corpus.

- Size depends on preprocessing: 500 - 100,000+ words
- Typically includes preprocessing:
  - Lowercasing ("Hello" → "hello")
  - Removing punctuation
  - Stopword removal ("the", "a", "is")
  - Stemming/lemmatization ("running" → "run")

Example vocabulary:

.. code-block:: python

   vocab = np.array([
       'research',     # word_0
       'data',         # word_1
       'experiment',   # word_2
       'science',      # word_3
       ...
       'cooking'       # word_999
   ])

Topics and Word Distributions
==============================

A **topic** is a probability distribution over vocabulary terms.

Example topic (topic_2):

.. code-block:: python

   P(word | topic_2) = {
       'research': 0.08,
       'data': 0.07,
       'experiment': 0.06,
       'cooking': 0.001,
       'science': 0.05,
       ...
   }

Top words for this topic: 'research', 'data', 'experiment', 'science', ...

**Interpretation**: High-probability words characterize the topic; low-probability
words are just noise.

Topics are represented as vectors:

.. code-block:: python

   topic_2 = np.array([0.08, 0.07, 0.06, 0.05, ...])  # shape: (vocab_size,)

Document-Topic Mixtures
=======================

A **document** is a mixture of topics - a probability distribution over topics.

Example document:

.. code-block:: python

   P(topic | doc_1) = {
       'topic_0': 0.60,  # Science
       'topic_1': 0.25,  # Commerce/Business
       'topic_2': 0.15,  # Politics
   }

Interpretation: This document is primarily about science (60%), some business (25%),
and a bit about politics (15%).

Represented as a vector:

.. code-block:: python

   doc_1_topics = np.array([0.60, 0.25, 0.15])  # shape: (num_topics,)

The Complete Picture
====================

Combined view:

.. code-block:: text

   Documents              Topics (β)        Document-Topic (θ)

   [word counts]  →  [matrix mult]  →   [topic mixture]

   d1: [5, 2, 3, ...]     β_0: [0.1, 0.05, 0.02, ...]     θ_d1: [0.60, 0.25, 0.15]
   d2: [0, 8, 1, ...]  ×  β_1: [0.02, 0.08, 0.06, ...]  =  θ_d2: [0.25, 0.50, 0.25]
   d3: [2, 1, 7, ...]     β_2: [0.05, 0.02, 0.09, ...]     θ_d3: [0.15, 0.10, 0.75]
                          β_K: [0.01, 0.03, 0.02, ...]

Poisson Factorization Model
============================

The core model in poisson-topicmodels is **Poisson Factorization** (PF).

**Generative Process** (how data is created):

1. For each document d:

   - Draw document-topic distribution: $\theta_d \sim \text{Gamma}(\alpha, \alpha)^K$

2. For each topic k:

   - Draw topic-word distribution: $\beta_k \sim \text{Gamma}(\eta, \eta)^V$

3. For each document-word pair (d, w):

   - Draw count: $C_{dw} \sim \text{Poisson}(\sum_k \theta_d^k \beta_k^w)$

Where:
- $C_{dw}$ = observed word count
- $\theta_d^k$ = intensity of topic k in document d
- $\beta_k^w$ = intensity of word w in topic k
- K = number of topics
- V = vocabulary size

**Why Poisson?**

Traditional LDA uses:
- Multinomial: Exactly K topics per document
- Hierarchical Dirichlet: Complex sampling

Poisson factorization:
- Natural for count data
- Linear combination of topic-word factors
- Efficient SVI training
- Flexible for extensions

Inference: Learning from Data
================================

We observe:

- Document-term matrix: $\{C_{dw}\}$ for all d, w

We want to learn:

- Topics: $\{\beta_k\}$ - what each topic is about
- Document-topics: $\{\theta_d\}$ - topic mixtures per document

**Bayesian Approach**:

1. Place priors: $\theta_d, \beta_k \sim \text{prior}$
2. Compute posterior: $P(\theta, \beta | C)$ given observed data
3. Optimize: Maximize evidence lower bound (ELBO) with SVI

This is done via **Stochastic Variational Inference (SVI)**:

- Approximate posterior with mean-field variational family
- Update with mini-batches of documents
- Converge to local optimum

Hyperparameters
================

Each model has hyperparameters controlling the inference process:

**Learning Rate** (lr)
   Controls step size in optimization. Typical range: 0.001 - 0.1

   - Higher → faster learning but less stable
   - Lower → slower but more stable

**Number of Topics** (K)
   How many topics to discover. No universal "right" answer.

   - Start with 10-20
   - Evaluate using coherence, perplexity, or domain knowledge

**Batch Size**
   Documents per training iteration. Typical: 32, 64, 128, 256

   - Larger → more stable gradients, slower iterations
   - Smaller → noisier but faster iterations

**Iterations/Epochs**
   How long to train. Usually 100-1000 iterations until convergence.

Convergence and Loss
====================

Training is monitored through **loss** (negative ELBO):

- **Early iterations**: Loss decreases rapidly (large changes)
- **Late iterations**: Loss decreases slowly (fine-tuning)
- **Convergence**: Loss plateaus (further training unlikely to help)

Example learning curve:

.. code-block:: text

   Loss
   ^
   |     .-'       (Learning plateau)
   |    /'
   |  .'           (Steep learning)
   |.'
   |___________________> Iteration

**Early stopping**: Stop when loss plateus rather than training to fixed number of iterations.

Interpreting Topics
====================

After training, topics are interpretable through their top words:

**High-quality topic**: Top words form coherent theme

.. code-block:: text

   Topic 5: research, experiment, data, variable, analysis, hypothesis
   Topic 12: president, congress, vote, senator, legislation, party

**Low-quality topic**: Top words scattered or similar across topics

.. code-block:: text

   Topic 3: the, of, and, to, a, in  (mostly stopwords - bad!)
   Topic 7: research, data, president, cooking, vote  (incoherent)

Quality assessment:
- Manual inspection of top words
- Coherence metrics
- Domain expert evaluation
- Downstream task performance

Model Assumptions
=================

Topic models make several assumptions—understanding these helps you use them effectively:

**Bag-of-Words**: Word order doesn't matter

- "cat chases mouse" = "mouse chases cat"
- Topic models only see word counts, not sequences
- Good for thematic analysis, not for capturing narrative structure

**Mixture Assumption**: Documents are mixtures of topics

- Not all documents follow this (some are single-topic)
- More relevant for long documents (100+ words) than short snippets

**Topic Independence**: Topics are independent

- In reality, some topics co-occur
- Model learns this through document-topic mixtures

**Homogeneous Vocabulary**: Same vocabulary across all documents

- Strong assumption but rarely problematic in practice

Common Pitfalls
================

**Too many topics**: Topics become redundant and incoherent

**Too few topics**: Topics are vague, combining distinct themes

**Poor preprocessing**: Stopwords or junk data create low-quality topics

**Short documents**: Sparse word counts → unreliable inference (need >50 words typically)

**Wrong batch size**: Too large → slow iterations; too small → noisy updates

**Insufficient training**: Model hasn't converged

**No validation**: Blindly trusting discovered topics without inspection

Next Steps
==========

- :doc:`poisson_factorization` - Deep dive into Poisson Factorization model
- :doc:`seeded_models` - How to guide discovery with keywords
- :doc:`covariate_models` - Modeling topic structure with metadata
- :doc:`../api/index` - API reference for implementation details
