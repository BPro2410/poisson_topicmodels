# CSPF2 Experiments on 10k Amazon Reviews

## Overview

We applied the **Covariate Seeded Poisson Factorization (CSPF2)** model to a corpus of 10,000 Amazon product reviews spanning 6 categories. The goal was to test whether CSPF2's grouped shrinkage prior can distinguish **informative covariates** from **pure noise covariates** at the group level.

**Script:** `examples/test_amazon_grouped.py`

---

## Data Setup

### Corpus

- **Source:** `data/10k_amazon.csv`
- **Documents:** 10,000 (sampled with `random_state=42`)
- **Vocabulary:** 5,000 terms (CountVectorizer, English stopwords removed, `min_df=5`)

### Seed Keywords (6 guided topics, 0 residual)

| Topic | Keywords |
|-------|----------|
| pet supplies | dog, cat, litter, cats, dogs, food, box, collar, water, pet |
| toys games | toy, game, play, fun, old, son, year, loves, kids, daughter |
| beauty | hair, skin, product, color, scent, smell, used, dry, using, products |
| baby products | baby, seat, diaper, diapers, stroller, bottles, son, pump, gate, months |
| health personal care | product, like, razor, shave, time, day, shaver, better, work, years |
| grocery gourmet food | tea, taste, flavor, coffee, sauce, chocolate, sugar, eat, sweet, delicious |

### Covariates (2 groups, 4 dummy columns)

We constructed two categorical covariate groups — one with a real association to the text, one purely random:

**Group 1: `animal` (dog / cat)**
- 80% of "pet supplies" documents receive a random `dog` or `cat` label
- 10% of "toys games" documents receive a random `dog` or `cat` label
- All other documents: both dummies = 0

**Group 2: `eating` (ice cream / banana)**
- Every document receives exactly one label at random (50/50 split)
- By construction, this group carries **no information** about any topic

Observed covariate means:

| Column | Mean |
|--------|------|
| `animal::dog` | 0.065 |
| `animal::cat` | 0.062 |
| `eating::ice cream` | 0.501 |
| `eating::banana` | 0.499 |

The `::` separator tells CSPF2 which dummies belong to the same group (animal = group 0, eating = group 1).

---

## Experiment 1: Horseshoe prior (a = 0.5), 1200 SVI steps

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| `a_tau`, `a_rho_tau` | 0.5 |
| `a_delta`, `a_rho_delta` | 0.5 |
| `b_rho_tau`, `b_rho_delta` | 1.0 |
| `num_steps` | 1200 |
| `lr` | 0.01 |
| `batch_size` | 1024 |
| `residual_topics` | 0 |

### Results

| Metric | Value |
|--------|-------|
| Final loss | 190.75 |
| Topic accuracy (vs Cat1) | 69.78% |

#### Covariate effects (λ)

|  | pet supplies | toys games | beauty | baby products | health personal care | grocery gourmet food |
|--|-------------|-----------|--------|--------------|---------------------|---------------------|
| animal::dog | **+0.37** | +0.01 | −0.05 | −0.04 | −0.04 | −0.06 |
| animal::cat | **+0.38** | +0.01 | −0.04 | +0.01 | −0.07 | −0.04 |
| eating::ice cream | −0.04 | −0.00 | +0.01 | +0.01 | −0.03 | −0.01 |
| eating::banana | +0.02 | +0.02 | −0.02 | +0.00 | +0.00 | −0.02 |

#### Group shrinkage E[δ²_{gk}]

|  | pet supplies | toys games | beauty | baby products | health personal care | grocery gourmet food |
|--|-------------|-----------|--------|--------------|---------------------|---------------------|
| animal | 3.71 | 5.44 | 4.18 | 2.44 | 3.01 | 1.86 |
| eating | 3.16 | 2.55 | 3.95 | 2.37 | 4.60 | 3.63 |

#### Effective group variance (τ²_k × δ²_{gk})

|  | pet supplies | toys games | beauty | baby products | health personal care | grocery gourmet food |
|--|-------------|-----------|--------|--------------|---------------------|---------------------|
| animal | 13.72 | 16.97 | 25.73 | 7.45 | 14.37 | 7.65 |
| eating | 11.67 | 7.96 | 24.32 | 7.25 | 21.98 | 14.91 |

### Interpretation

- The **λ coefficients** correctly identify animal as relevant for pet supplies (+0.37) and eating as noise (~0).
- However, the **shrinkage parameters (δ²)** do not clearly separate the two groups. The effective variances for animal and eating are of similar magnitude across most topics.
- The ratio of effective variance (animal / eating) for pet supplies is only **1.2x** — too small to confidently declare group-level relevance from shrinkage alone.

**Diagnosis:** 1200 SVI steps with batch size 1024 were insufficient for the variational posterior over the shrinkage parameters to converge.

---

## Experiment 2: Horseshoe prior (a = 0.5), 3000 SVI steps

This is the key experiment. We kept the same horseshoe prior (a = 0.5) but increased training from 1200 to 3000 steps to test whether convergence was the bottleneck.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| `a_tau`, `a_rho_tau` | 0.5 |
| `a_delta`, `a_rho_delta` | 0.5 |
| `b_rho_tau`, `b_rho_delta` | 1.0 |
| `num_steps` | **3000** |
| `lr` | 0.01 |
| `batch_size` | 1024 |
| `residual_topics` | 0 |

### Results

| Metric | Value |
|--------|-------|
| Final loss | 187.16 |
| Topic accuracy (vs Cat1) | **73.47%** |

#### Covariate effects (λ)

|  | pet supplies | toys games | beauty | baby products | health personal care | grocery gourmet food |
|--|-------------|-----------|--------|--------------|---------------------|---------------------|
| animal::dog | **+1.504** | −0.079 | −0.548 | −0.307 | −0.333 | −0.262 |
| animal::cat | **+1.500** | −0.094 | −0.501 | −0.282 | −0.346 | −0.250 |
| eating::ice cream | −0.046 | −0.010 | −0.008 | −0.022 | +0.047 | +0.042 |
| eating::banana | −0.039 | −0.004 | −0.004 | +0.085 | −0.024 | −0.044 |

The animal coefficients are now **4x larger** than in Experiment 1 (1.50 vs 0.37). The model has fully recovered the true effect: documents with a dog/cat label have substantially higher topic intensity for "pet supplies" through the softplus link. Meanwhile, eating coefficients remain at or near zero across all topics, correctly reflecting the absence of any real association.

The negative animal coefficients on non-pet topics (−0.26 to −0.55) indicate that the model also learns a *contrastive* pattern: documents about pets are somewhat less likely to be about beauty, health, etc.

#### Global shrinkage E[τ²_k] (per topic)

| | pet supplies | toys games | beauty | baby products | health personal care | grocery gourmet food |
|--|-------------|-----------|--------|--------------|---------------------|---------------------|
| E[τ²] | 5.69 | 4.30 | 8.47 | 6.38 | 6.67 | 6.33 |

These values reflect how much *any* covariate group is allowed to influence each topic. Beauty has the largest global scale (8.47), suggesting the model allocates more capacity for covariate effects there.

#### Local group shrinkage E[δ²_{gk}] (per group × topic)

|  | pet supplies | toys games | beauty | baby products | health personal care | grocery gourmet food |
|--|-------------|-----------|--------|--------------|---------------------|---------------------|
| animal | **18.45** | 6.85 | 4.81 | 2.92 | 3.87 | 2.56 |
| eating | 4.18 | 3.70 | 4.91 | 4.30 | 6.42 | 4.69 |

This is the core group-relevance table. The animal group's δ² for pet supplies (**18.45**) is dramatically larger than eating's (4.18) — a **4.4x ratio**. For all other topics, animal and eating have similar δ² values (2–7), indicating that neither group matters much outside pet supplies.

The fact that animal's δ² is high specifically for pet supplies (and not for other topics) confirms that the model correctly identifies **both** which group is relevant **and** for which topic.

#### Effective group variance (τ²_k × δ²_{gk})

|  | pet supplies | toys games | beauty | baby products | health personal care | grocery gourmet food |
|--|-------------|-----------|--------|--------------|---------------------|---------------------|
| animal | **104.96** | 29.45 | 40.72 | 18.61 | 25.82 | 16.18 |
| eating | 23.77 | 15.90 | 41.60 | 27.41 | 42.84 | 29.70 |

The effective variance combines global (τ²) and local (δ²) scales. The headline number: **animal × pet supplies = 105.0**, versus eating × pet supplies = 23.8 — a ratio of **4.4x**. This is the model's quantification of how much more important the animal group is for the pet supplies topic.

#### Most important group per topic

| Topic | Most important group | Effective variance |
|-------|---------------------|-------------------|
| pet supplies | **animal** | 105.0 |
| toys games | **animal** | 29.5 |
| beauty | eating | 41.6 |
| baby products | eating | 27.4 |
| health personal care | eating | 42.8 |
| grocery gourmet food | eating | 29.7 |

Animal is correctly identified as the most important group for pet supplies and toys games (recall: 10% of toys games docs also received animal labels). For the remaining topics, neither group has a real signal, and the "winner" is essentially arbitrary noise — the eating group has slightly higher variance there simply because its dummies are more uniformly distributed (50% of all docs vs. ~13% for animal).

#### Top 10 words per topic

| Topic | Top words |
|-------|-----------|
| pet supplies | dog, box, product, cat, just, water, like, cats, don, dogs |
| toys games | toy, old, play, game, year, fun, loves, like, great, just |
| beauty | hair, product, like, skin, use, used, really, using, just, great |
| baby products | baby, use, just, easy, great, seat, love, son, fit, old |
| health personal care | product, time, just, use, years, used, ve, good, amazon, work |
| grocery gourmet food | like, good, tea, taste, great, just, product, flavor, really, ve |

The top words align well with the seed keywords and category semantics. The model produces interpretable topics while simultaneously estimating covariate effects.

---

## Experiment 3: Stronger shrinkage prior (a = 0.1), 3000 SVI steps

We reduced all shrinkage hyperparameters from 0.5 to 0.1 (as suggested in the model specification, following Cadonna, Frühwirth-Schnatter & Knaus 2019) to test whether a more aggressive prior improves group-level discrimination.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| `a_tau`, `a_rho_tau` | **0.1** |
| `a_delta`, `a_rho_delta` | **0.1** |
| `b_rho_tau`, `b_rho_delta` | 1.0 |
| `num_steps` | 3000 |
| `lr` | 0.01 |
| `batch_size` | 1024 |
| `residual_topics` | 0 |

### Results

| Metric | Value |
|--------|-------|
| Final loss | 187.16 |
| Topic accuracy (vs Cat1) | 73.43% |

#### Covariate effects (λ)

|  | pet supplies | toys games | beauty | baby products | health personal care | grocery gourmet food |
|--|-------------|-----------|--------|--------------|---------------------|---------------------|
| animal::dog | **+1.505** | −0.079 | −0.549 | −0.307 | −0.333 | −0.262 |
| animal::cat | **+1.501** | −0.094 | −0.501 | −0.282 | −0.346 | −0.250 |
| eating::ice cream | −0.046 | −0.010 | −0.008 | −0.022 | +0.047 | +0.042 |
| eating::banana | −0.039 | −0.004 | −0.004 | +0.086 | −0.024 | −0.044 |

#### Effective group variance (τ²_k × δ²_{gk})

|  | pet supplies | toys games | beauty | baby products | health personal care | grocery gourmet food |
|--|-------------|-----------|--------|--------------|---------------------|---------------------|
| animal | **105.51** | 29.61 | 40.86 | 18.62 | 25.87 | 16.19 |
| eating | 23.78 | 15.90 | 41.70 | 27.41 | 42.96 | 29.72 |

### Comparison with Experiment 2

| Metric | a = 0.5, 3000 steps | a = 0.1, 3000 steps | Difference |
|--------|---------------------|---------------------|------------|
| Final loss | 187.16 | 187.16 | ~0 |
| Topic accuracy | 73.47% | 73.43% | ~0 |
| λ animal::dog → pet supplies | 1.504 | 1.505 | < 0.001 |
| δ² animal → pet supplies | 18.45 | 18.55 | < 0.1 |
| Eff. variance animal → pet supplies | 105.0 | 105.5 | < 0.5 |

The results are **virtually identical** — differences are in the third decimal place.

---

## Key Findings

### 1. Training convergence matters more than prior choice (for this dataset)

The jump from 1200 to 3000 steps produced dramatic improvements:

| Metric | 1200 steps | 3000 steps | Change |
|--------|-----------|-----------|--------|
| Topic accuracy | 69.78% | 73.47% | +3.7pp |
| λ animal → pet supplies | 0.37 | 1.50 | **4.1x** |
| Eff. variance ratio (animal/eating) for pet supplies | 1.2x | 4.4x | **3.7x** |

Changing the prior from 0.5 to 0.1 at 3000 steps produced **no measurable difference**. This indicates:
- With sufficient training, the data dominates the prior
- The animal signal in this dataset is strong enough that even a less aggressive prior can recover it
- Prior choice would likely matter more with weaker signals, more groups, or fewer training iterations

### 2. CSPF2 correctly identifies informative covariate groups

At convergence (3000 steps), the model:
- Assigns **large positive λ** (+1.50) to animal for pet supplies, near-zero for eating
- Gives the animal group **4.4x higher effective variance** than eating for pet supplies
- Learns contrastive effects: animal coefficients are negative for non-pet topics

### 3. The shrinkage prior does not fully concentrate eating toward zero

Even after 3000 steps, the eating group's effective variance sits at 15–43 across topics rather than near zero. This is a known limitation of **mean-field variational inference**, which tends to underestimate posterior concentration. An MCMC-based inference scheme would likely push the noise group's δ² closer to zero.

### 4. Practical recommendation

For this model and dataset:
- Use at least **3000 SVI steps** (or compute steps as a function of epochs)
- The default horseshoe prior (a = 0.5) is sufficient
- Interpret group relevance primarily through the **λ coefficients** (which clearly separate signal from noise), using the **effective variance** as a supporting indicator
- Consider the effective variance *ratio* between groups for a given topic rather than absolute values
