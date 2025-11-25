# Examples - Topic Modeling with topicmodels

This directory contains example scripts demonstrating how to use the topicmodels package for topic modeling with different configurations and features.

## Quick Start

All examples can be run directly:

```bash
python examples/01_getting_started.py
```

## Examples Overview

### Example 1: Getting Started (`01_getting_started.py`)

**Level:** Beginner
**Duration:** ~5 minutes
**Topics Covered:**
- Loading/creating document-term matrices
- Initializing a Poisson Factorization (PF) model
- Training the model
- Extracting topics and results
- Demonstrating reproducibility with seeds

**Best for:**
- Users new to topic modeling
- Understanding basic workflow
- Learning the library API

**Output:**
- Trained model with topics
- Top words for each topic
- Loss trajectory during training

---

### Example 2: Seeded Poisson Factorization (`02_spf_keywords.py`)

**Level:** Intermediate
**Duration:** ~10 minutes
**Topics Covered:**
- Defining domain-specific vocabulary
- Creating seed words (keywords) for topics
- Guided topic discovery with SPF
- Comparing guided vs. unsupervised approaches

**Best for:**
- Users with domain knowledge
- Incorporating expert guidance
- Steering topics toward interpretable concepts

**Output:**
- Guided topics influenced by seed words
- Comparison with unsupervised PF
- Demonstration of keyword effects

---

### Example 3: Covariate Modeling (`03_cpf_covariates.py`)

**Level:** Intermediate
**Duration:** ~10 minutes
**Topics Covered:**
- Working with document-level covariates
- Using pandas DataFrames for covariates
- Covariate Poisson Factorization (CPF)
- Analyzing how topics vary with metadata
- Interpreting covariate effects

**Best for:**
- Users with document metadata
- Understanding covariate effects on topics
- Modeling document attributes

**Output:**
- Topic-covariate relationships
- Covariate effects matrix
- Scenario-based topic predictions

---

### Example 4: Advanced Combined Modeling (`04_advanced_cspf.py`)

**Level:** Advanced
**Duration:** ~15 minutes
**Topics Covered:**
- Combining keywords AND covariates (CSPF)
- Comprehensive model comparison (PF vs SPF vs CPF vs CSPF)
- Model selection guidelines
- Performance analysis
- Best practices for topic modeling

**Best for:**
- Users wanting maximum model flexibility
- Comparing different modeling approaches
- Production workflows
- Making informed model selection

**Output:**
- Comparison of 4 different models
- Loss trajectories
- Model selection recommendations
- Best practices guide

---

## Data Format Requirements

All examples create synthetic data, but here's what real data should look like:

### Document-Term Matrix
- **Format:** scipy.sparse.csr_matrix (recommended)
- **Shape:** (D, V) where D = documents, V = vocabulary size
- **Values:** Non-negative counts (word frequencies)
- **Sparsity:** Typically 95%+ sparse for text data

```python
import scipy.sparse as sparse
import numpy as np

# Load from file or create
counts = sparse.load_npz("document_term_matrix.npz")
# or
counts = sparse.random(100, 1000, density=0.05, format="csr")
```

### Vocabulary
- **Format:** numpy array of strings
- **Shape:** (V,) where V = vocabulary size
- **Content:** Word terms corresponding to matrix columns

```python
vocab = np.array(["word_0", "word_1", "word_2", ...])
```

### Keywords (for SPF)
- **Format:** Dictionary mapping topic IDs to word lists
- **Usage:** Guide the model toward specific topics

```python
keywords = {
    0: ["climate", "weather", "environment"],
    1: ["economy", "trade", "market"],
}
```

### Covariates (for CPF)
- **Format:** numpy array or pandas DataFrame
- **Shape:** (D, C) where C = number of covariates
- **Values:** Continuous features (e.g., -1 to 1)

```python
import pandas as pd
import numpy as np

covariates = np.random.randn(100, 3)  # numpy array
# or
covariates = pd.DataFrame(np.random.randn(100, 3),
                         columns=["feature_1", "feature_2", "feature_3"])
```

---

## Common Workflows

### Workflow 1: Quick Exploration
```bash
# Get started with unsupervised topic discovery
python examples/01_getting_started.py
```

### Workflow 2: Guided Discovery
```bash
# Add domain knowledge through keywords
python examples/02_spf_keywords.py
```

### Workflow 3: Metadata Integration
```bash
# Incorporate document metadata
python examples/03_cpf_covariates.py
```

### Workflow 4: Full Analysis
```bash
# Compare all approaches
python examples/04_advanced_cspf.py
```

---

## Customizing Examples

### Change Number of Topics
```python
num_topics = 10  # Increase from default 5
model = PF(counts, vocab, num_topics=num_topics, batch_size=batch_size)
```

### Change Training Parameters
```python
num_steps = 100  # More training steps
learning_rate = 0.001  # Lower learning rate
model.train_step(num_steps=num_steps, lr=learning_rate, random_seed=42)
```

### Use Your Own Data
```python
# Load your data
counts = sparse.load_npz("my_data.npz")
vocab = np.load("my_vocab.npy")

# Use in example
model = PF(counts, vocab, num_topics=5, batch_size=10)
```

---

## Troubleshooting

### Q: Examples import fails
**A:** Make sure topicmodels is installed: `pip install -e .`

### Q: JAX/NumPyro errors
**A:** Install dependencies: `pip install jax numpyro optax`

### Q: Out of memory
**A:** Reduce batch_size or use smaller datasets in examples

### Q: Training is slow
**A:** Normal on CPU - reduce num_steps for testing, or use GPU/Metal

---

## Model Selection Guide

| Model | Guidance | Covariates | When to Use |
|-------|----------|-----------|------------|
| **PF** | No | No | Unsupervised exploration |
| **SPF** | Yes | No | When you have domain knowledge |
| **CPF** | No | Yes | When you have document metadata |
| **CSPF** | Yes | Yes | Maximum flexibility |

---

## Best Practices

1. **Always set random_seed for reproducibility**
   ```python
   model.train_step(num_steps=50, lr=0.01, random_seed=42)
   ```

2. **Start with fewer topics and scale up**
   ```python
   num_topics = 5  # Start small, increase if needed
   ```

3. **Monitor loss during training**
   ```python
   print(f"Loss: {model.Metrics.loss[-1]:.4f}")
   ```

4. **Validate results on held-out data**
   ```python
   # Split data before training
   train_counts, test_counts = split_data(counts)
   ```

5. **Compare multiple models**
   ```python
   # Try different approaches
   for model_type in [PF, SPF, CPF]:
       model = model_type(...)
   ```

---

## Running All Examples

```bash
# Run each example in sequence
python examples/01_getting_started.py
python examples/02_spf_keywords.py
python examples/03_cpf_covariates.py
python examples/04_advanced_cspf.py
```

---

## Additional Resources

- **Documentation:** See main README.md
- **Tests:** See tests/ directory for more code examples
- **API Reference:** See package docstrings
- **Paper:** See CITATION.cff for publication details

---

## Questions?

For more information, refer to:
- Main README.md - Overview and installation
- Package documentation - API reference
- Test files - Integration examples
- Issue tracker - Known issues and solutions

---

**Last Updated:** November 19, 2025
