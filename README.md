# topicmodels: Probabilistic Topic Modeling with Bayesian Inference

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/topicmodels.svg)](https://pypi.org/project/seededPF/)
[![codecov](https://codecov.io/gh/BPro2410/topicmodels_package/branch/main/graph/badge.svg)](https://codecov.io/gh/BPro2410/topicmodels_package)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**topicmodels** is a modern Python package for probabilistic topic modeling using Bayesian inference, built on [JAX](https://github.com/google/jax) and [NumPyro](https://github.com/pyro-ppl/numpyro).

## Statement of Need

Traditional topic modeling packages (e.g., Gensim, scikit-learn's LDA) use older inference methods and lack flexibility for emerging research needs. **poisson-topicmodels** addresses key gaps:

1. **Modern Probabilistic Inference**: Built on NumPyro, enabling automatic differentiation, probabilistic programming, and integration with cutting-edge Bayesian methods.

2. **Advanced Topic Models**: Goes beyond LDA with guided topic discovery (keyword priors), covariate effects, ideal point estimation, and embeddings—all with principled Bayesian inference.

3. **GPU Acceleration**: Leverages JAX for transparent GPU computation, essential for large-scale corpus analysis and enabling research that would be prohibitively slow on CPU.

4. **Scalability & Reproducibility**: Optimized for mini-batch SVI training with built-in seed control for exact reproducibility—critical for research validation and publication.

5. **Research-Friendly API**: Purpose-built for computational social science and NLP researchers who need interpretable, flexible models beyond black-box approaches.

Whether analyzing legislative text, social media discourse, or scientific abstracts, **poisson-topicmodels** enables researchers to extract interpretable semantic structure with confidence in results.

## Features

**poisson-topicmodels** provides multiple topic modeling approaches:

| Model | Use Case | Key Feature |
|-------|----------|------------|
| **Poisson Factorization (PF)** | Unsupervised baseline | Fast, interpretable word-topic associations |
| **Seeded PF (SPF)** | Guided discovery | Incorporate domain knowledge via keyword priors |
| **Covariate PF (CPF)** | Covariate effects | Model topics influenced by document metadata |
| **Covariate Seeded PF (CSPF)** | Guided + covariates | Combine keyword guidance with external factors |
| **Text-Based Ideal Points (TBIP)** | Ideal point estimation | Estimate author positions from legislative/social text |
| **Embedded Topic Models (ETM)** | Modern embeddings | Integrate pre-trained word embeddings |

**Core Capabilities**:
- ✨ Stochastic Variational Inference (SVI) with mini-batch training
- ✨ Transparent GPU acceleration via JAX
- ✨ Reproducible results with seed control
- ✨ Type hints and comprehensive API documentation
- ✨ >70% test coverage with continuous integration
- ✨ Clear error messages and input validation

## Comparison with Existing Tools

| Feature | topicmodels | Gensim | scikit-learn | BTM |
|---------|-------------|--------|--------------|-----|
| GPU Support | ✅ JAX backend | ❌ | ❌ | ❌ |
| Seeded Topics | ✅ SPF, CSPF | ⚠️ Limited | ❌ | ✅ |
| Covariate Effects | ✅ CPF, CSPF | ❌ | ❌ | ❌ |
| Ideal Points | ✅ TBIP | ❌ | ❌ | ❌ |
| Embeddings | ✅ ETM | ⚠️ Limited | ❌ | ❌ |
| Type Hints | ✅ Full | ⚠️ Partial | ✅ Full | ❌ |
| Active Development | ✅ Modern stack | ⚠️ Mature | ✅ Active | ⚠️ Limited |
| Research-Focused | ✅ By researchers | ⚠️ General-purpose | ⚠️ General | ✅ |


## Quick Start

Get started in 5 minutes:

```python
import numpy as np
from scipy.sparse import csr_matrix
from poisson_topicmodels import PF

# Prepare data: document-term matrix and vocabulary
counts = csr_matrix(np.random.poisson(2, (100, 500)).astype(np.float32))
vocab = np.array([f'word_{i}' for i in range(500)])

# Initialize and train model
model = PF(counts, vocab, num_topics=10, batch_size=32)
params = model.train_step(num_steps=100, lr=0.01, random_seed=42)

# Extract results
topics = model.return_topics()
top_words = model.return_top_words_per_topic(n_words=10)
print(f"Found {topics.shape[1]} topics")
print(f"Top words: {top_words[:3]}")
```

See `examples/` directory for detailed notebooks.

## Installation

### From PyPI (recommended)
```bash
pip install poisson-topicmodels
```

### From Source
```bash
git clone https://github.com/BPro2410/topicmodels_package.git
cd topicmodels_package
pip install -e .
```

### Development Setup
```bash
git clone https://github.com/BPro2410/topicmodels_package.git
cd topicmodels_package
pip install -e ".[dev]"
pytest tests/  # Verify installation
```

## Requirements

- Python ≥ 3.11
- JAX ≥ 0.4.35 (with optional GPU support)
- NumPyro ≥ 0.15.3
- NumPy, SciPy, scikit-learn, pandas

See `pyproject.toml` for complete dependency list.

## Documentation

- **[API Reference](https://topicmodels.readthedocs.io)** – Complete model and method documentation
- **[User Guide](docs/intro/user_guide.rst)** – Detailed tutorials and workflows
- **[Examples](examples/)** – Jupyter notebooks demonstrating all features
- **[Contributing](CONTRIBUTING.md)** – How to contribute improvements

## Basic Usage Examples

### 1. Unsupervised Topic Discovery (PF)

```python
from poisson_topicmodels import PF

model = PF(counts, vocab, num_topics=10, batch_size=64)
model.train_step(num_steps=500, lr=0.001, random_seed=42)

# Extract topics
topics, doc_ids = model.return_topics()
top_words = model.return_top_words_per_topic(n_words=15)
```

### 2. Guided Topic Modeling with Keywords (SPF)

```python
from poisson_topicmodels import SPF

keywords = {
    0: ['climate', 'environment', 'carbon'],
    1: ['economy', 'growth', 'trade'],
}

model = SPF(counts, vocab, keywords, residual_topics=5, batch_size=64)
model.train_step(num_steps=500, lr=0.001, random_seed=42)
```

### 3. Covariate Effects (CPF)

```python
from poisson_topicmodels import CPF

# Include document-level covariates
covariates = np.random.randn(100, 3)  # 100 documents, 3 covariates

model = CPF(counts, vocab, covariates, num_topics=10, batch_size=64)
model.train_step(num_steps=500, lr=0.001, random_seed=42)
```

## Example Data

The repository includes `data/10k_amazon.csv` with ~10,000 Amazon product reviews for quick experimentation. See `examples/01_getting_started.ipynb` for a complete walkthrough.

## Docker Setup (Optional)

For a reproducible, isolated environment with JupyterLab:

```bash
# Build image
docker build -t topicmodels-jupyter .

# Run container (Linux/macOS)
docker run --rm -p 8888:8888 -v "$(pwd)":/workspace topicmodels-jupyter

# Then open http://localhost:8888 in your browser
```

## Citation

If you use **poisson_topicmodels** in your research, please cite:

```bibtex
@software{topicmodels2025,
  title = {Poisson-topicmodels: Probabilistic Topic Modeling with Bayesian Inference},
  author = {Prostmaier, Bernd and Grün, Bettina and Hofmarcher, Paul},
  year = {2025},
  url = {https://github.com/BPro2410/topicmodels_package},
}
```

See `CITATION.cff` for additional citation formats.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Reporting bugs
- Submitting pull requests
- Code style and testing requirements
- Documentation standards

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Support

- **Issues & Bug Reports**: [GitHub Issues](https://github.com/BPro2410/topicmodels_package/issues)
- **Discussions**: [GitHub Discussions](https://github.com/BPro2410/topicmodels_package/discussions)
- **Documentation**: [ReadTheDocs](https://topicmodels.readthedocs.io)

---

**Built with ❤️ for researchers and practitioners in computational social science and NLP**
