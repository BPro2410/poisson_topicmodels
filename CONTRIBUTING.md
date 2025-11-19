# Contributing to topicmodels

Thank you for your interest in contributing to **topicmodels**! This document provides guidelines and instructions for contributing.

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to Contribute

### 1. Report Bugs

If you find a bug, please open an issue with:

- **Title:** Clear, descriptive title
- **Description:** What did you try? What happened? What did you expect?
- **Minimal example:** Code that reproduces the issue
- **Environment:** Python version, OS, JAX/NumPyro versions
- **Traceback:** Full error message

**Example:**
```
Title: SPF model crashes with empty keywords dictionary

Description:
When initializing SPF with an empty keywords dict, the model crashes with IndexError.

Minimal example:
```python
from topicmodels import SPF
keywords = {}  # Empty dict
model = SPF(counts, vocab, keywords, residual_topics=2, batch_size=32)
```

Error: IndexError: tuple index out of range
```

### 2. Request Features

Feature requests are welcome! Please open an issue with:

- **Title:** Feature description (e.g., "Support for GPU memory optimization")
- **Motivation:** Why is this useful? What problem does it solve?
- **Proposed implementation:** (Optional) Rough idea of how it might work
- **Alternatives:** Any workarounds currently?

### 3. Submit Pull Requests

#### Before You Start

1. **Fork the repository** to your GitHub account
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/topicmodels_package.git
   cd topicmodels_package
   ```
3. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
4. **Install in development mode:**
   ```bash
   pip install -e ".[dev,docs]"
   ```

#### Making Changes

1. **Create a feature branch:**
   ```bash
   git checkout -b fix/issue-123-short-description
   # or
   git checkout -b feature/new-model-implementation
   ```

2. **Make your changes:**
   - Keep commits atomic (one logical change per commit)
   - Write descriptive commit messages
   - Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide

3. **Code style requirements:**

   - **Formatting:** Use `black` (line length: 100)
     ```bash
     black topicmodels tests
     ```
   
   - **Import organization:** Use `isort`
     ```bash
     isort topicmodels tests
     ```
   
   - **Linting:** Check with `flake8`
     ```bash
     flake8 topicmodels tests --max-line-length=100
     ```
   
   - **Type checking:** Use `mypy` (where applicable)
     ```bash
     mypy topicmodels --ignore-missing-imports
     ```

4. **Write tests:**
   - Add unit tests for new functionality in `tests/`
   - Aim for >80% code coverage on new code
   - Test edge cases and error conditions
   
   **Example test file structure:**
   ```python
   """Tests for new_module."""
   
   import pytest
   import numpy as np
   from topicmodels.models.new_module import NewModel
   
   
   class TestNewModel:
       """Test suite for NewModel."""
       
       @pytest.fixture
       def sample_data(self):
           """Fixture: sample input data."""
           counts = sparse.random(10, 20, density=0.5, format='csr')
           vocab = np.array([f'word_{i}' for i in range(20)])
           return counts, vocab
       
       def test_initialization(self, sample_data):
           """Test model initializes with valid inputs."""
           counts, vocab = sample_data
           model = NewModel(counts, vocab, param=5)
           assert model.param == 5
       
       def test_invalid_param(self, sample_data):
           """Test model rejects invalid parameters."""
           counts, vocab = sample_data
           with pytest.raises(ValueError):
               NewModel(counts, vocab, param=-1)  # Invalid negative param
   ```

5. **Run tests locally:**
   ```bash
   pytest tests/ -v --cov=topicmodels --cov-report=term-only
   ```

6. **Update documentation:**
   - Add docstrings following [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html)
   - Update relevant `.rst` files in `docs/`
   - Update `CHANGELOG.md` with your changes

#### Documentation Style

All functions/classes must have NumPy-style docstrings:

```python
def train_step(
    self,
    num_steps: int,
    lr: float,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train the model using stochastic variational inference.

    Parameters
    ----------
    num_steps : int
        Number of training iterations. Must be > 0.
    lr : float
        Learning rate for the optimizer. Must be > 0.
    random_seed : int, optional
        Random seed for reproducibility. If None, no fixed seed is set.
        Default is None.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'parameters': Estimated model parameters
        - 'loss': Training loss history
        - 'n_steps': Number of steps completed

    Raises
    ------
    ValueError
        If num_steps <= 0 or lr <= 0.
    
    Examples
    --------
    >>> model = PF(counts, vocab, num_topics=10, batch_size=128)
    >>> params = model.train_step(num_steps=100, lr=0.01, random_seed=42)
    >>> print(f"Training complete. Final loss: {params['loss'][-1]}")
    """
```

#### Submitting Your PR

1. **Push to your fork:**
   ```bash
   git push origin fix/issue-123-short-description
   ```

2. **Create a Pull Request on GitHub:**
   - Link related issues: "Closes #123" in PR description
   - Describe what changes you made and why
   - Reference any related discussions or PRs
   - Include before/after examples if applicable

3. **PR Description Template:**
   ```markdown
   ## Description
   Brief description of changes.

   ## Related Issue
   Closes #123

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Performance improvement
   - [ ] Documentation update

   ## How Has This Been Tested?
   Describe testing approach:
   - [ ] Unit tests added
   - [ ] Integration tests passed
   - [ ] Manual testing (describe)

   ## Checklist
   - [ ] Code follows style guide (black, isort, flake8)
   - [ ] Documentation updated
   - [ ] Tests added/updated
   - [ ] All tests passing locally
   - [ ] CHANGELOG updated
   ```

4. **Address feedback:**
   - Review CI/CD results
   - Make requested changes
   - Push additional commits (no force push unless requested)

5. **Merge:**
   - Maintainers will merge once approved
   - Your contribution will be acknowledged in CHANGELOG

## Development Workflow

### Running Full Test Suite

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all checks
black topicmodels tests                          # Format code
isort topicmodels tests                          # Organize imports
flake8 topicmodels tests --max-line-length=100   # Lint
mypy topicmodels --ignore-missing-imports        # Type check
pytest tests/ -v --cov=topicmodels               # Run tests with coverage
```

### Building Documentation

```bash
pip install -e ".[docs]"
cd docs
make html  # Build HTML documentation
# Then open _build/html/index.html
```

### Debugging

Use `pytest` with debugging flags:

```bash
# Show print statements
pytest tests/ -v -s

# Drop into debugger on failure
pytest tests/ --pdb

# Run specific test
pytest tests/test_models.py::TestPFModel::test_initialization -v

# Run tests matching pattern
pytest tests/ -k "test_pf" -v
```

## Project Structure

```
topicmodels_package/
├── topicmodels/              # Main package
│   ├── __init__.py          # Package exports
│   ├── models/              # Model implementations
│   │   ├── __init__.py
│   │   ├── numpyro_model.py # Abstract base class
│   │   ├── PF.py            # Poisson Factorization
│   │   ├── SPF.py           # Seeded PF
│   │   ├── CPF.py           # Covariate PF
│   │   ├── CSPF.py          # Covariate Seeded PF
│   │   ├── TBIP.py          # Text-Based Ideal Points
│   │   ├── ETM.py           # Embedded Topic Models
│   │   ├── Metrics.py       # Metrics tracking
│   │   └── topicmodels.py   # Factory function (being refactored)
│   └── utils/               # Utility functions
│       ├── __init__.py
│       └── utils.py         # Helpers (embeddings, etc.)
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── test_models.py       # Model tests
│   └── test_utils.py        # Utility tests
├── docs/                     # Sphinx documentation
│   ├── conf.py
│   ├── index.rst
│   └── intro/
├── data/                     # Example datasets
│   └── 10k_amazon.csv
├── examples/                 # Example notebooks
│   └── tutorial_spf.ipynb
├── pyproject.toml            # Project metadata
├── requirements.txt          # Dependencies
├── LICENSE                   # MIT License
├── CITATION.cff             # Citation metadata
├── CONTRIBUTING.md          # This file
├── CODE_OF_CONDUCT.md       # Community guidelines
└── CHANGELOG.md             # Version history
```

## Guidelines & Best Practices

### Model Implementation

When implementing a new topic model:

1. **Inherit from `NumpyroModel`:**
   ```python
   from topicmodels.models import NumpyroModel
   
   class NewModel(NumpyroModel):
       """Documentation of your model."""
   ```

2. **Implement required methods:**
   ```python
   def _model(self, Y_batch, d_batch):
       """NumPyro model definition."""
       pass
   
   def _guide(self, Y_batch, d_batch):
       """NumPyro variational guide."""
       pass
   ```

3. **Add comprehensive docstrings** (NumPy style)

4. **Include type hints** on all methods

5. **Write at least 5 unit tests** covering:
   - Valid initialization
   - Invalid parameter handling
   - Training step execution
   - Output shape/type verification
   - Reproducibility with fixed seed

### Utility Functions

For helper functions in `utils/`:

1. Keep functions focused and single-purpose
2. Add type hints
3. Include NumPy-style docstrings
4. Write tests with edge cases

## Reporting Security Issues

If you discover a security vulnerability, please email **b.prostmaier@icloud.com** instead of using the issue tracker. Do not disclose the issue publicly until a fix is available.

## Questions?

- **Documentation:** See [docs/](docs/)
- **Issues:** Create a GitHub issue labeled "question"
- **Discussions:** Use GitHub Discussions for general topics
- **Email:** b.prostmaier@icloud.com

## Recognition

Contributors are recognized in:
- Pull request comments
- `CHANGELOG.md`
- GitHub contributors page

Thank you for helping improve topicmodels! 🚀
