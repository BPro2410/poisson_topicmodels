# Implementation Roadmap: Priority Actions

## Overview

This document provides a clear, step-by-step roadmap for implementing the recommendations from `PUBLICATION_AUDIT.md`.

**Total estimated effort:** 60-80 hours
**Target completion:** 8 weeks at 8 hours/week

---

## Phase 1: Critical Blockers (Week 1-2)
**Effort:** 8-10 hours
**Blocking:** Everything else

### 1.1 Create LICENSE (30 min)

**File:** `LICENSE`

```bash
# Choose license type and create file
# Recommended: MIT (simple, permissive) or GPL-3.0 (copyleft)
```

**Acceptance criteria:**
- [ ] `LICENSE` file exists
- [ ] File contains full license text
- [ ] License field added to `pyproject.toml`: `license = {text = "MIT"}`

---

### 1.2 Create CITATION.cff (30 min)

**File:** `CITATION.cff` (already created)

**Acceptance criteria:**
- [ ] File exists in repository root
- [ ] All author information filled
- [ ] Affiliations added
- [ ] GitHub recognizes it (shows "Cite this repository" button)

---

### 1.3 Fix Dependency Conflicts (1 hour)

**Files:** `pyproject.toml`, `requirements.txt`

**Current issue:**
- `pyproject.toml` specifies `jax>=0.8.0,<0.9.0` but `requirements.txt` pins `jax==0.4.35`
- These are incompatible versions

**Action:** Choose one approach:

**Approach A: Use tested versions (RECOMMENDED)**
```toml
[project]
dependencies = [
    "jax==0.4.35",
    "jaxlib==0.4.35",
    "numpyro==0.15.3",
    "optax==0.2.4",
    "numpy==2.2.4",
    "scipy==1.15.2",
    "pandas==2.2.3",
    "scikit-learn>=1.6.0,<2.0.0",
    "matplotlib>=3.10.0,<4.0.0",
    "tqdm==4.66.6",
    "flax==0.8.4",
]
```

**Approach B: Use newer versions (requires testing)**
- Test locally with updated `requirements.txt`
- Ensure all models work correctly
- Update both files to newer versions

**Acceptance criteria:**
- [ ] `pyproject.toml` and `requirements.txt` versions align
- [ ] Test installation succeeds: `pip install -e .`
- [ ] Can import all models: `from topicmodels import PF, SPF, CPF`
- [ ] Example script runs without dependency errors

---

### 1.4 Restructure Package (2 hours)

**Problem:** Package structure `packages/models/` is non-standard

**Action:**
```bash
# 1. Create new structure
mkdir -p topicmodels/models
mkdir -p topicmodels/utils

# 2. Move files
mv packages/models/*.py topicmodels/models/
mv packages/utils/* topicmodels/utils/

# 3. Update imports in moved files
# In each .py file:
#   Replace: from packages.models.X import Y
#   With:    from topicmodels.models.X import Y

# 4. Create __init__.py files
# topicmodels/__init__.py (see below)
# topicmodels/models/__init__.py (with all exports)
# topicmodels/utils/__init__.py (with utils exports)

# 5. Remove old structure
rm -rf packages/
```

**New `topicmodels/__init__.py`:**
```python
"""topicmodels: Probabilistic topic modeling with Bayesian inference."""

__version__ = "0.1.0"

from .models import (
    PF, SPF, CPF, CSPF, TBIP, ETM,
    NumpyroModel, Metrics
)

__all__ = [
    "PF", "SPF", "CPF", "CSPF", "TBIP", "ETM",
    "NumpyroModel", "Metrics",
]
```

**New `topicmodels/models/__init__.py`:**
```python
from .PF import PF
from .SPF import SPF
from .CPF import CPF
from .CSPF import CSPF
from .TBIP import TBIP
from .ETM import ETM
from .numpyro_model import NumpyroModel
from .Metrics import Metrics

__all__ = ['PF', 'SPF', 'CPF', 'CSPF', 'TBIP', 'ETM', 'NumpyroModel', 'Metrics']
```

**Acceptance criteria:**
- [ ] `topicmodels/` directory structure created
- [ ] All `.py` files moved to correct locations
- [ ] Imports updated in all moved files
- [ ] `from topicmodels import PF` works
- [ ] Old `packages/` directory removed
- [ ] `run_topicmodels.py` updated to use new import paths
- [ ] All models can be instantiated

---

### 1.5 Update Root Package Exports (30 min)

**File:** `topicmodels/__init__.py`

Already created in step 1.4. Ensure it's correct.

**Acceptance criteria:**
- [ ] Can import from root: `from topicmodels import PF`
- [ ] IDE autocompletion works
- [ ] `__all__` is defined
- [ ] `__version__` is accessible: `import topicmodels; print(topicmodels.__version__)`

---

### 1.6 Create Basic Test Suite (3 hours)

**File:** `tests/test_basic_models.py`

Create basic unit tests covering:
- Model initialization
- Parameter validation
- Training step execution
- Output shape verification

**Minimum test coverage:**
```python
tests/
├── __init__.py
├── test_pf.py           # 5-10 tests
├── test_spf.py          # 5-10 tests
├── test_cpf.py          # 5-10 tests
├── test_tbip.py         # 5-10 tests
└── fixtures.py          # Shared fixtures
```

**Sample test structure:**
```python
import pytest
import numpy as np
import scipy.sparse as sparse
from topicmodels import PF

@pytest.fixture
def sample_data():
    counts = sparse.random(10, 20, density=0.3, format='csr', dtype=np.float32)
    vocab = np.array([f'word_{i}' for i in range(20)])
    return counts, vocab

def test_pf_initialization(sample_data):
    counts, vocab = sample_data
    model = PF(counts, vocab, num_topics=5, batch_size=4)
    assert model.K == 5
    assert model.D == 10
    assert model.V == 20
```

**Acceptance criteria:**
- [ ] `tests/` directory created
- [ ] At least 25 basic tests written
- [ ] All tests pass locally: `pytest tests/`
- [ ] Coverage reported: `pytest --cov=topicmodels`

---

### 1.7 Set Up CI/CD (1 hour)

**File:** `.github/workflows/tests.yml`

```yaml
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12', '3.13']

    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - run: pip install -e ".[dev]"
    - run: pytest tests/ -v --cov=topicmodels
```

**Acceptance criteria:**
- [ ] `.github/workflows/tests.yml` created
- [ ] Workflow runs on push to main
- [ ] All test jobs pass
- [ ] Coverage badge displays on repo

---

### Phase 1 Verification Checklist

After completing Phase 1, verify:

```bash
# 1. Check structure
ls -la topicmodels/
ls -la topicmodels/models/
ls -la tests/

# 2. Test imports
python -c "from topicmodels import PF, SPF, CPF; print('✓ Imports work')"

# 3. Run tests
pytest tests/ -v

# 4. Check dependencies
python -c "import jax; import numpyro; print('✓ Dependencies OK')"

# 5. Verify package is installable
pip install -e .
```

**Expected output:** All commands succeed, no errors.

---

## Phase 2: High-Impact Documentation (Week 3-4)
**Effort:** 10-12 hours
**Blocking:** JOSS/JMLR submission

### 2.1 Update README (2 hours)

**File:** `README.md`

Use `README_TEMPLATE.md` as basis. Add:

1. **Statement of Need** (2 paragraphs)
   - Why use this over Gensim/Scikit-learn?
   - What problems does it solve?

2. **Comparison Table**
   - Features vs. competing tools
   - See template for example

3. **Quick Start** (minimal working example)
   - ~15 lines of code
   - Shows: load data → init model → train → results

4. **Installation** (clear, tested instructions)
   - PyPI (future)
   - From source
   - Development setup

5. **Documentation Links**
   - User guide
   - API reference
   - Examples/tutorials

**Acceptance criteria:**
- [ ] README under 5 KB
- [ ] GitHub renders properly
- [ ] Quick start example runs
- [ ] All links work

---

### 2.2 Create/Update CONTRIBUTING.md (1 hour)

**File:** `CONTRIBUTING.md` (already created)

Ensure it includes:
- Bug reporting template
- PR workflow
- Code style requirements (black, isort, flake8)
- Testing expectations
- Documentation standards

**Acceptance criteria:**
- [ ] File exists
- [ ] GitHub recognizes it
- [ ] Clear instructions for developers
- [ ] Links to relevant resources

---

### 2.3 Complete API Documentation (3 hours)

**Files:** Model docstrings, `docs/` RST files

For each model class:
1. Update class docstring (2-3 sentences)
2. Document all `__init__` parameters
3. Document all methods (parameter types, return types)
4. Add examples section

**Example format (NumPy style):**
```python
class PF(NumpyroModel):
    """
    Poisson Factorization (PF) topic model.

    Unsupervised baseline topic model using Poisson likelihood
    for word counts. Suitable for exploratory topic discovery.

    Parameters
    ----------
    counts : scipy.sparse.csr_matrix
        Document-term matrix of shape (D, V).
    vocab : np.ndarray
        Vocabulary array of shape (V,).
    num_topics : int
        Number of topics K. Must be > 0.
    batch_size : int
        Mini-batch size for SVI. Must be ≤ D.

    Examples
    --------
    >>> model = PF(counts, vocab, num_topics=10, batch_size=32)
    >>> params = model.train_step(num_steps=100, lr=0.01)
    """
```

**Acceptance criteria:**
- [ ] All model classes documented
- [ ] All public methods have complete docstrings
- [ ] Return types specified
- [ ] Examples included
- [ ] Sphinx autodoc builds without errors

---

### 2.4 Add Type Hints (2 hours)

**Files:** All Python files in `topicmodels/`

Add type hints to all method signatures:

```python
from typing import Optional, Dict, Any, Tuple
import scipy.sparse as sparse

def train_step(
    self,
    num_steps: int,
    lr: float,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Train model."""
    pass

def return_topics(self) -> Tuple[np.ndarray, np.ndarray]:
    """Return topic assignments."""
    pass
```

**Acceptance criteria:**
- [ ] All public methods have type hints
- [ ] No `Any` used without reason
- [ ] `mypy --ignore-missing-imports topicmodels/` runs clean

---

### 2.5 Update Docs Build (1 hour)

**Files:** `docs/conf.py`, `docs/index.rst`, RST files

Ensure:
- Autodoc is configured correctly
- All modules documented
- Build produces valid HTML
- No warnings on build

```bash
cd docs
make clean
make html
# Check for errors/warnings
```

**Acceptance criteria:**
- [ ] `docs/_build/html/index.html` exists
- [ ] No build warnings
- [ ] All models appear in module list
- [ ] API reference is complete

---

## Phase 3: Code Quality & Testing (Week 5-6)
**Effort:** 12-15 hours

### 3.1 Expand Test Suite (4 hours)

Target 70%+ coverage:

```bash
# Current state (after Phase 1): ~30% coverage
# Goal: ~70% coverage

# Test structure:
tests/
├── unit/              # Unit tests for individual functions
│   ├── test_pf.py
│   ├── test_spf.py
│   ├── test_cpf.py
│   ├── test_cspf.py
│   ├── test_tbip.py
│   ├── test_etm.py
│   └── test_utils.py
├── integration/       # Integration tests
│   ├── test_workflows.py
│   └── test_reproducibility.py
├── fixtures.py        # Shared fixtures
└── conftest.py       # pytest configuration
```

**Key test areas:**
- [ ] All model initialization variants
- [ ] Parameter validation (invalid inputs)
- [ ] Training with different seeds (reproducibility)
- [ ] Output shapes and types
- [ ] Edge cases (single document, sparse data, etc.)
- [ ] GPU/CPU compatibility (if applicable)

**Acceptance criteria:**
- [ ] Coverage ≥70%: `pytest --cov=topicmodels --cov-report=term`
- [ ] All tests pass
- [ ] Edge cases covered
- [ ] Performance tests (optional)

---

### 3.2 Add Code Quality Checks (2 hours)

**Files:** `.pre-commit-config.yaml`, `pyproject.toml`

Set up automated checks:

```bash
# 1. Install tools
pip install black isort flake8 mypy pylint

# 2. Run all checks
black topicmodels tests
isort topicmodels tests
flake8 topicmodels tests --max-line-length=100
mypy topicmodels --ignore-missing-imports

# 3. Optional: Set up pre-commit hooks
pip install pre-commit
# Create .pre-commit-config.yaml
pre-commit install
```

**Acceptance criteria:**
- [ ] `black` formatting consistent
- [ ] `isort` imports organized
- [ ] `flake8` returns no errors
- [ ] `mypy` passes on main code
- [ ] Pre-commit hooks configured (optional)

---

### 3.3 Fix Shared Mutable State (1 hour)

**File:** `topicmodels/models/numpyro_model.py`

**Problem:**
```python
class NumpyroModel(ABC):
    Metrics = Metrics(loss=list())  # WRONG! Shared across instances
```

**Fix:**
```python
class NumpyroModel(ABC):
    def __init__(self, ...):
        self.Metrics = Metrics(loss=[])  # Per-instance
```

**Acceptance criteria:**
- [ ] Each model instance has separate Metrics
- [ ] Training one model doesn't affect another
- [ ] Loss tracking works correctly

---

### 3.4 Add Input Validation (2 hours)

Add validation to all model `__init__` methods:

```python
def __init__(self, counts, vocab, num_topics, batch_size):
    # Validate counts
    if not sparse.issparse(counts):
        raise TypeError("counts must be a scipy sparse matrix")
    if counts.shape[0] == 0 or counts.shape[1] == 0:
        raise ValueError("counts matrix is empty")

    # Validate parameters
    if num_topics <= 0:
        raise ValueError(f"num_topics must be > 0, got {num_topics}")
    if batch_size <= 0 or batch_size > counts.shape[0]:
        raise ValueError(f"batch_size must be between 1 and {counts.shape[0]}")

    # Store validated inputs
    self.counts = counts
    self.vocab = vocab
    self.K = num_topics
    self.batch_size = batch_size
```

**Acceptance criteria:**
- [ ] All models validate inputs
- [ ] Clear error messages
- [ ] Tests verify validation works

---

### 3.5 Add Reproducibility Features (2 hours)

**File:** `topicmodels/models/numpyro_model.py`

Add seed parameter to `train_step()`:

```python
def train_step(
    self,
    num_steps: int,
    lr: float,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Train model.

    Parameters
    ----------
    random_seed : int, optional
        Seed for JAX random number generator. If provided,
        ensures reproducible results. Default is None (random).
    """
    if random_seed is not None:
        rng = jax.random.PRNGKey(random_seed)
    else:
        rng = jax.random.PRNGKey(int(time.time()))

    # ... rest of training ...
```

**Acceptance criteria:**
- [ ] Same seed produces identical results
- [ ] Without seed, results vary
- [ ] Documented in docstrings
- [ ] Tests verify reproducibility

---

## Phase 4: Advanced Features & Refactoring (Week 7-8)
**Effort:** 12-15 hours

### 4.1 Refactor Factory Function (3 hours)

**Problem:** String-based dispatch is non-Pythonic

```python
# Current (bad)
tm = topicmodels("SPF", counts, vocab, keywords, residual_topics=0)

# Better
from topicmodels import SPF
tm = SPF(counts, vocab, keywords, residual_topics=0)
```

**Action:**
1. Update `run_topicmodels.py` examples to use direct imports
2. Deprecate factory function with warning
3. Remove factory function in v0.2.0

**Acceptance criteria:**
- [ ] Direct imports work for all models
- [ ] Factory function (if kept) shows deprecation warning
- [ ] Examples updated
- [ ] Documentation shows new approach

---

### 4.2 Add Performance Benchmarks (2 hours)

Create `benchmarks/benchmark_models.py`:

```python
"""Benchmarks for topic models."""

import time
import numpy as np
import scipy.sparse as sparse
from topicmodels import PF, SPF, TBIP

def benchmark_model(model_class, *args, **kwargs):
    """Benchmark a single model."""
    model = model_class(*args, **kwargs)

    start = time.time()
    model.train_step(num_steps=100, lr=0.01, random_seed=42)
    duration = time.time() - start

    return duration

# Results table
results = {
    'PF': benchmark_model(PF, counts, vocab, num_topics=10, batch_size=128),
    'SPF': benchmark_model(SPF, counts, vocab, keywords, residual_topics=2, batch_size=128),
}
```

**Acceptance criteria:**
- [ ] Benchmark script runs
- [ ] Results documented in README
- [ ] GPU vs CPU comparison (if applicable)

---

### 4.3 Create Example Notebooks (3 hours)

Create Jupyter notebooks in `examples/`:

```
examples/
├── 01_getting_started.ipynb          # Quick intro
├── 02_model_comparison.ipynb         # PF vs SPF vs CPF
├── 03_seeded_topics.ipynb            # SPF with keywords
├── 04_covariate_models.ipynb         # CPF/CSPF
└── 05_interpretation.ipynb           # Results analysis
```

Each notebook should include:
- Clear narrative explanations
- Runnable code cells
- Visualizations
- Export to both `.ipynb` and `.py`

**Acceptance criteria:**
- [ ] 3-5 notebooks created
- [ ] All notebooks run without errors
- [ ] Clear explanations
- [ ] Reproducible (fixed seeds)

---

### 4.4 Add Configuration File (1 hour)

**File:** `topicmodels/config.py`

```python
"""Configuration for topicmodels."""

# Default hyperparameters
DEFAULT_PRIORS = {
    'gamma_shape': 0.3,
    'gamma_rate': 0.3,
}

# Performance settings
DEVICE_MEMORY = {
    'cpu': 2048,  # MB
    'gpu': 8192,
}

# Model constants
MAX_BATCH_SIZE = 10000
MIN_VOCAB_SIZE = 10
```

**Acceptance criteria:**
- [ ] Config file created
- [ ] Used in model initialization
- [ ] Easy to customize
- [ ] Documented in README

---

## Phase 5: Final Review & Submission (Week 9)
**Effort:** 4-6 hours

### 5.1 Complete JOSS/JMLR Checklist

- [ ] LICENSE file present and valid
- [ ] CITATION.cff with author info
- [ ] README with Statement of Need
- [ ] Tests with >70% coverage
- [ ] CI/CD passing (GitHub Actions)
- [ ] Documentation complete
- [ ] Type hints on public API
- [ ] No critical dependencies conflicts
- [ ] Clear contribution guidelines
- [ ] Code of Conduct present

### 5.2 Final Documentation Review

- [ ] README accurate and complete
- [ ] API reference generated correctly
- [ ] Examples run without errors
- [ ] Installation instructions tested
- [ ] All links working

### 5.3 Create CHANGELOG

**File:** `CHANGELOG.md`

```markdown
# Changelog

## [0.1.0] - 2025-11-20

### Added
- Initial release
- PF, SPF, CPF, CSPF models
- TBIP model for ideal point estimation
- ETM model with embeddings
- Comprehensive test suite (>70% coverage)
- Documentation and API reference
- CI/CD pipeline with GitHub Actions

### Changed
- Restructured package from `packages/` to `topicmodels/`

### Fixed
- Dependency version conflicts
- Shared mutable state in Metrics class
```

---

### 5.4 Prepare Submission

**Before submitting to JOSS/JMLR:**

```bash
# 1. Final test run
pytest tests/ -v --cov=topicmodels

# 2. Code quality check
black --check topicmodels tests
flake8 topicmodels tests
mypy topicmodels

# 3. Documentation build
cd docs && make html && cd ..

# 4. Install and test package
pip uninstall topicmodels -y
pip install .
python -c "from topicmodels import *; print('✓ Package OK')"

# 5. Check GitHub badges/workflows
# Verify on GitHub:
#   - CI/CD passing
#   - Coverage reported
#   - Links working

# 6. Create release
git tag v0.1.0
git push origin v0.1.0
```

---

## Success Criteria Checklist

### ✅ Packaging
- [ ] Package installable via `pip install topicmodels`
- [ ] All dependencies specified correctly
- [ ] License file present
- [ ] CITATION.cff created
- [ ] Version number proper
- [ ] No deprecated dependencies

### ✅ Documentation
- [ ] README with Statement of Need (>500 words)
- [ ] API reference auto-generated
- [ ] At least 3 working examples/tutorials
- [ ] Contributing guide complete
- [ ] Code of Conduct present
- [ ] CHANGELOG maintained

### ✅ Testing
- [ ] Test coverage ≥70%
- [ ] All tests passing
- [ ] CI/CD workflow active
- [ ] GitHub Actions showing "passing" badge

### ✅ Code Quality
- [ ] Type hints on all public API
- [ ] No PEP 8 violations
- [ ] Docstrings NumPy style
- [ ] No code duplicates
- [ ] Input validation implemented

### ✅ Reproducibility
- [ ] Fixed random seeds work
- [ ] Environment specs clear
- [ ] Examples reproducible
- [ ] GPU/CPU support documented

### ✅ JOSS/JMLR Requirements
- [ ] Statement of Need clear
- [ ] Comparison to existing tools
- [ ] Author/contributor metadata
- [ ] Academic references included
- [ ] Community engagement plan

---

## Timeline Estimate

| Phase | Duration | Start | End |
|---|---|---|---|
| Phase 1: Critical | 8-10h | Week 1 | Week 2 |
| Phase 2: Documentation | 10-12h | Week 3 | Week 4 |
| Phase 3: Code Quality | 12-15h | Week 5 | Week 6 |
| Phase 4: Advanced | 12-15h | Week 7 | Week 8 |
| Phase 5: Final | 4-6h | Week 9 | Week 9 |
| **TOTAL** | **46-58h** | — | — |

*Assumes 6-8 hours/week of focused work*

---

## Next Step

**Start with Phase 1, Task 1.1:** Create LICENSE file

Then work through each task sequentially, verifying completion before moving to the next.

Good luck with publication! 🚀
