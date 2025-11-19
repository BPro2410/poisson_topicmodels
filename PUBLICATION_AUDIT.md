# Publication Readiness Audit: topicmodels Package

**Prepared for:** JOSS / JMLR Submission  
**Date:** November 2025  
**Auditor:** Comprehensive Repository Assessment

---

## Executive Summary

The `topicmodels` package presents a solid foundation with implemented topic modeling algorithms and reasonable initial structure. However, it requires **significant work** across multiple dimensions before publication in tier-1 venues like JOSS or JMLR. Key gaps include:

- **Critical:** No tests; no CI/CD; missing LICENSE; no CITATION.cff
- **High:** Incomplete documentation; poor API stability; version conflicts in dependencies
- **Medium:** Type hints missing; API design inconsistencies; code organization issues
- **Low:** Code style and complexity improvements

**Estimated effort:** 60-80 hours of focused work for publication-ready quality.

---

## Part 1: Detailed Findings

### 1. Packaging & Distribution Quality

#### ✅ What's Good
- `pyproject.toml` present with basic metadata
- Clear dependency declaration in both `pyproject.toml` and `requirements.txt`
- Reasonable version pinning strategy
- Python version constraints specified (3.11-3.13)

#### ❌ Critical Issues

1. **Version Conflict:** `requirements.txt` and `pyproject.toml` mismatch
   - `pyproject.toml`: `jax>=0.8.0,<0.9.0`, `numpyro>=0.19.0,<0.20.0`
   - `requirements.txt`: `jax==0.4.35`, `numpyro==0.15.3` (incompatible!)
   - This will cause installation failures

2. **Missing LICENSE file**
   - No LICENSE exists (critical for open-source)
   - No license identifier in `pyproject.toml`

3. **No CITATION.cff file**
   - Required for JMLR/JOSS - enables automatic citation

4. **Missing metadata in pyproject.toml**
   - No `license` field
   - No `repository` URL
   - No `keywords` array
   - No `classifiers` (e.g., "Development Status", "License")
   - No `documentation` URL

5. **Top-level `__init__.py` is commented out**
   - Package root exports are disabled
   - Users can't do `from topicmodels import PF`
   - Forces awkward `from packages.models import PF`

6. **`packages/__init__.py` is empty**
   - Subpackage exports not exposed

#### 📋 Code Organization Issues

- **Non-standard package structure:** `packages/models/` instead of `topicmodels/models/`
- **Project name mismatch:** `topicmodels_package` vs `topicmodels-package` (inconsistent naming)
- Utilities are exposed via `packages.utils` instead of top-level

---

### 2. API Design & Stability

#### ❌ Critical Issues

1. **Factory function `topicmodels()` design is non-standard**
   ```python
   tm = topicmodels("PF", counts, vocab, num_topics, batch_size)  # String-based dispatch
   ```
   - Fragile string-based instantiation
   - No IDE autocompletion
   - Error checking via string comparison is error-prone
   - Not Pythonic; violates PEP 20

   **Better approach:** Direct imports
   ```python
   from topicmodels import PF
   tm = PF(counts, vocab, num_topics=10, batch_size=1024)
   ```

2. **Inconsistent method signatures across models**
   - `PF.__init__`: `(counts, vocab, num_topics, batch_size)`
   - `SPF.__init__`: `(counts, vocab, keywords, residual_topics, batch_size)`
   - `TBIP.__init__`: includes `authors`, `time_varying`, `beta_shape_init`, `beta_rate_init`
   - No common interface documentation

3. **Unclear parameter semantics**
   - What is `residual_topics` in SPF/CSPF? (Not explained in docstring)
   - What format should `keywords` dictionary use?
   - What is the expected shape/type of `counts`? (Assumes CSR sparse, never enforced)

4. **Return types are undocumented**
   - `train_step()` returns what exactly? (Implied: dict, but not typed)
   - `return_topics()`, `return_beta()` return types undefined

#### ⚠️ Medium Issues

- No validation of input parameters
- No type hints on any method signatures (PEP 484)
- Inconsistent naming: `model()`, `_model()`, `_guide()` (underscore convention unclear)
- Class attributes stored at module-level (`Metrics = Metrics(loss = list())`) are shared across instances

---

### 3. Documentation Quality

#### ❌ Critical Gaps

1. **README is incomplete**
   - No "Statement of Need" (required for JOSS/JMLR)
   - No comparison to existing tools (Gensim, Scikit-learn, pyLDAvis, etc.)
   - No clear description of when/why to use each model
   - Installation section lacks version/compatibility warnings
   - No mention of GPU/JAX requirements

2. **No formal API Reference**
   - Sphinx docs (`docs/`) exist but appear incomplete
   - No auto-generated API docs for all models
   - Missing method documentation

3. **No User Guide or Tutorials**
   - `intro/user_guide.rst` exists but is likely incomplete
   - No examples of model interpretation
   - No visualization examples
   - No hyperparameter tuning guidance

4. **No Contribution Guide**
   - `intro/contributing.rst` exists but needs content
   - No code style guidelines
   - No testing requirements
   - No PR workflow documented

5. **No CHANGELOG**
   - No version history
   - No breaking changes documented

#### ⚠️ Medium Issues

- Example script (`run_topicmodels.py`) is not a proper tutorial
  - Lacks narrative explanation
  - Has too many commented sections
  - Not reproducible (no random seed control shown)
- Docstrings present but inconsistent in style
- No architecture documentation explaining the inheritance hierarchy

---

### 4. Testing & Quality Assurance

#### ❌ Critical Issues

1. **No test files exist**
   - Zero unit tests
   - No edge case coverage
   - No integration tests
   - No CI/CD pipeline

2. **No error handling**
   - No input validation
   - Silent failures likely
   - Example: sparse matrix shape not validated in `_get_batch()`

3. **Reproducibility concerns**
   - Random seeds hardcoded (`random.PRNGKey(0)`, `random.PRNGKey(1)`, `random.PRNGKey(2)`)
   - Users cannot set seeds for reproducible results
   - No seed configuration mechanism

#### ⚠️ Medium Issues

- No static analysis (mypy, pylint, flake8)
- No code formatting (black, isort)
- No linting configuration
- Type hints completely absent

---

### 5. Code Quality & Maintainability

#### ❌ Issues

1. **Type hints missing entirely**
   - No function signatures have types
   - Makes code harder to understand
   - IDE support compromised
   - Violates modern Python best practices (PEP 484)

2. **Docstring inconsistencies**
   - Some follow NumPy style, some are incomplete
   - Some parameters lack descriptions
   - Return types not always specified

3. **Code duplication**
   - Similar model implementations repeat patterns
   - `_get_batch()` method likely could be simplified
   - Hyperparameter initialization (Gamma(0.3, 0.3)) hardcoded everywhere

4. **Magic numbers**
   - Gamma parameters hardcoded (0.3, 0.3)
   - Learning rates in examples are ad-hoc
   - Batch sizes not validated

5. **Shared mutable state**
   ```python
   class NumpyroModel(ABC):
       Metrics = Metrics(loss = list())  # WRONG! Shared across all instances!
   ```
   - All instances share the same `Metrics.loss` list
   - This causes state leakage between models

#### ⚠️ Medium Issues

- Unused imports in files (e.g., `run_topicmodels.py` has many commented sections)
- Long files (e.g., `TBIP.py` is 272 lines) could be split
- No comprehensive error messages
- JAX/NumPyro expertise required to understand code flow

---

### 6. Reproducibility

#### ❌ Issues

1. **Environment specification incomplete**
   - `requirements.txt` and `pyproject.toml` conflict
   - JAX Metal GPU support is Mac-specific but not documented for Linux/Windows
   - No `environment.yml` for conda users
   - No pinned JAX build instructions

2. **Random seed control missing**
   - Seeds hardcoded in training loop
   - No way to reproduce results
   - `train_step()` doesn't accept `random_seed` parameter

3. **Data handling undocumented**
   - Example dataset `10k_amazon.csv` is included but no metadata
   - No guidance on data format requirements
   - No validation of input data

4. **Dependency lock file incomplete**
   - `poetry.lock` exists but `requirements.txt` conflicts with it
   - No pip lock file (`pip freeze` equivalent)

---

### 7. JOSS/JMLR Requirements

#### ❌ Critical Gaps

| Requirement | Status | Notes |
|---|---|---|
| Statement of Need | Missing | README needs comparison to alternatives |
| License | Missing | No LICENSE file |
| Repository URL | Missing | Not in pyproject.toml |
| CITATION.cff | Missing | Required for automatic citation |
| Test Suite | Missing | 0% coverage |
| CI/CD | Missing | No GitHub Actions |
| Documentation | Incomplete | Sphinx setup exists but needs content |
| Code of Conduct | Missing | Required for JOSS |
| Contributing Guide | Incomplete | Needs detailed guidelines |
| Authors/Attribution | Present | In pyproject.toml and docs/conf.py |

#### ⚠️ Medium

- No references to academic papers in docstrings
- No algorithm complexity analysis documented
- No benchmarks or performance comparison

---

## Part 2: Prioritized Recommendations

### 🔴 PRIORITY 1: Critical Blockers (Must Fix)

These prevent publication completely:

#### 1.1 Create LICENSE file
- **Effort:** 15 minutes
- **Action:** Choose a license (recommend MIT, Apache 2.0, or GPL-3.0 depending on goals)
- **File:** `/LICENSE`

#### 1.2 Fix dependency conflicts
- **Effort:** 30 minutes
- **Action:** Reconcile `pyproject.toml` and `requirements.txt`
  - Option A: Update `pyproject.toml` to match actual tested versions
  - Option B: Update `requirements.txt` to match `pyproject.toml` and test thoroughly
- **Recommendation:** Resolve to versions that work (test first!)

#### 1.3 Create CITATION.cff
- **Effort:** 20 minutes
- **Action:** Create `CITATION.cff` with author/affiliation metadata

#### 1.4 Enable root-level imports
- **Effort:** 30 minutes
- **Actions:**
  - Uncomment and fix `/root/__init__.py`
  - Rename `packages/` folder to match package name
  - Restructure to: `topicmodels/` containing `models/`, `utils/`, etc.

#### 1.5 Create comprehensive test suite
- **Effort:** 8-12 hours
- **Target:** 70%+ coverage
- **Include:**
  - Input validation tests
  - Model initialization tests
  - Training step tests
  - Output format tests
  - Edge cases (empty data, single document, etc.)

#### 1.6 Set up CI/CD
- **Effort:** 2 hours
- **Action:** Create `.github/workflows/tests.yml` for GitHub Actions
- **Include:** Lint, type checking, tests

---

### 🟠 PRIORITY 2: High-Impact Improvements (Should Fix)

#### 2.1 Refactor factory function
- **Effort:** 4-6 hours
- **Action:** Replace string-based dispatch with direct class imports
- **Benefit:** Better IDE support, more Pythonic, clearer error messages

#### 2.2 Add complete API reference documentation
- **Effort:** 4-6 hours
- **Action:**
  - Ensure all docstrings are complete (NumPy format)
  - Add parameter validation errors to docstrings
  - Document all return types
  - Generate Sphinx autodoc properly

#### 2.3 Add type hints throughout
- **Effort:** 6-8 hours
- **Action:** Add PEP 484 type annotations to all methods
- **Example:**
  ```python
  def train_step(self, num_steps: int, lr: float) -> Dict[str, jnp.ndarray]:
  ```

#### 2.4 Write comprehensive README update
- **Effort:** 2-3 hours
- **Include:**
  - Statement of Need: Why use this over Gensim/Scikit-learn?
  - Quick-start tutorial
  - Model selection guide
  - Performance considerations
  - Comparison table to existing tools

#### 2.5 Create User Guide with examples
- **Effort:** 4-5 hours
- **Include:**
  - Per-model usage examples
  - Hyperparameter tuning guide
  - Output interpretation guide
  - Visualization examples

#### 2.6 Add reproducibility features
- **Effort:** 2-3 hours
- **Actions:**
  - Add `random_seed` parameter to `train_step()`
  - Document seed reproducibility guarantees
  - Add environment setup script

---

### 🟡 PRIORITY 3: Code Quality (Should Fix)

#### 3.1 Fix shared mutable state in Metrics
- **Effort:** 1 hour
- **Action:** Move `Metrics` to instance attributes, not class attributes

#### 3.2 Input validation
- **Effort:** 2-3 hours
- **Actions:**
  - Validate `counts` is sparse matrix
  - Check `vocab` length matches matrix dimensions
  - Validate all numeric parameters
  - Provide clear error messages

#### 3.3 Remove hardcoded hyperparameters
- **Effort:** 2 hours
- **Actions:**
  - Make Gamma priors configurable
  - Document default choices
  - Allow users to override

#### 3.4 Code organization
- **Effort:** 3-4 hours
- **Actions:**
  - Split large files (e.g., TBIP.py)
  - Extract common patterns to base class
  - Organize utilities by function

---

### 🟢 PRIORITY 4: Nice-to-Have (Can Defer)

- [ ] Performance benchmarks
- [ ] Comparison plots vs Gensim/Scikit-learn
- [ ] Video tutorials
- [ ] Example notebooks for each model
- [ ] Contributes guide with detailed workflows
- [ ] Code of Conduct
- [ ] Author/contributor website links

---

## Part 3: Actionable Recommendations

### Immediate Actions (Week 1)

1. **Create LICENSE** → Pick appropriate license, add file
2. **Reconcile dependencies** → Test and fix conflicts
3. **Create CITATION.cff** → Add proper citation metadata
4. **Rename packages folder** → Move to `topicmodels/` structure
5. **Fix root imports** → Enable top-level imports

### Short-term (Week 2-3)

6. **Write 20-30 basic unit tests** → Cover happy paths
7. **Add type hints** → Use `pyright` or `mypy` in CI
8. **Update README** → Add Statement of Need + comparison table
9. **Document all methods** → Complete NumPy-style docstrings

### Medium-term (Week 4-6)

10. **Expand test suite** → 70%+ coverage, edge cases
11. **Create tutorials** → Jupyter notebooks for each model
12. **Set up CI/CD** → GitHub Actions with linting + tests
13. **Add API reference** → Auto-generate from docstrings
14. **Refactor factory** → Replace string dispatch with class imports

### Final Polish (Week 7-8)

15. **Reproducibility improvements** → Random seed control, environment files
16. **Performance documentation** → Benchmarks, scaling info
17. **Contribution guide** → Developer setup, PR workflow
18. **Final review** → Against JOSS/JMLR checklists

---

## Part 4: Example Code Templates

### Template 1: Proper `__init__.py` Structure

```python
# topicmodels/__init__.py
"""
topicmodels: Probabilistic topic modeling with Bayesian inference.

A Python package for advanced topic modeling using JAX and NumPyro,
providing implementations of PF, SPF, CPF, CSPF, TBIP, and ETM models.
"""

__version__ = "0.1.0"
__author__ = "Bernd Prostmaier"
__email__ = "b.prostmaier@icloud.com"

from .models import (
    PF,
    SPF,
    CPF,
    CSPF,
    TBIP,
    ETM,
    NumpyroModel,
    Metrics,
)

__all__ = [
    "PF",
    "SPF",
    "CPF",
    "CSPF",
    "TBIP",
    "ETM",
    "NumpyroModel",
    "Metrics",
]
```

### Template 2: CITATION.cff

```yaml
# CITATION.cff
cff-version: 1.2.0
title: "topicmodels: Probabilistic Topic Modeling with JAX"
authors:
  - family-names: Prostmaier
    given-names: Bernd
    email: b.prostmaier@icloud.com
  - family-names: Grün
    given-names: Bettina
  - family-names: Hofmarcher
    given-names: Paul
keywords:
  - topic-modeling
  - bayesian-inference
  - jax
  - probabilistic-modeling
version: 0.1.0
date-released: 2025-11-19
repository-code: "https://github.com/BPro2410/topicmodels_package"
license: MIT
abstract: >
  A Python package for probabilistic topic modeling using
  Bayesian inference built on JAX and NumPyro. Provides
  implementations of advanced topic models including PF, SPF,
  CPF, CSPF, TBIP, and ETM.
```

### Template 3: Updated pyproject.toml

```toml
[project]
name = "topicmodels"
version = "0.1.0"
description = "Probabilistic topic modeling with Bayesian inference using JAX and NumPyro"
authors = [
    {name = "Bernd Prostmaier", email = "b.prostmaier@icloud.com"},
    {name = "Bettina Grün"},
    {name = "Paul Hofmarcher"}
]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11,<3.14"
keywords = ["topic-modeling", "bayesian-inference", "jax", "probabilistic"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Information Analysis",
]

dependencies = [
    "jax>=0.4.0",
    "jaxlib>=0.4.0",
    "numpyro>=0.15.0",
    "optax>=0.2.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "matplotlib>=3.7",
    "tqdm>=4.65",
    "gensim>=4.3",  # for embeddings
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "isort>=5.12",
    "mypy>=1.0",
    "pylint>=2.17",
    "ruff>=0.1",
]
docs = [
    "sphinx>=6.0",
    "sphinx-rtd-theme>=1.2",
    "myst-parser>=1.0",
    "sphinx-autodoc-typehints>=1.22",
]

[project.urls]
repository = "https://github.com/BPro2410/topicmodels_package"
documentation = "https://topicmodels.readthedocs.io"
issues = "https://github.com/BPro2410/topicmodels_package/issues"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

### Template 4: Basic Test File

```python
# tests/test_pf.py
"""Tests for the Poisson Factorization (PF) model."""

import pytest
import numpy as np
import scipy.sparse as sparse
from topicmodels import PF


class TestPFInitialization:
    """Test PF model initialization."""

    @pytest.fixture
    def sample_data(self):
        """Create sample sparse matrix and vocabulary."""
        counts = sparse.csr_matrix(
            np.array([
                [1, 2, 3, 0],
                [0, 2, 1, 1],
                [3, 0, 0, 2],
            ], dtype=np.float32)
        )
        vocab = np.array(['word1', 'word2', 'word3', 'word4'])
        return counts, vocab

    def test_initialization(self, sample_data):
        """Test model initializes with valid inputs."""
        counts, vocab = sample_data
        model = PF(counts, vocab, num_topics=5, batch_size=32)
        
        assert model.K == 5
        assert model.D == 3
        assert model.V == 4
        assert model.batch_size == 32

    def test_initialization_invalid_batch_size(self, sample_data):
        """Test that invalid batch size raises error."""
        counts, vocab = sample_data
        with pytest.raises((ValueError, AssertionError)):
            PF(counts, vocab, num_topics=5, batch_size=0)

    def test_initialization_invalid_num_topics(self, sample_data):
        """Test that invalid num_topics raises error."""
        counts, vocab = sample_data
        with pytest.raises((ValueError, AssertionError)):
            PF(counts, vocab, num_topics=0, batch_size=32)
```

### Template 5: Type-Hinted Model Base Class

```python
# topicmodels/models/numpyro_model.py
"""Abstract base class for probabilistic models."""

from abc import abstractmethod, ABC
from typing import Tuple, Dict, Any, Optional

import jax.numpy as jnp
from jax import random
import numpy as np
import scipy.sparse as sparse


class NumpyroModel(ABC):
    """Abstract base class for NumPyro-based probabilistic models."""

    def __init__(
        self,
        counts: sparse.csr_matrix,
        vocab: np.ndarray,
        batch_size: int,
    ) -> None:
        """
        Initialize the model.

        Parameters
        ----------
        counts : scipy.sparse.csr_matrix
            Document-term matrix of shape (D, V) with word counts.
        vocab : np.ndarray
            Vocabulary array of shape (V,).
        batch_size : int
            Number of documents per batch. Must be > 0.

        Raises
        ------
        ValueError
            If batch_size <= 0 or counts not a sparse matrix.
        """
        if not sparse.issparse(counts):
            raise ValueError("counts must be a scipy sparse matrix")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.counts = counts
        self.D, self.V = counts.shape
        self.vocab = vocab
        self.batch_size = batch_size

    @abstractmethod
    def _model(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray) -> None:
        """Define the probabilistic model."""
        pass

    @abstractmethod
    def _guide(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray) -> None:
        """Define the variational guide."""
        pass

    def train_step(
        self,
        num_steps: int,
        lr: float,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform training steps using stochastic variational inference.

        Parameters
        ----------
        num_steps : int
            Number of training iterations. Must be > 0.
        lr : float
            Learning rate. Must be > 0.
        random_seed : int, optional
            Random seed for reproducibility. If None, uses random state.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing estimated parameters and training history.
        """
        pass
```

---

## Part 5: JOSS/JMLR Submission Checklist

Before submitting, verify:

### Required Files
- [ ] `LICENSE` file present and valid
- [ ] `CITATION.cff` with all author info
- [ ] `.github/workflows/tests.yml` (CI/CD passing)
- [ ] `CONTRIBUTING.md` with contribution guidelines
- [ ] `CODE_OF_CONDUCT.md` (JOSS requirement)
- [ ] `CHANGELOG.md` documenting releases

### Documentation
- [ ] README with "Statement of Need" section
- [ ] Comparison to existing tools (Gensim, Scikit-learn, etc.)
- [ ] API reference auto-generated from docstrings
- [ ] At least 3 complete tutorials/examples
- [ ] Installation instructions that work
- [ ] Contribution guide

### Testing & Quality
- [ ] Test coverage ≥70%
- [ ] All tests passing
- [ ] CI/CD workflow active
- [ ] No linting errors (flake8, pylint)
- [ ] Type hints on all public APIs
- [ ] Reproducibility: fixed random seeds in examples

### Packaging
- [ ] Package installable via `pip install topicmodels`
- [ ] Dependencies properly pinned (no conflicts)
- [ ] Metadata complete in `pyproject.toml`
- [ ] Version number follows semantic versioning
- [ ] No deprecated dependencies

### Academic
- [ ] Algorithm descriptions reference original papers
- [ ] Benchmarks/performance analysis documented
- [ ] Complexity analysis provided
- [ ] Reproducible figures/results

### Community
- [ ] Proper issue and PR templates in GitHub
- [ ] Recent commits (not abandoned)
- [ ] Clear maintenance responsibility

---

## Summary: Effort Estimates

| Category | Hours | Priority |
|---|---|---|
| Fix dependencies & licensing | 1-2 | CRITICAL |
| Refactor package structure | 3-4 | CRITICAL |
| Create test suite (70% coverage) | 10-12 | CRITICAL |
| Add type hints | 6-8 | HIGH |
| Complete documentation | 8-10 | HIGH |
| Set up CI/CD | 2-3 | HIGH |
| Code quality improvements | 4-5 | MEDIUM |
| Reproducibility features | 2-3 | MEDIUM |
| API refactoring | 4-6 | MEDIUM |
| **TOTAL** | **40-53** | — |

**With thorough testing & final review: 60-80 hours total.**

---

## Recommended Next Steps

1. **This week:** Implement PRIORITY 1 items (licensing, dependencies, structure)
2. **Next week:** Build test suite and CI/CD
3. **Week 3:** Complete documentation and type hints
4. **Week 4:** API refactoring and advanced features
5. **Submit:** To JOSS/JMLR editorial office

Good luck with the publication! 🚀
