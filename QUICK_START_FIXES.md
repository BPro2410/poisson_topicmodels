# Quick Start: Critical Fixes for Publication

This file lists the **immediate, actionable steps** to get the package publication-ready.

## Week 1: Critical Blockers (Fix These First)

### ✅ Task 1: Create LICENSE (15 min)

```bash
# Create MIT License (recommended for academic software)
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 Bernd Prostmaier, Bettina Grün, Paul Hofmarcher

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
```

**Verify:** `ls LICENSE` should exist

---

### ✅ Task 2: Create CITATION.cff (20 min)

Create file: `CITATION.cff`

```yaml
cff-version: 1.2.0
title: "topicmodels: Probabilistic Topic Modeling with JAX"
authors:
  - family-names: Prostmaier
    given-names: Bernd
    email: b.prostmaier@icloud.com
    affiliation: "University/Institution Name"  # Add your affiliation
  - family-names: Grün
    given-names: Bettina
    affiliation: "University/Institution Name"
  - family-names: Hofmarcher
    given-names: Paul
    affiliation: "University/Institution Name"
keywords:
  - "topic-modeling"
  - "bayesian-inference"
  - "jax"
  - "numpyro"
  - "probabilistic-modeling"
version: 0.1.0
date-released: 2025-11-19
repository-code: "https://github.com/BPro2410/topicmodels_package"
license: MIT
abstract: >
  A Python package for probabilistic topic modeling using Bayesian
  inference built on JAX and NumPyro. Provides implementations of
  advanced topic models including Poisson Factorization (PF), Seeded
  Poisson Factorization (SPF), Covariate Poisson Factorization (CPF),
  Covariate Seeded Poisson Factorization (CSPF), Text-Based Ideal Points
  (TBIP), and Embedded Topic Models (ETM).
contact:
  - name: "Bernd Prostmaier"
    email: "b.prostmaier@icloud.com"
```

**Verify:** `git add CITATION.cff && git status` should show it

---

### ✅ Task 3: Fix Dependency Conflicts (30 min)

**Problem:** `requirements.txt` and `pyproject.toml` have conflicting versions.

**Option A: Update pyproject.toml to use tested versions** (RECOMMENDED)

Edit `pyproject.toml`:

```toml
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

**Then test:**
```bash
python -m venv test_env
source test_env/bin/activate
pip install -e .
python -c "import topicmodels; print('Success')"
```

**Verify:** Installation succeeds without errors

---

### ✅ Task 4: Rename Package Folder (1 hour)

**Problem:** Package structure is `packages/models/` instead of `topicmodels/`

**Steps:**
```bash
# 1. Create new structure
mkdir -p topicmodels/models topicmodels/utils

# 2. Move files
mv packages/models/*.py topicmodels/models/
mv packages/utils/* topicmodels/utils/

# 3. Update imports in moved files
# Replace: from packages.models import X -> from topicmodels.models import X
# Replace: from packages.utils import X -> from topicmodels.utils import X

# 4. Create __init__.py files
touch topicmodels/__init__.py
touch topicmodels/models/__init__.py  
touch topicmodels/utils/__init__.py

# 5. Remove old structure
rm -rf packages/

# 6. Test imports
python -c "from topicmodels import PF; print('Success')"
```

**Verify:** `from topicmodels import PF` works

---

### ✅ Task 5: Update Root __init__.py (30 min)

Create/update: `topicmodels/__init__.py`

```python
"""
topicmodels: Probabilistic topic modeling with Bayesian inference.

A Python package for advanced topic modeling using JAX and NumPyro,
providing implementations of several state-of-the-art topic models:
- Poisson Factorization (PF)
- Seeded Poisson Factorization (SPF)
- Covariate Poisson Factorization (CPF)
- Covariate Seeded Poisson Factorization (CSPF)
- Text-Based Ideal Points (TBIP)
- Embedded Topic Models (ETM)
"""

__version__ = "0.1.0"
__author__ = "Bernd Prostmaier, Bettina Grün, Paul Hofmarcher"
__email__ = "b.prostmaier@icloud.com"

from .models.PF import PF
from .models.SPF import SPF
from .models.CPF import CPF
from .models.CSPF import CSPF
from .models.TBIP import TBIP
from .models.ETM import ETM
from .models.numpyro_model import NumpyroModel
from .models.Metrics import Metrics

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

**Verify:** 
```bash
python -c "from topicmodels import PF, SPF, CPF; print('All imports work')"
```

---

### ✅ Task 6: Create Tests Directory (2 hours)

Create: `tests/test_models.py`

```python
"""Basic tests for topic models."""

import pytest
import numpy as np
import scipy.sparse as sparse
from topicmodels import PF, SPF, Metrics


@pytest.fixture
def sample_data():
    """Create sample sparse matrix and vocabulary."""
    np.random.seed(42)
    counts = sparse.random(10, 20, density=0.5, format='csr', dtype=np.float32)
    vocab = np.array([f'word_{i}' for i in range(20)])
    return counts, vocab


class TestPFModel:
    """Test Poisson Factorization model."""
    
    def test_pf_initialization(self, sample_data):
        """Test PF model initializes correctly."""
        counts, vocab = sample_data
        model = PF(counts, vocab, num_topics=5, batch_size=4)
        
        assert model.K == 5
        assert model.D == 10
        assert model.V == 20
        assert model.batch_size == 4
    
    def test_pf_invalid_params(self, sample_data):
        """Test PF rejects invalid parameters."""
        counts, vocab = sample_data
        
        with pytest.raises((ValueError, AssertionError)):
            PF(counts, vocab, num_topics=0, batch_size=4)
        
        with pytest.raises((ValueError, AssertionError)):
            PF(counts, vocab, num_topics=5, batch_size=0)


class TestSPFModel:
    """Test Seeded Poisson Factorization model."""
    
    def test_spf_initialization(self, sample_data):
        """Test SPF model initializes correctly."""
        counts, vocab = sample_data
        keywords = {
            'topic1': ['word_0', 'word_1', 'word_2'],
            'topic2': ['word_10', 'word_11', 'word_12'],
        }
        
        model = SPF(counts, vocab, keywords, residual_topics=1, batch_size=4)
        assert model.K == 3  # 2 seed topics + 1 residual
        assert model.D == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

Create: `tests/__init__.py` (empty file)

**Verify:** 
```bash
cd tests
pytest test_models.py -v
```

---

## Week 2: High-Impact Improvements

### 📝 Task 7: Update README (1-2 hours)

Replace README.md with comprehensive version. See `PUBLICATION_AUDIT.md` for template.

**Key additions:**
- Statement of Need (why use this over alternatives?)
- Comparison table to Gensim, Scikit-learn
- Quick installation (fixed dependencies)
- Minimal working example
- Link to full documentation

**Verify:** README is under 5 KB, GitHub renders correctly

---

### 🏗️ Task 8: Set up CI/CD (1 hour)

Create: `.github/workflows/tests.yml`

```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12', '3.13']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Lint with flake8
      run: |
        flake8 topicmodels tests --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Type check with mypy
      run: |
        mypy topicmodels --ignore-missing-imports
    
    - name: Run tests with pytest
      run: |
        pytest tests/ -v --cov=topicmodels --cov-report=term-only
```

**Verify:** GitHub Actions tab shows workflow running

---

### 📚 Task 9: Add Type Hints to Base Class (1 hour)

Update: `topicmodels/models/numpyro_model.py`

Add type hints to all methods:

```python
from typing import Tuple, Dict, Any, Optional
import scipy.sparse as sparse

class NumpyroModel(ABC):
    """Abstract base class for probabilistic models."""
    
    def __init__(
        self,
        counts: sparse.csr_matrix,
        vocab: np.ndarray,
        batch_size: int,
    ) -> None:
        """Initialize model."""
        # ... existing code ...
    
    def train_step(
        self,
        num_steps: int,
        lr: float,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train model using SVI."""
        # ... existing code ...
```

---

## Checklist

- [ ] LICENSE file created
- [ ] CITATION.cff created
- [ ] Dependencies reconciled (pyproject.toml and requirements.txt match)
- [ ] Package renamed to `topicmodels/`
- [ ] Root `__init__.py` updated with proper exports
- [ ] Basic test suite created (tests/test_models.py)
- [ ] CI/CD workflow created (.github/workflows/tests.yml)
- [ ] README updated with Statement of Need
- [ ] Type hints added to base model class
- [ ] All tests passing
- [ ] GitHub Actions workflow running successfully

---

## Verification Commands

After each task, run these to verify:

```bash
# Verify package imports
python -c "from topicmodels import PF, SPF, CPF; print('✓ Imports work')"

# Verify tests
pytest tests/ -v

# Verify pyproject.toml is valid
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb')); print('✓ pyproject.toml valid')"

# Verify installation
pip install -e .

# Check for issues
python -m flake8 topicmodels --count

# List changes
git status
```

---

## Next Steps

After completing these tasks:
1. Push changes to GitHub
2. Verify CI/CD passes
3. Review documentation against JOSS/JMLR requirements
4. Create issues for remaining work (see PUBLICATION_AUDIT.md PRIORITY 2-4)
5. Engage with potential users for feedback

**Estimated time to complete:** 6-8 hours
