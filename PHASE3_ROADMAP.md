# 🔜 PHASE 3 ROADMAP: Code Quality & Polish

**Estimated Duration:** 1-2 weeks at 6-8 hours/week  
**Remaining Time to Publication:** 2-3 weeks total  
**Current Publication Readiness:** 65-70%

---

## 📋 Phase 3 Tasks (12-15 hours)

### Task 1: Complete Type Hints (3-4 hours)

**Files to Update:**
- `packages/models/SPF.py` - Add type hints (following PF pattern)
- `packages/models/CPF.py` - Add type hints
- `packages/models/CSPF.py` - Add type hints
- `packages/models/TBIP.py` - Add type hints
- `packages/models/ETM.py` - Add type hints
- `packages/models/Metrics.py` - Add type hints
- `packages/utils/utils.py` - Add type hints

**Pattern to Follow (from PF.py):**
```python
from typing import Dict, Tuple, Any, Optional
import scipy.sparse as sparse

class SPF(NumpyroModel):
    def __init__(
        self,
        counts: sparse.csr_matrix,
        vocab: np.ndarray,
        keywords: Dict[int, list],
        residual_topics: int,
        batch_size: int,
    ) -> None:
        """Docstring with Parameters, Raises, Examples."""
        super().__init__()
        # Validation code
        pass
    
    def _model(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray) -> None:
        """Model definition with type hints."""
        pass
```

**Checklist:**
- [ ] SPF.py: Type hints + input validation + docstring
- [ ] CPF.py: Type hints + input validation + docstring
- [ ] CSPF.py: Type hints + input validation + docstring
- [ ] TBIP.py: Type hints + docstring
- [ ] ETM.py: Type hints + docstring
- [ ] Metrics.py: Type hints for all methods
- [ ] utils.py: Type hints for utility functions

---

### Task 2: Run Code Quality Tools (2-3 hours)

**Step 1: Format with Black**
```bash
cd /Users/bernd/Documents/01_Coding/02_GitHub/topicmodels_package
black packages/ tests/ --line-length 100
```

**Step 2: Sort Imports with isort**
```bash
isort packages/ tests/ --profile black
```

**Step 3: Lint with flake8**
```bash
flake8 packages/ tests/ --max-line-length 100 --extend-ignore E203,W503
```

**Step 4: Type Check with mypy**
```bash
mypy packages/ --ignore-missing-imports --no-error-summary 2>&1 | head -50
```

**Checklist:**
- [ ] Black formatting applied
- [ ] isort import ordering applied
- [ ] flake8 passes (0 errors)
- [ ] mypy passes (0 or minimal errors)

---

### Task 3: Expand Test Coverage (4-5 hours)

**Create: `tests/test_spf.py` (Expanded)**
- Expand from current ~8 tests to ~20 tests
- Add keyword validation tests
- Add residual topic tests
- Add integration tests with keywords

**Create: `tests/test_cpf.py`**
- ~15 tests for CPF model
- Covariate handling tests
- Covariate shape tests
- Integration tests

**Create: `tests/test_reproducibility.py`**
- Seed reproducibility across all models
- Different seed variation tests
- Loss trajectory consistency

**Update: `tests/conftest.py`**
- Add more fixture variants
- Add keywords_dict fixture
- Add covariates fixture

**Target Coverage:**
- PF: 20+ tests
- SPF: 20+ tests
- CPF: 15+ tests
- Reproducibility: 10+ tests
- Total: 150+ tests (30% increase)

**Checklist:**
- [ ] SPF tests expanded
- [ ] CPF tests created
- [ ] Reproducibility tests added
- [ ] Fixtures expanded
- [ ] Coverage > 75%

---

### Task 4: Pre-commit Hooks (1-2 hours)

**Create: `.pre-commit-config.yaml`**
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=100"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.4.1
    hooks:
      - id: mypy
        args: ["--ignore-missing-imports"]
```

**Installation:**
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

**Checklist:**
- [ ] `.pre-commit-config.yaml` created
- [ ] Pre-commit installed
- [ ] All hooks passing

---

### Task 5: Documentation Polish (2-3 hours)

**Update: `docs/conf.py`**
- Ensure autodoc is configured
- Add all models to documentation
- Set up API reference properly

**Create: `docs/intro/api_reference.rst`**
- Auto-generate from docstrings
- List all public classes
- List all public methods

**Update: `docs/index.rst`**
- Add link to API reference
- Add examples section
- Add contributing section

**Verify Build:**
```bash
cd docs
make clean
make html
# Check _build/html/index.html
```

**Checklist:**
- [ ] Autodoc configured
- [ ] API reference complete
- [ ] Documentation builds without warnings
- [ ] All models documented
- [ ] All methods documented

---

## ⏱️ Time Estimate Breakdown

| Task | Hours | Priority |
|------|-------|----------|
| Type hints (remaining models) | 3-4 | HIGH |
| Code quality tools | 2-3 | HIGH |
| Test expansion | 4-5 | HIGH |
| Pre-commit setup | 1-2 | MEDIUM |
| Documentation polish | 2-3 | MEDIUM |
| **TOTAL** | **12-17** | |

---

## ✅ Phase 3 Success Criteria

- [ ] All models have complete type hints
- [ ] Black formatting applied to all code
- [ ] isort applied to all imports
- [ ] flake8 passes with 0 errors
- [ ] mypy passes (non-blocking)
- [ ] Test count > 150
- [ ] Test coverage > 75%
- [ ] Pre-commit hooks installed
- [ ] Documentation builds cleanly
- [ ] No warnings in docs build

**Publication Readiness After Phase 3:** 85-90%

---

## 📝 Phase 3 Completion Template

```markdown
# ✅ PHASE 3 IMPLEMENTATION COMPLETE

**Completion Date:** [DATE]  
**Time Invested:** [X] hours

## Deliverables

### Type Hints
- [x] SPF.py updated
- [x] CPF.py updated
- [x] CSPF.py updated
- [x] TBIP.py updated
- [x] ETM.py updated
- [x] Metrics.py updated
- [x] utils.py updated

### Code Quality
- [x] Black formatting: 100%
- [x] isort import ordering: 100%
- [x] flake8 linting: 0 errors
- [x] mypy type checking: [X] errors

### Testing
- [x] Test count: 150+
- [x] Coverage: 75%+
- [x] SPF tests: 20+
- [x] CPF tests: 15+
- [x] Reproducibility tests: 10+

### Documentation
- [x] API reference generated
- [x] Sphinx docs build cleanly
- [x] 0 warnings in build
- [x] All models documented
- [x] Pre-commit hooks installed

## Metrics

| Metric | Value |
|--------|-------|
| Type Hints | 95%+ |
| Code Quality | 100% |
| Test Count | 150+ |
| Coverage | 75%+ |
| Publication Readiness | 85-90% |

**Status: ✅ PHASE 3 COMPLETE**
```

---

## 🎯 Next: Phase 4 (Final Polish)

After Phase 3 completes, Phase 4 includes:
- [ ] Create example notebooks (3-5)
- [ ] Add performance benchmarks
- [ ] Enhance CONTRIBUTING.md
- [ ] Add CODE_OF_CONDUCT.md
- [ ] Final review
- [ ] Create v0.1.0 release
- [ ] Submit to JOSS/JMLR

**Phase 4 Estimated Time:** 12-15 hours (1 week)

---

## 📊 Publication Timeline

| Phase | Status | Hours | Publication % |
|-------|--------|-------|---|
| Phase 1 | ✅ Complete | 8-10 | 30% |
| Phase 2 | ✅ Complete | 10-12 | 35% |
| Phase 3 | 🔜 Ready | 12-15 | +20% |
| Phase 4 | 📋 Planned | 12-15 | +15% |
| **Total** | **In Progress** | **42-52** | **100%** |

**Current: 65-70% ready**  
**After Phase 3: 85-90% ready**  
**After Phase 4: 95%+ ready for submission**

---

**Phase 3 Status: Ready to Begin**  
**Recommended Start Date:** Immediately after Phase 2  
**Estimated Completion:** 1-2 weeks
