.. _testing:

================================================================================
Testing Guide
================================================================================

How to test poisson-topicmodels code and write tests for your own code.

Running Tests
=============

Run all tests:

.. code-block:: bash

   pytest tests/

Run specific test file:

.. code-block:: bash

   pytest tests/test_pf.py

Run with coverage:

.. code-block:: bash

   pytest tests/ --cov=poisson_topicmodels

Test Categories
===============

**Unit Tests**: Function/class behavior

- Located in: `tests/test_*.py`
- Fast, focused
- Test individual components

**Integration Tests**: Multiple components together

- Located in: `tests/test_integration.py`
- Slower, broader scope
- Test workflows

**Comprehensive Tests**: All models and features

- Located in: `tests/test_models_comprehensive.py`
- Slowest
- Test all variations

Test Coverage
=============

Current coverage: >70%

View coverage report:

.. code-block:: bash

   pytest tests/ --cov=poisson_topicmodels --cov-report=html
   open htmlcov/index.html  # View in browser

Writing Tests
=============

Test structure:

.. code-block:: python

   import pytest
   import numpy as np
   from scipy.sparse import csr_matrix
   from poisson_topicmodels import PF

   class TestPF:
       """Tests for Poisson Factorization model."""

       @pytest.fixture
       def sample_data(self):
           """Create sample data for testing."""
           counts = csr_matrix(
               np.random.poisson(2, (50, 100)).astype(np.float32)
           )
           vocab = np.array([f'word_{i}' for i in range(100)])
           return counts, vocab

       def test_initialization(self, sample_data):
           """Test model initializes correctly."""
           counts, vocab = sample_data
           model = PF(counts, vocab, num_topics=10)
           assert model.num_topics == 10
           assert model.vocab.shape[0] == 100

       def test_training(self, sample_data):
           """Test model trains successfully."""
           counts, vocab = sample_data
           model = PF(counts, vocab, num_topics=5)
           params = model.train(num_iterations=10, learning_rate=0.01)
           assert 'loss' in params or params is not None

       def test_get_topics(self, sample_data):
           """Test topic extraction."""
           counts, vocab = sample_data
           model = PF(counts, vocab, num_topics=5)
           model.train(num_iterations=10, learning_rate=0.01)
           topics = model.get_topics()
           assert topics.shape == (100, 5)

Continuous Integration
======================

Tests run automatically on:

- Every push to main/develop
- Every pull request
- Scheduled nightly runs

Via GitHub Actions:

- Python 3.11, 3.12, 3.13
- CPU and GPU (if available)
- Multiple OS: Linux, macOS, Windows

View results on GitHub Actions tab.

Testing Best Practices
======================

**For users**:

- Run tests after installation: ``pytest tests/test_imports.py``
- Before reporting bugs: run full test suite
- Document any test failures

**For developers**:

- Write tests for new features
- Aim for >80% code coverage
- Test edge cases and error conditions
- Document test purpose

Test Organization
=================

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - File
     - Purpose
   * - test_imports.py
     - Check packages import correctly
   * - test_input_validation.py
     - Validate input handling
   * - test_pf.py
     - Poisson Factorization tests
   * - test_spf.py
     - Seeded models tests
   * - test_integration.py
     - End-to-end workflows
   * - test_models_comprehensive.py
     - All models, all variants
   * - test_training_and_outputs.py
     - Training process & outputs

Common Test Patterns
====================

**Checking shape**:

.. code-block:: python

   topics = model.get_topics()
   assert topics.shape == (vocab_size, num_topics)

**Checking values**:

.. code-block:: python

   topics = model.get_topics()
   assert np.all(topics >= 0)  # Non-negative
   assert np.allclose(topics.sum(axis=0), 1.0)  # Normalized

**Checking errors**:

.. code-block:: python

   with pytest.raises(ValueError):
       model = PF(invalid_counts, vocab, num_topics=5)

**Parametrized tests**:

.. code-block:: python

   @pytest.mark.parametrize("num_topics", [5, 10, 20])
   def test_different_topics(self, sample_data, num_topics):
       """Test with different numbers of topics."""
       counts, vocab = sample_data
       model = PF(counts, vocab, num_topics=num_topics)
       model.train(num_iterations=5)
       assert model.get_topics().shape[1] == num_topics

Debugging Tests
===============

Run with verbose output:

.. code-block:: bash

   pytest tests/ -v  # Verbose
   pytest tests/ -vv # Very verbose
   pytest tests/ -s  # Show print statements

Stop at first failure:

.. code-block:: bash

   pytest tests/ -x

Debug with pdb:

.. code-block:: bash

   pytest tests/ --pdb  # Drop to debugger on failure

Performance Testing
===================

Time specific tests:

.. code-block:: bash

   pytest tests/ --durations=10  # 10 slowest tests

Mark slow tests:

.. code-block:: python

   @pytest.mark.slow
   def test_large_model(self):
       """This test is slow."""
       ...

Run only fast tests:

.. code-block:: bash

   pytest tests/ -m "not slow"

Testing GPU Code
================

Tests detect GPU automatically:

.. code-block:: python

   import jax

   def test_with_gpu_if_available():
       if not jax.devices():
           pytest.skip("GPU not available")

       # Test GPU-specific code
       ...

Or mark GPU tests:

.. code-block:: python

   @pytest.mark.gpu
   def test_gpu_training(self):
       """Only run on GPU."""
       ...

Run GPU tests:

.. code-block:: bash

   pytest tests/ -m gpu

Troubleshooting Tests
=====================

**Tests fail on import**:

Install development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

**Random test failures**:

Set seed for reproducibility:

.. code-block:: python

   np.random.seed(42)

**Memory issues during testing**:

Run with smaller datasets or skip large tests:

.. code-block:: bash

   pytest tests/ -m "not large"

**Tests hang**:

Add timeout:

.. code-block:: bash

   pytest tests/ --timeout=60  # 60 second timeout per test

Contributing Tests
==================

When submitting code:

1. Write tests for new functionality
2. Ensure all tests pass
3. Maintain or increase coverage (aim for >70%)
4. Document test purpose

See :doc:`../contributing_guide/index` for full guidelines.

Test Configuration
==================

Tests configured in:

- ``pytest.ini`` – Pytest settings
- ``conftest.py`` – Shared fixtures
- ``.github/workflows/`` – CI configuration

Examples from conftest:

.. code-block:: python

   @pytest.fixture
   def random_seed():
       """Ensure reproducibility."""
       np.random.seed(42)
       return 42

   @pytest.fixture
   def sample_dtm():
       """Reusable sample data."""
       counts = csr_matrix(
           np.random.poisson(2, (100, 500)).astype(np.float32)
       )
       vocab = np.array([f'word_{i}' for i in range(500)])
       return counts, vocab

Testing Progress
================

Latest test results:

- Total tests: 150+
- Coverage: >70%
- CI status: Passing on main

Contribution Checklist
======================

When adding code:

✓ Write unit tests
✓ Write integration tests
✓ All tests pass locally
✓ Coverage doesn't decrease
✓ Document tested behavior
✓ Test on Python 3.11, 3.12, 3.13

Next Steps
==========

- **Run tests**: ``pytest tests/``
- **Write tests**: Follow patterns above
- **CI/CD**: Automated via GitHub Actions
- **Contribute tests**: See :doc:`../contributing_guide/index`
