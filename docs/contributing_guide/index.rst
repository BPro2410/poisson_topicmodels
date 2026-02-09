.. _contributing:

================================================================================
Contributing Guide
================================================================================

Thank you for your interest in contributing to poisson-topicmodels!

This guide explains how to contribute code, documentation, examples, and bug reports.

Ways to Contribute
==================

**Code**

- New features
- Bug fixes
- Performance improvements
- Refactoring

**Documentation**

- Fix typos or clarify docs
- Add tutorials or guides
- Improve existing documentation
- Translate documentation

**Examples**

- Create practical examples
- Write Jupyter notebooks
- Document use cases
- Build plugins

**Testing**

- Improve test coverage
- Add edge case tests
- Performance testing

**Discussion**

- Answer questions
- Provide feedback
- Share ideas

Development Setup
=================

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/topicmodels_package.git
   cd topicmodels_package

3. Create a development environment:

.. code-block:: bash

   python -m venv dev_env
   source dev_env/bin/activate
   pip install -e ".[dev,docs]"

4. Verify setup:

.. code-block:: bash

   pytest tests/test_imports.py

Code Style
==========

We use `Black <https://github.com/psf/black>`_ for code formatting.

Format your code:

.. code-block:: bash

   black poisson_topicmodels/

Check format without changing:

.. code-block:: bash

   black --check poisson_topicmodels/

Other tools:

- **isort**: Organize imports
- **mypy**: Type checking
- **pylint**: Code linting
- **flake8**: Style guide enforcement

Run all checks:

.. code-block:: bash

   black poisson_topicmodels/
   isort poisson_topicmodels/
   mypy poisson_topicmodels/
   pylint poisson_topicmodels/
   flake8 poisson_topicmodels/

Pre-commit Hooks
================

Automatically check code before committing:

.. code-block:: bash

   pre-commit install
   # Now checks run automatically on git commit

Manual checks:

.. code-block:: bash

   pre-commit run --all-files

Workflow
========

1. **Fork and branch**:

.. code-block:: bash

   git checkout -b feature/my-feature

2. **Make changes**:

   - Write code
   - Write tests
   - Update docs
   - Ensure tests pass

.. code-block:: bash

   pytest tests/

3. **Commit with clear message**:

.. code-block:: bash

   git add .
   git commit -m "feature: Add topic filtering

   - Implement filter_rare_words() function
   - Add tests for filtering
   - Update API documentation"

Follow commit style:

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation
- **test**: Tests
- **refactor**: Code refactor
- **chore**: Maintenance

4. **Push and create PR**:

.. code-block:: bash

   git push origin feature/my-feature

   # Create PR on GitHub

5. **Respond to review**:

   - Address feedback
   - Update as requested
   - Add commits on top

6. **Merge**:

   Once approved, maintainers merge your PR.

Contributing Code
=================

**Before starting**:

- Check open issues and PRs (avoid duplicates)
- Discuss major features first (open issue)

**During development**:

- Follow code style (Black, isort)
- Write tests for new code
- Update docstrings (NumPy style)
- Update changelog

**Before submitting PR**:

- All tests pass locally
- Code coverage doesn't decrease
- Docstrings are complete
- Type hints are included

Example PR:

.. code-block:: text

   Title: Add topic filtering API

   Description:
   - Adds filter_rare_words() to filter infrequent topics
   - Useful for reducing noise in results
   - Includes comprehensive tests
   - Adds documentation with examples

   Type of Change:
   - [x] New feature
   - [ ] Bug fix
   - [ ] Documentation

   Checklist:
   - [x] Tests added
   - [x] Documentation updated
   - [x] Code follows style guidelines
   - [x] No breaking changes

Contributing Documentation
==========================

Documentation uses Sphinx and reStructuredText.

File locations:

- Main docs: ``docs/*.rst``
- Sections: ``docs/section/``
- Examples: ``examples/``

Fix typo:

1. Edit ``docs/file.rst``
2. Build locally: ``make html`` (in docs/)
3. Check changes
4. Submit PR

Add new page:

1. Create ``docs/section/page.rst``
2. Add to toctree in ``docs/section/index.rst``
3. Build and check
4. Submit PR

Contributing Examples
=====================

Create new file: ``examples/example_name.py``

Structure:

.. code-block:: python

   """
   Example: Clear title

   Demonstrates:
   - Feature 1
   - Feature 2
   - Feature 3
   """

   import numpy as np
   from poisson_topicmodels import [Model]

   # Setup
   # ... data preparation

   # Training
   # ... model training

   # Analysis
   # ... results analysis

Requirements:

- Clear docstring
- Runnable without modification
- Comments explaining steps
- Real or realistic data
- ~50-100 lines

Reporting Bugs
==============

**Before reporting**:

- Check existing issues
- Try latest version
- Include minimal reproducible example

**File a bug report**:

1. Go to `GitHub Issues <https://github.com/BPro2410/topicmodels_package/issues>`_
2. Click "New Issue"
3. Select "Bug Report" template
4. Fill in:
   - Title: concise description
   - Description: what happened
   - Expected: what should happen
   - Reproduction: steps to reproduce
   - Environment: Python version, OS, GPU, etc.
   - Code: minimal example

Example bug report:

.. code-block:: text

   Title: Training fails with ValueError on sparse matrix

   Description:
   When training a PF model with a sparse document-term matrix
   in COO format, I get a ValueError.

   Steps to Reproduce:
   1. Create COO matrix
   2. call model.train()
   3. Error occurs

   Error Message:
   [Full traceback]

   Code:
   [Minimal example that reproduces issue]

   Environment:
   - Python: 3.11
   - OS: macOS
   - JAX: 0.8.0

Requesting Features
===================

**Feature request**:

1. Go to `GitHub Discussions <https://github.com/BPro2410/topicmodels_package/discussions>`_
2. Create "Ideas" discussion
3. Or open issue with "Feature Request" template

**Include**:

- Clear description of requested feature
- Use case and motivation
- Example of how you'd use it
- Alternatives if any

Code Review Process
===================

Every PR gets reviewed for:

**Code Quality**:

- Follows style guide
- Tests included
- No obvious bugs
- Performance is acceptable

**Documentation**:

- Docstrings present
- Type hints included
- Comments explain complex logic

**Testing**:

- Tests cover new code
- Edge cases tested
- No regression in other tests

**Process**:

1. You submit PR
2. Automated checks run (CI)
3. Reviewers provide feedback
4. You address feedback
5. Maintainers approve and merge

Merging & Release
=================

**Merged PR**:

- Code in main branch
- Automatic in next release

**Release cycle**:

- Minor: Bug fixes, small features (monthly)
- Minor: New features (quarterly)
- Major: Breaking changes (as needed)

**Version numbering**:

- MAJOR.MINOR.PATCH
- 0.1.0 → 0.2.0 (new feature)
- 0.2.0 → 0.2.1 (bug fix)
- 1.0.0 → 2.0.0 (breaking change)

Getting Help
============

**Questions?**

- GitHub Discussions
- GitHub Issues (labeled as questions)
- Check documentation

**Need to discuss?**

- Start GitHub Discussion
- Comment on related issues

**Mentoring**:

- Ask for help in issue
- Maintainers happy to guide

Code of Conduct
===============

Be respectful and inclusive:

- Treat everyone with respect
- Constructive feedback
- No harassment or discrimination
- Welcoming to all backgrounds

See `CODE_OF_CONDUCT.md <https://github.com/BPro2410/topicmodels_package/blob/main/CODE_OF_CONDUCT.md>`_.

Recognition
===========

Contributors are recognized:

- Listed in ``CONTRIBUTORS.md``
- Mentioned in release notes
- Added as GitHub contributor

Thank You!
==========

Contributors make this project better. Your help is greatly appreciated!

**Interested in contributing?**

- Start with "good first issue" label
- Check documentation style
- Ask questions in discussions
- Submit your first PR!

Questions? Open an issue or start a discussion.

We look forward to collaborating with you!
