.. _release_notes:

================================================================================
Release Notes & Changelog
================================================================================

Version history and changelog for poisson-topicmodels.

.. toctree::
   :maxdepth: 1
   :caption: Versions

   changelog

Current Version
===============

**Latest**: 0.1.0 (February 2025)

For full changelog, see below.

Version 0.1.0 - Initial Release
===============================

*February 9, 2025*

**New Features**:

- ✨ Poisson Factorization (PF) - Unsupervised topic modeling
- ✨ Seeded PF (SPF) - Guided topic discovery with keyword priors
- ✨ Covariate PF (CPF) - Model topic variation by document metadata
- ✨ Covariate Seeded PF (CSPF) - Combine seeds and covariates
- ✨ Text-Based Ideal Points (TBIP) - Estimate author positions
- ✨ Embedded Topic Models (ETM) - Integration with pre-trained embeddings
- ✨ Stochastic Variational Inference (SVI) with mini-batch training
- ✨ GPU acceleration via JAX
- ✨ Comprehensive type hints (90%+ coverage)
- ✨ >70% test coverage
- ✨ Complete documentation and tutorials

**Documentation**:

- Getting Started guide
- Fundamentals covering all models
- 4 detailed tutorials
- How-to guides for common tasks
- Complete API reference
- Testing guide
- Contributing guidelines

**Quality**:

- Code follows Black, isort, mypy standards
- Comprehensive error messages
- Input validation
- Reproducibility via seeding
- GitHub Actions CI/CD

**Breaking Changes**:

- N/A (first release)

**Migration Guide**:

- N/A (first release)

**Contributors**:

- Bernd Prostmaier (Lead)
- Bettina Grün
- Paul Hofmarcher

**Known Issues**:

- GPU memory estimation could be improved
- Some edge cases in covariate handling
- Documentation for advanced features could be more detailed

**Future Roadmap**:

- Dynamic topic models (time-varying topics)
- Online learning from streaming data
- Hierarchical topic models (HTM)
- Better visualization toolkit
- Performance optimizations
- Additional embedding support

Coming Soon (0.2.0)
===================

Planned features:

- **Dynamic Topic Models**: Topics that evolve over time
- **Streaming Mode**: Learn from new documents continuously
- **Better Visualization**: Interactive topic visualization
- **More Metrics**: Additional evaluation metrics
- **Documentation Enhancements**: More examples and tutorials

Not Planned for 0.2.0:

- Breaking API changes (we'll maintain compatibility)
- Complete rewrite of inference (current approach is solid)

Deprecation Policy
==================

**Stability Guarantees**:

- Public API stable across minor versions
- Breaking changes only in major versions
- Deprecations announced one release ahead

**How to stay updated**:

- Watch GitHub releases
- Subscribe to changelog
- Monitor upgrade guides

Installation by Version
=======================

Install specific version:

.. code-block:: bash

   pip install poisson-topicmodels==0.1.0

List available versions:

.. code-block:: bash

   pip index versions poisson-topicmodels

Upgrade to latest:

.. code-block:: bash

   pip install --upgrade poisson-topicmodels

Compatibility Matrix
====================

+---------------------+-------+-------+-------+
| poisson-topicmodels | Python 3.11 | Python 3.12 | Python 3.13 |
+=====================+===============+===============+===============+
| 0.1.0               | ✓           | ✓           | ✓           |
+---------------------+-------+-------+-------+

Dependency Versions
===================

Core dependencies for 0.1.0:

- jax==0.8.0
- jaxlib==0.8.0
- numpyro==0.19.0
- numpy>=2.2.0,<3.0.0
- scipy>=1.15.0,<2.0.0
- pandas>=2.2.0,<3.0.0
- scikit-learn>=1.6.0,<2.0.0
- matplotlib>=3.10.0,<4.0.0

Optional dependencies:

- sphinx>=6.0 (for documentation)
- sphinx-rtd-theme>=1.2 (for docs theme)
- myst-parser>=1.0 (for markdown docs)
- pytest>=9.0.1 (for testing)

Citing poisson-topicmodels
============================

If you use poisson-topicmodels in research, please cite:

.. code-block:: bibtex

   @software{prostmaier2025poisson,
     title={poisson-topicmodels: Probabilistic Topic Modeling with Bayesian Inference},
     author={Prostmaier, Bernd and Grün, Bettina and Hofmarcher, Paul},
     year={2025},
     url={https://github.com/BPro2410/topicmodels_package}
   }

Or in plain text:

Prostmaier, B., Grün, B., & Hofmarcher, P. (2025).
poisson-topicmodels: Probabilistic Topic Modeling with Bayesian Inference.
Retrieved from https://github.com/BPro2410/topicmodels_package

Community Feedback
==================

Help improve poisson-topicmodels:

- 🐛 **Bug reports**: GitHub Issues
- 💡 **Feature ideas**: GitHub Discussions
- 📚 **Documentation feedback**: GitHub Issues
- 💬 **General discussion**: GitHub Discussions
- 🤝 **Contributions**: See Contributing Guide

Report Issues
=============

Found a bug? Please report it:

1. Check existing issues
2. Create detailed bug report
3. Include:
   - Python version
   - OS
   - JAX version
   - Minimal reproducible example
   - Full error traceback

See :doc:`../contributing_guide/index#reporting-bugs` for detailed instructions.

Acknowledgments
===============

**Core Team**:

- Bernd Prostmaier
- Bettina Grün
- Paul Hofmarcher

**Built on**:

- `JAX <https://jax.readthedocs.io>`_ - Automatic differentiation
- `NumPyro <https://numpyro.readthedocs.io>`_ - Probabilistic programming
- `SciPy <https://scipy.org>`_ - Scientific computing
- `Sphinx <https://www.sphinx-doc.org>`_ - Documentation

**Inspired by**:

- Gensim
- scikit-learn
- PyMC
- NumPyro

License
=======

poisson-topicmodels is licensed under the MIT License.

See `LICENSE <https://github.com/BPro2410/topicmodels_package/blob/main/LICENSE>`_
for details.

Next Steps
==========

- **Update to latest**: ``pip install --upgrade poisson-topicmodels``
- **Get started**: See :doc:`../getting_started/index`
- **Report issues**: `GitHub Issues <https://github.com/BPro2410/topicmodels_package/issues>`_
- **Contribute**: :doc:`../contributing_guide/index`

**Questions?** Open a discussion or create an issue on GitHub!

Questions About Releases
=========================

**Q: When is the next release?**

A: Quarterly minor releases, monthly patch releases. Check milestones on GitHub.

**Q: How do I get a feature in the next release?**

A: Open an issue or discussion to propose it, or submit a PR!

**Q: Can I use development version?**

A: Yes: ``pip install git+https://github.com/BPro2410/topicmodels_package.git``

**Q: What about backwards compatibility?**

A: Public API stable within major versions. Breaking changes disclosed ahead of time.
