# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-05

### Added
- Initial public release
- **PF (Poisson Factorization)**: Unsupervised baseline topic model
- **SPF (Seeded Poisson Factorization)**: Guided topic modeling with keyword priors
- **CPF (Covariate Poisson Factorization)**: Topics influenced by external covariates
- **CSPF (Covariate Seeded PF)**: Combines seeded guidance with covariate effects
- **TBIP (Text-Based Ideal Points)**: Estimates latent ideal points from text
- **TVTBIP (Time-Varying TBIP)**: Captures temporal dynamics in author positions
- **ETM (Embedded Topic Models)**: Integrates pre-trained word embeddings
- Comprehensive documentation with tutorials and API reference
- Full test suite with >70% code coverage
- CI/CD pipeline with GitHub Actions (Python 3.11-3.13)
- Example datasets and Jupyter notebooks
- Docker support for reproducible environments
- NumPy-style docstrings for all public APIs
- Type hints on core model interfaces

### Changed
- Updated package structure from `poisson_topicmodels/models/` to standard layout
- Reconciled dependency versions (pyproject.toml and requirements.txt)
- Improved metadata in pyproject.toml

### Fixed
- Dependency version conflicts resolved
- Package installation issues fixed
- Shared mutable state bug in Metrics class

### Documentation
- Added comprehensive README with Statement of Need
- Created CONTRIBUTING.md with developer guidelines
- Created CITATION.cff for automatic citation generation

### Development
- Established GitHub Actions CI/CD workflow
- Created pytest-based test suite
- Added development tools (black, isort, flake8, mypy)
- Set up code coverage tracking

### Known Limitations
- Factory function (`topicmodels("PF", ...)`) to be deprecated in future releases
- GPU support requires JAX-compatible hardware
- Metal GPU support on macOS requires special configuration

## Future Plans

### v0.2.0 (Q1 2026)
- Add more probabilistic models to the stack

---

## Notes on Versioning

- **v0.1.0**: Initial alpha release, API subject to change
- Semantic versioning: MAJOR.MINOR.PATCH (e.g., 0.1.0)
- Pre-1.0.0 releases may have breaking changes with minor version bumps
- Releases synchronized with academic publications when applicable
