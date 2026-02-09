.. _how_to_guides:

================================================================================
How-To Guides
================================================================================

Practical recipes for common topic modeling tasks.

.. note::

   Individual how-to guides for specific tasks are coming soon.
   For now, please refer to:

   - :doc:`../tutorials/index` for complete step-by-step examples
   - :doc:`../fundamentals/index` for understanding core concepts
   - :doc:`../getting_started/index` for a quick introduction

Common Topics
=============

**Data & Input**

- Loading text files and creating document-term matrices
- Tokenization, cleaning, and preprocessing
- Working with sparse matrix formats
- Creating and managing vocabularies

**Training & Configuration**

- Mini-batch vs full-batch training
- GPU acceleration and multi-GPU setups
- Hyperparameter tuning and validation
- Reproducibility with seeds

**Results & Analysis**

- Extracting topic distributions and top words
- Interpreting and visualizing results
- Model evaluation and coherence metrics
- Exporting results for downstream analysis

**Troubleshooting**

- Handling data issues and edge cases
- GPU memory problems
- Training failures and convergence issues
- Improving topic quality

Tips & Best Practices
=====================

**General Workflow**

1. **Prepare Data**: Load and clean text, create document-term matrix
2. **Train Model**: Start with basic PF before trying advanced variants
3. **Evaluate**: Check topic quality with metrics and manual inspection
4. **Extract & Analyze**: Get top words, distributions, and visualizations
5. **Improve**: Adjust hyperparameters or model type if needed

**Performance Optimization**

- Use GPU for datasets with >100k documents (see :doc:`../tutorials/tutorial_gpu`)
- Filter rare words to reduce vocabulary size
- Use sparse matrices for large inputs
- Start with fewer topics for testing, then scale up

**Model Selection**

- **PF**: Start here - simple, unsupervised baseline
- **SPF**: Use if you have domain knowledge (keywords/seeds)
- **CPF/CSPF**: Use if documents have metadata (authors, dates, etc.)
- **ETM**: Use if you have pre-trained word embeddings
- **TBIP**: Use for discovering ideological positions (political text)

Learn More
==========

- For theoretical foundations: :doc:`../fundamentals/index`
- For complete tutorials: :doc:`../tutorials/index`
- For API details: :doc:`../api/index`
- For examples: :doc:`../examples_guide/index`
