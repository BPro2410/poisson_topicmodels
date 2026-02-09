.. _how_to_guides:

================================================================================
How-To Guides
================================================================================

Practical recipes for common topic modeling tasks.

Quick Navigation
================

**Data & Input**

- :doc:`load_data` – Prepare text data for topic modeling

**Training & Configuration**

- :doc:`train_models` – Advanced training techniques
- :doc:`extract_results` – Get and analyze results
- :doc:`customize_inference` – Fine-tune inference

**Evaluation & Improvement**

- :doc:`evaluate_models` – Assess model quality
- :doc:`troubleshoot` – Fix common issues

How-To Topics
=============

load_data
---------

How to load and preprocess text data into document-term matrices.

Topics covered:

- Loading text files
- Tokenization and cleaning
- Creating vocabulary
- Building document-term matrices
- Handling sparse formats
- Dealing with edge cases

train_models
------------

Advanced approaches to training topic models.

Topics covered:

- Mini-batch vs full-batch training
- Multi-GPU training (if supported)
- Reproducibility and seeding
- Monitoring training progress
- Saving and loading models

extract_results
---------------

How to get and work with model results.

Topics covered:

- Topic distributions (topics × vocabulary)
- Document distributions (documents × topics)
- Top words for interpretation
- Document recommendations
- Exporting results

customize_inference
-------------------

Advanced customization of the inference process.

Topics covered:

- Adjusting priors
- Custom initialization
- Semi-supervised variants
- Domain adaptation

evaluate_models
---------------

Methods for assessing topic quality without ground truth.

Topics covered:

- Coherence metrics
- Perplexity and surprisal
- Topic diversity
- Document coverage analysis
- Visualization techniques

troubleshoot
------------

Solutions to common problems.

Topics covered:

- Data issues
- Training problems
- GPU issues
- Memory problems
- Unexpected results

Common Scenarios
================

**"I want to prepare my text data"**

→ :doc:`load_data`

**"Training is slow or uses too much memory"**

→ :doc:`troubleshoot` + :doc:`../tutorials/tutorial_gpu`

**"My topics look bad"**

→ :doc:`evaluate_models` + :doc:`troubleshoot`

**"I need to extract results for further analysis"**

→ :doc:`extract_results`

**"I want to customize the model"**

→ :doc:`customize_inference` + :doc:`../fundamentals/index`

**"I'm unsure about optimal settings"**

→ :doc:`../tutorials/tutorial_hyperparameters`

Organized by Task
=================

- **Data Preparation** – :doc:`load_data` – From raw text to document-term matrices
- **Model Training** – :doc:`train_models` – Advanced training techniques
- **Result Extraction** – :doc:`extract_results` – Get and analyze model outputs
- **Model Customization** – :doc:`customize_inference` – Fine-tune inference process
- **Model Evaluation** – :doc:`evaluate_models` – Assess and compare models
- **Troubleshooting** – :doc:`troubleshoot` – Fix common issues

Best Practices
==============

**General workflow**:

1. :doc:`load_data` – Prepare data
2. :doc:`train_models` – Train model
3. :doc:`evaluate_models` – Check quality
4. :doc:`extract_results` – Get results
5. :doc:`troubleshoot` – Fix issues if needed

**Tips**:

- Start simple (basic PF) before advanced models
- Use validation set to tune hyperparameters
- Always inspect topics manually
- Track all experiments
- Save intermediate results

**Performance**:

- Use GPU for large datasets (see :doc:`../tutorials/tutorial_gpu`)
- Adjust batch_size for your hardware
- Use sparse matrices for large vocabularies
- Filter rare words to reduce dimensionality

Frequently Asked Questions
=============================

**Q: Which how-to guide should I read first?**

A: Start with :doc:`load_data` if you have raw text, otherwise :doc:`train_models`.

**Q: How do I know if my topics are good?**

A: See :doc:`evaluate_models` for evaluation techniques.

**Q: What if I encounter errors?**

A: Check :doc:`troubleshoot` for solutions.

**Q: Can I combine multiple guides?**

A: Yes! Use them in sequence based on your workflow.

**Q: Where are more advanced topics?**

A: See :doc:`../tutorials/index` for step-by-step tutorials and
:doc:`../fundamentals/index` for theoretical details.

Contributing a Guide
====================

Want to add a new how-to guide? `Contribute on GitHub <https://github.com/BPro2410/topicmodels_package>`_.

Guidelines:

- Focus on practical, actionable steps
- Include working code examples
- Cover common errors and solutions
- Keep length reasonable (1-2 pages)
- Reference related documentation

See :doc:`../contributing_guide/index` for contribution guidelines.
