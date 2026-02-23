.. _tutorials:

================================================================================
Tutorials
================================================================================

Step-by-step tutorials guiding you through different aspects of topic modeling
with poisson-topicmodels.

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorial_training
   tutorial_gpu
   tutorial_validation
   tutorial_hyperparameters

Tutorial Overview
==================

Each tutorial covers a specific aspect of working with topic models:

**Training & Basics** (:doc:`tutorial_training`)
   How to prepare data, train models, and interpret results. Best for beginners.

**GPU Acceleration** (:doc:`tutorial_gpu`)
   Leverage GPU computation for large-scale analysis and significant speedup.

**Model Validation** (:doc:`tutorial_validation`)
   Techniques to evaluate topic quality and model performance without ground truth.

**Hyperparameter Tuning** (:doc:`tutorial_hyperparameters`)
   Systematic approaches to selecting optimal number of topics, learning rates, and batch sizes.

Quick Start Tutorials
=====================

**I want to train my first topic model**

→ Start with :doc:`../getting_started/index`, then :doc:`tutorial_training`

**I have a large corpus and want to speed things up**

→ Read :doc:`tutorial_gpu`

**I'm not sure if my topics are good**

→ Check :doc:`tutorial_validation`

**My model doesn't seem to improve**

→ Review :doc:`tutorial_hyperparameters`

Prerequisites
=============

All tutorials assume:

- Python 3.11+ installed
- poisson-topicmodels installed (see :doc:`../installation/index`)
- Basic familiarity with topic modeling concepts (see :doc:`../fundamentals/index`)
- Text data preprocessed into document-term matrices

Tutorial Series: From Data to Insights
=======================================

Recommended progression:

1. :doc:`../getting_started/index` - 5-minute quickstart
2. :doc:`tutorial_training` - Train and interpret your first model
3. :doc:`tutorial_validation` - Assess topic quality
4. :doc:`tutorial_hyperparameters` - Optimize your model
5. :doc:`tutorial_gpu` - Scale to large datasets
6. :doc:`../how_to_guides/index` - Practical recipes for common tasks

Troubleshooting Tutorials
==========================

Running into issues? Check relevant tutorials:

- **Installation problems?** See :doc:`../installation/index#troubleshooting`
- **Results don't look good?** See :doc:`tutorial_validation`
- **Training is slow?** See :doc:`tutorial_gpu`
- **Not sure about settings?** See :doc:`tutorial_hyperparameters`
- **Data format issues?** See :doc:`../how_to_guides/load_data`

Feedback
========

Tutorials can always be improved! If you:

- Found a tutorial unclear or incomplete
- Discovered an error
- Have a topic you'd like a tutorial for

Please open an issue on `GitHub <https://github.com/BPro2410/topicmodels_package/issues>`_.
