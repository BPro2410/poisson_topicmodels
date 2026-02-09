.. _examples:

================================================================================
Examples & Applications
================================================================================

Practical examples showing poisson-topicmodels in action.

The `examples/` directory in the repository contains complete, runnable examples.

Example Scripts
===============

**01_getting_started.py**

Basic workflow: load data, train model, interpret results.

Level: Beginner
Time: ~5 minutes

Topics:
- Creating synthetic data
- Training PF model
- Extracting and displaying topics
- Basic interpretation

**02_spf_keywords.py**

Guided topic discovery with keyword priors.

Level: Intermediate
Time: ~10 minutes

Topics:
- Defining seed words
- Training SPF model
- Seed strength parameter
- Comparing with unsupervised PF

**03_cpf_covariates.py**

Modeling topic variation by document metadata.

Level: Intermediate
Time: ~10 minutes

Topics:
- Document covariates
- Training CPF model
- Analyzing covariate effects
- Interpretation

**04_advanced_cspf.py**

Combining seeds and covariates.

Level: Advanced
Time: ~15 minutes

Topics:
- Complex model setup
- Multiple features
- Advanced analysis

Running Examples
================

From command line:

.. code-block:: bash

   cd examples
   python 01_getting_started.py
   python 02_spf_keywords.py
   # etc.

In Jupyter (recommended):

.. code-block:: bash

   cd examples
   jupyter notebook run_topicmodels.ipynb

Example Notebooks
=================

**run_topicmodels.ipynb**

Interactive notebook covering:

- Data preparation
- Model training
- Visualization
- All model types
- Interpretation techniques

Open with Jupyter:

.. code-block:: bash

   jupyter notebook examples/run_topicmodels.ipynb

Use this for learning and experimentation!

Custom Examples
===============

Want to create your own example?

**Checklist**:

1. Clear problem statement
2. Realistic data
3. Step-by-step code
4. Results interpretation
5. Key insights highlighted

**Template**:

.. code-block:: python

   """
   Example: [Clear title describing what it does]

   This example demonstrates:
   - Point 1
   - Point 2
   - Point 3
   """

   import numpy as np
   from poisson_topicmodels import [Model]

   # 1. Data preparation
   # ... load/create data

   # 2. Model setup
   model = [Model](...)

   # 3. Training
   model.train(...)

   # 4. Analysis
   # ... extract and analyze results

   # 5. Interpretation
   # ... discuss findings

Applications
============

Real-world use cases where poisson-topicmodels excels:

**Political Analysis**

Ideal for legislative or political discourse analysis.

Topics:
- Bill text analysis
- Speech topic discovery
- Ideal point estimation (TBIP)
- Political polarization measurement

**Social Media Analysis**

Understand trending topics and discourse.

Topics:
- Tweet topic discovery
- Hashtag grouping
- User position estimation (TBIP)
- Discourse evolution

**Academic Research**

Explore research literature and trends.

Topics:
- Paper topic discovery
- Literature reviews
- Research trend analysis
- Cross-discipline connections

**Business & Marketing**

Customer and product insights from text.

Topics:
- Customer review analysis
- Product feedback grouping
- Sentiment-topic combinations
- Market trend discovery

**News & Media**

Content understanding and organization.

Topics:
- News story classification
- Event detection
- Editorial stance analysis
- Content trends

**Computational Social Science**

Complex human behavior through language.

Topics:
- Cultural evolution
- Identity discourse
- Value mapping
- Belief structure

Using Examples as Templates
============================

Each example uses a standard pattern you can adapt:

.. code-block:: python

   # 1. Load/create data (copy from example)
   counts = load_counts()
   vocab = load_vocab()

   # 2. Initialize model (adapt parameters)
   model = PF(counts=counts, vocab=vocab, num_topics=20)

   # 3. Train (tune hyperparameters)
   params = model.train(num_iterations=100, learning_rate=0.01)

   # 4. Analyze (customize based on task)
   top_words = model.get_top_words(n=10)
   # ... more analysis

Next Steps
==========

- **Run an example**: Start with ``01_getting_started.py``
- **Try in notebook**: Open ``run_topicmodels.ipynb``
- **Create your own**: Adapt template for your problem
- **Learn more**: See :doc:`../fundamentals/index` for theory
- **Solve problems**: Check :doc:`../how_to_guides/index` for recipes

Contributing Examples
=====================

Want to share an example?

**Process**:

1. Create a clear, documented script or notebook
2. Include comments explaining each step
3. Use realistic data (or explain synthetic data generation)
4. Test that it runs without errors
5. Submit as pull request to examples/

See :doc:`../contributing_guide/index` for contribution guidelines.

File Organization
=================

.. code-block:: text

   examples/
   ├── 01_getting_started.py       # Beginner
   ├── 02_spf_keywords.py          # Intermediate
   ├── 03_cpf_covariates.py        # Intermediate
   ├── 04_advanced_cspf.py         # Advanced
   ├── run_topicmodels.ipynb       # Interactive
   └── README.md                   # This file

Quick Reference
===============

+---------------------+---------------+----------+
| Script              | Level         | Duration |
+=====================+===============+==========+
| 01_getting_started  | Beginner      | 5 min    |
+---------------------+---------------+----------+
| 02_spf_keywords     | Intermediate  | 10 min   |
+---------------------+---------------+----------+
| 03_cpf_covariates   | Intermediate  | 10 min   |
+---------------------+---------------+----------+
| 04_advanced_cspf    | Advanced      | 15 min   |
+---------------------+---------------+----------+
| run_topicmodels     | Interactive   | Variable |
+---------------------+---------------+----------+

Support
=======

- **Issues with examples?** Open GitHub issue
- **Have an example idea?** Suggest in discussions
- **Want to contribute?** See contribution guide
