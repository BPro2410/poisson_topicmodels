.. _tutorial_hyperparameters:

================================================================================
Tutorial: Hyperparameter Tuning
================================================================================

Systematically optimize your topic models for best performance.

**Duration**: ~15 minutes
**Prerequisites**: :doc:`tutorial_validation`

Key Hyperparameters
===================

Three main hyperparameters control model quality and training:

1. **num_topics (K)**: How many topics to discover
2. **learning_rate (lr)**: Optimization step size
3. **batch_size**: Documents per training iteration

Lesser-used but important:

- **num_iterations**: Training steps (usually 100-1000)
- **random_seed**: Reproducibility

num_topics: The Critical Parameter
===================================

The number of topics affects results the most.

**Too few topics**:

- ❌ Vague, broad themes
- ❌ Everything maps to few topics
- ❌ Loss of granularity

**Too many topics**:

- ❌ Redundant/overlapping topics
- ❌ Low coherence
- ❌ Noise capture

**Optimal**:

- ✓ Coherent, interpretable themes
- ✓ No obvious redundancy
- ✓ Good downstream performance

Selecting num_topics:

.. code-block:: python

   # Strategy: try multiple values
   num_topics_to_try = [5, 10, 15, 20, 30, 50, 75, 100]

   results = {}
   for k in num_topics_to_try:
       print(f"Training with {k} topics...")
       model = PF(counts, vocab, num_topics=k, batch_size=64)
       model.train(num_iterations=100, learning_rate=0.01)

       coherence = model.compute_coherence()
       results[k] = {
           'coherence_mean': coherence.mean(),
           'coherence_std': coherence.std(),
           'coherence_min': coherence.min(),
           'model': model
       }

   # Analyze results
   import pandas as pd
   df = pd.DataFrame(results).T
   print(df)

   # Best by coherence
   best_k = df['coherence_mean'].idxmax()
   print(f"\nOptimal num_topics: {best_k}")

**Practical guideline**:

- **Small corpus** (<10k docs): Start with K=5-20
- **Medium corpus** (10-100k docs): K=20-50
- **Large corpus** (100k+ docs): K=50-200

learning_rate: Optimization Speed
==================================

Controls how fast the model learns.

**Too low (0.001)**:

- Learning is very slow
- Many iterations needed
- More stable but inefficient

**Too high (0.5)**:

- Learning is erratic
- Loss may increase
- Unstable convergence

**Just right (0.01-0.05)**:

- Steady decrease in loss
- Converges in reasonable time
- Reproducible results

Finding optimal learning rate:

.. code-block:: python

   lrs_to_try = [0.001, 0.005, 0.01, 0.05, 0.1]

   for lr in lrs_to_try:
       print(f"\nTraining with learning_rate={lr}")
       model = PF(counts, vocab, num_topics=20, batch_size=64)

       # Track loss over iterations
       loss_history = []
       for i in range(10):
           params = model.train(num_iterations=10, learning_rate=lr, verbose=False)
           loss_history.append(params.get('loss', float('nan')))

       avg_loss = np.mean(loss_history)
       print(f"  Average loss: {avg_loss:.1f}")

       # Loss should be stable and decreasing
       # Erratic loss = lr too high

**Default recommendation**: Start with 0.01

batch_size: Gradient Stability
===============================

Batch size affects gradient noise and GPU utilization.

**Too small (16)**:

- Very noisy gradients
- Unstable training
- Each iteration fast but need many
- Not efficient on GPU

**Too large (1024)**:

- Stable gradients
- Few iterations needed
- Slower per-iteration
- May not fit in GPU memory

**Balanced (64-256)**:

- Good stability
- Good GPU utilization
- Efficient training

Choosing batch_size:

.. code-block:: python

   # Rule of thumb: experiment with powers of 2
   batch_sizes = [32, 64, 128, 256, 512]

   for bs in batch_sizes:
       model = PF(counts, vocab, num_topics=20, batch_size=bs)

       import time
       t0 = time.time()
       model.train(num_iterations=50, learning_rate=0.01)
       elapsed = time.time() - t0

       print(f"batch_size={bs:3d}: {elapsed:.1f}s")
       # Find sweet spot of speed/quality

With GPU:

- Start with batch_size=256 or 512
- Increase until GPU memory error
- Then reduce by half

Systematic Hyperparameter Search
=================================

Grid search over parameter combinations:

.. code-block:: python

   from itertools import product

   # Parameter grid
   param_grid = {
       'num_topics': [10, 20],
       'learning_rate': [0.01, 0.05],
       'batch_size': [64, 128]
   }

   best_score = -float('inf')
   best_params = None

   # Grid search
   for (k, lr, bs) in product(*param_grid.values()):
       params = {'num_topics': k, 'learning_rate': lr, 'batch_size': bs}

       model = PF(counts, vocab, **params)
       model.train(num_iterations=100)

       # Evaluate
       coherence = model.compute_coherence()
       score = coherence.mean()

       print(f"K={k}, lr={lr}, bs={bs}: coherence={score:.3f}")

       if score > best_score:
           best_score = score
           best_params = params

   print(f"\nBest parameters: {best_params}")
   print(f"Best coherence: {best_score:.3f}")

**Warning**: Grid search is expensive. With 100 combinations and 100 iterations each:

.. code-block:: text

   100 models × 100 iterations × 1 minute per 100 iters = 167 hours (!!)

   Solution: Use random search or limit combinations
   Or better: use GPU (10-40x faster)

Random Search (More Efficient)
==============================

.. code-block:: python

   import numpy as np

   # Sample 20 random combinations from space
   n_trials = 20
   results = []

   for trial in range(n_trials):
       # Random parameters
       k = np.random.choice([10, 15,20, 30, 50])
       lr = np.random.uniform(0.001, 0.1)  # log scale recommended
       bs = np.random.choice([32, 64, 128, 256])

       model = PF(counts, vocab, num_topics=k, batch_size=bs)
       model.train(num_iterations=100, learning_rate=lr)

       coherence = model.compute_coherence()
       results.append({
           'num_topics': k,
           'learning_rate': lr,
           'batch_size': bs,
           'coherence': coherence.mean()
       })

       print(f"Trial {trial+1}/{n_trials}: coherence={results[-1]['coherence']:.3f}")

   # Best configuration
   best_idx = np.argmax([r['coherence'] for r in results])
   best_config = results[best_idx]
   print(f"Best: {best_config}")

Practical Tuning Strategy
=========================

**Step 1: Find good num_topics** (most important)

.. code-block:: python

   # Try 5 values: rough search
   for k in [10, 20, 35, 50, 75]:
       model = PF(counts, vocab, num_topics=k)
       model.train(num_iterations=100, learning_rate=0.01)
       coherence = model.compute_coherence()
       print(f"K={k}: {coherence.mean():.3f}")

**Step 2: Refine around best K**

.. code-block:: python

   # If K=35 was best, try nearby
   best_k = 35
   for k in range(30, 41, 1):  # 30-40
       model = PF(counts, vocab, num_topics=k)
       model.train(num_iterations=100, learning_rate=0.01)
       coherence = model.compute_coherence()
       print(f"K={k}: {coherence.mean():.3f}")

**Step 3: Tune lr and batch_size**

.. code-block:: python

   # With best K, try different lr values
   best_k = 35  # from previous step

   for lr in [0.005, 0.01, 0.02, 0.05]:
       model = PF(counts, vocab, num_topics=best_k)
       model.train(num_iterations=100, learning_rate=lr)
       coherence = model.compute_coherence()
       print(f"lr={lr}: {coherence.mean():.3f}")

**Step 4: Final validation**

.. code-block:: python

   # Train final model with best parameters
   best_params = {'num_topics': 35, 'learning_rate': 0.02, 'batch_size': 128}
   final_model = PF(counts, vocab, **best_params)
   final_model.train(num_iterations=200)  # More iterations for final

   # Validate
   coherence = final_model.compute_coherence()
   print(f"Final model coherence: {coherence.mean():.3f}")

Early Stopping
==============

Stop training when loss plateaus:

.. code-block:: python

   model = PF(counts, vocab, num_topics=20, batch_size=64)

   loss_history = []
   patience = 10  # Stop if no improvement for 10 iterations
   best_loss = float('inf')
   patience_counter = 0

   for epoch in range(100):
       params = model.train(num_iterations=10, learning_rate=0.01)
       current_loss = params.get('loss', float('nan'))
       loss_history.append(current_loss)

       print(f"Epoch {epoch+1}: loss={current_loss:.1f}")

       # Check for improvement
       if current_loss < best_loss - 1.0:  # Improvement threshold
           best_loss = current_loss
           patience_counter = 0
           print("  ✓ Improvement!")
       else:
           patience_counter += 1
           print(f"  No improvement ({patience_counter}/{patience})")

       if patience_counter >= patience:
           print("Early stopping!")
           break

Documenting Experiments
=======================

Track your hyperparameter explorations:

.. code-block:: python

   import logging

   logging.basicConfig(
       filename='hyperparameter_log.txt',
       level=logging.INFO,
       format='%(asctime)s - %(message)s'
   )

   for k in [20, 30, 50]:
       model = PF(counts, vocab, num_topics=k)
       model.train(num_iterations=100, learning_rate=0.01)
       coherence = model.compute_coherence()

       logging.info(f"K={k}: coherence={coherence.mean():.3f}")

Common Mistakes & Solutions
============================

**Mistake**: Tuning learning_rate too aggressively

*Solution*: It's usually not the bottleneck. Focus on K first.

**Mistake**: Grid search over too many combinations

*Solution*: Use random search or tune one parameter at a time.

**Mistake**: Not tracking which configurations you've tried

*Solution*: Keep a log with timestamps and results.

**Mistake**: Overfitting to coherence on one dataset

*Solution*: Validate on held-out documents, multiple datasets.

**Mistake**: Not using GPU

*Solution*: Enable GPU - changes game for hyperparameter search!

Tuning Checklist
================

✓ Focus on num_topics first (most impact)
✓ Try at least 5 different values
✓ Use GPU to enable faster experimentation
✓ Document all trials
✓ Validate on held-out data
✓ Use early stopping when possible
✓ Final training: more iterations than tuning

Next Steps
==========

- Want to understand models better? See :doc:`../fundamentals/index`
- Ready to use your model? See :doc:`../how_to_guides/index`
- Need production setup? See :doc:`../contributing_guide/index`

Summary
=======

1. Num_topics is the most important parameter
2. Learning rate usually fine at 0.01
3. Batch size affects speed, not much else
4. Use GPU to enable rapid experimentation
5. Track all experiments for reproducibility
6. Stop training when loss plateaus
