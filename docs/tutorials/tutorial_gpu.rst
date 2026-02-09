.. _tutorial_gpu:

================================================================================
Tutorial: GPU Acceleration
================================================================================

This tutorial shows how to leverage GPU acceleration for 10-100x faster training.

**Duration**: ~10 minutes
**Prerequisites**: NVIDIA/AMD/Apple GPU available and drivers installed

Why GPU?
========

GPU acceleration is critical for large-scale topic modeling:

- **CPU time**: 1000 docs × 10000 words × 100 iters = hours
- **GPU time**: Same computation = minutes

JAX automatically handles GPU computations when available.

Checking GPU Availability
==========================

First, verify GPU access:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   # List available devices
   devices = jax.devices()
   print("Available devices:")
   for device in devices:
       print(f"  {device}")

Expected output with GPU:

.. code-block:: text

   Available devices:
     NVIDIA A100 GPU (cuda:0)
     NVIDIA A100 GPU (cuda:1)

Without GPU:

.. code-block:: text

   Available devices:
     cpu

Enabling GPU for poisson-topicmodels
============================

**Option 1: Environment variable (recommended)**

.. code-block:: bash

   export JAX_PLATFORMS=gpu
   python your_script.py

**Option 2: Set in Python before import**

.. code-block:: python

   import os
   os.environ['JAX_PLATFORMS'] = 'gpu'

   from poisson_topicmodels import PF
   # Now uses GPU

**Option 3: Use CUDA directly**

.. code-block:: bash

   # Force CUDA devices
   export CUDA_VISIBLE_DEVICES=0,1
   python your_script.py

Setting Up GPU Environment
===========================

**NVIDIA GPU (CUDA)**:

1. Install CUDA Toolkit 11.8+ and cuDNN 8.6+
2. Install GPU-enabled JAX:

.. code-block:: bash

   pip install jax[cuda12_cudnn]

3. Verify:

.. code-block:: bash

   python -c "import jax; print(jax.devices())"

**AMD GPU (ROCm)**:

.. code-block:: bash

   pip install jax[rocm]

**Apple Silicon (Metal)**:

.. code-block:: bash

   pip install jax[metal]

Training with GPU
=================

Once GPU is enabled, training automatically uses it:

.. code-block:: python

   from poisson_topicmodels import PF

   model = PF(counts, vocab, num_topics=20, batch_size=256)

   # This automatically uses GPU if available
   params = model.train(num_iterations=100, learning_rate=0.01)

   # That's it! No code changes needed.

No explicit GPU calls required—JAX handles it transparently.

Monitoring GPU Usage
====================

**Check GPU utilization**:

Command line:

.. code-block:: bash

   # NVIDIA: nvidia-smi shows GPU usage
   nvidia-smi -l 1  # Update every second

   # AMD: rocm-smi
   rocm-smi --watch

Expected during training: 80-95% GPU utilization

**In Python**:

.. code-block:: python

   import subprocess
   import time

   def monitor_gpu():
       """Print GPU utilization."""
       while True:
           result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
                                  '--format=csv,nounits,noheader'],
                                 capture_output=True, text=True)
           utilization = result.stdout.strip()
           print(f"GPU utilization: {utilization}%")
           time.sleep(2)

   # In another terminal while training runs
   monitor_gpu()

Memory Management
=================

**GPU Memory Issues**:

If you get "out of memory" errors:

.. code-block:: python

   # 1. Increase batch size (counterintuitively helps with memory)
   model = PF(counts, vocab, num_topics=20, batch_size=512)

   # 2. Reduce vocabulary size
   # Remove rare words: keep only top 5000 words

   # 3. Reduce number of documents
   # Sample documents or process in chunks

**Memory-efficient training**:

.. code-block:: python

   # Monitor memory during training
   from jax import monitoring

   model = PF(counts, vocab, num_topics=20, batch_size=128)
   params = model.train(num_iterations=100, learning_rate=0.01)

   # If memory issues: reduce batch_size → 64 or 32

Performance Benchmarking
========================

Compare CPU vs GPU timing:

.. code-block:: python

   import time
   from poisson_topicmodels import PF

   # Small dataset
   counts_small = csr_matrix(np.random.poisson(2, (100, 500)).astype(np.float32))
   vocab_small = np.array([f'word_{i}' for i in range(500)])

   # Time CPU (disable GPU first)
   import os
   os.environ['JAX_PLATFORMS'] = 'cpu'

   model_cpu = PF(counts_small, vocab_small, num_topics=10, batch_size=32)
   t0 = time.time()
   model_cpu.train(num_iterations=50)
   cpu_time = time.time() - t0

   # Time GPU
   os.environ['JAX_PLATFORMS'] = 'gpu'  # Requires restart
   # (In practice, use separate scripts or notebooks)

   model_gpu = PF(counts_small, vocab_small, num_topics=10, batch_size=32)
   t0 = time.time()
   model_gpu.train(num_iterations=50)
   gpu_time = time.time() - t0

   print(f"CPU time: {cpu_time:.2f}s")
   print(f"GPU time: {gpu_time:.2f}s")
   print(f"Speedup: {cpu_time/gpu_time:.1f}x")

Real-world example:

.. code-block:: text

   Dataset: 50k documents, 50k vocabulary

   CPU (16 cores):      ~2 hours per 100 iterations
   GPU (A100):          ~3 minutes per 100 iterations
   Speedup:             ~40x

Optimizing for Speed
====================

**Tips for maximum performance**:

1. **Batch Size**: Larger batches = better GPU utilization

   .. code-block:: python

      # Start with batch_size=256 or 512 on modern GPUs
      model = PF(counts, vocab, num_topics=20, batch_size=512)

2. **Multiple GPUs**: Distribute across cards (if supported)

   .. code-block:: bash

      export CUDA_VISIBLE_DEVICES=0,1,2,3
      python script.py  # Uses all 4 GPUs

3. **Mixed Precision**: Trade accuracy for speed (advanced)

   .. code-block:: python

      # JAX supports this via jax.experimental.key_reuse
      # Not currently exposed in poisson-topicmodels
      # Future enhancement

4. **Profiling**: Identify bottlenecks

   .. code-block:: python

      import jax

      # Enable profiling
      jax.profiling.pluck_counts()

      # Train and profile
      model.train(num_iterations=10)

      # Analyze results
      # Check if data transfer or computation is bottleneck

Troubleshooting GPU
===================

**Problem**: JAX can't find GPU

.. code-block::

   jax._src.lib.xla_extension.XlaRuntimeError: CUDA not found

*Solution*:
- Verify CUDA installation: ``nvcc --version``
- Reinstall JAX: ``pip install --upgrade jax[cuda12_cudnn]``
- Check CUDA_HOME: ``echo $CUDA_HOME``

**Problem**: GPU out of memory

*Solution*:
- Reduce batch_size: ``batch_size=64`` instead of 256
- Reduce num_topics
- Reduce vocabulary size
- Process data in chunks

**Problem**: GPU slower than CPU (!?)

*Solution*:
- GPU overhead for small datasets (< 10k docs)
- GPU shines with 100k+ documents
- Check GPU utilization (should be >80%)
- Increase batch_size to improve utilization

**Problem**: Training hangs on GPU

*Solution*:
- Timeout issue with GPU
- Reduce batch_size or num_topics
- Update JAX: ``pip install --upgrade jax``
- Check GPU memory: ``nvidia-smi``

Best Practices
==============

**Development**:

- Start on CPU for quick iterations
- Verify results make sense
- Switch to GPU for final large-scale runs

**Production**:

- Always use GPU for meaningful datasets (100k+ docs)
- Monitor GPU utilization
- Use optimal batch size (64-512 depending on GPU memory)

**Research reproducibility**:

- Document which GPU was used
- Set random seed (results consistent across runs)
- GPU results may slightly differ from CPU due to floating-point precision

Scaling to Large Datasets
=========================

With GPU, you can now handle:

- **100k documents**: ~10 minutes (vs hours on CPU)
- **500k documents**: ~50 minutes
- **1M documents**: ~2 hours

Example:

.. code-block:: python

   # Large dataset
   num_docs, num_words = 500_000, 100_000

   # GPU can handle this
   model = PF(
       counts=counts,
       vocab=vocab,
       num_topics=50,
       batch_size=1024  # Can use large batch on GPU
   )

   # Train efficiently
   params = model.train(
       num_iterations=200,
       learning_rate=0.01
   )

   # Takes ~1 hour instead of whole day

Next Steps
==========

- :doc:`tutorial_training` - Refresh on model training
- :doc:`tutorial_hyperparameters` - Optimize settings for GPU
- :doc:`../how_to_guides/train_models` - Advanced training techniques

Key Takeaway
============

**GPU acceleration requires zero code changes—just enable it!**

Once enabled, training automatically uses GPU for massive speedups.
