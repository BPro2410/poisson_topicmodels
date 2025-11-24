import jax
import numpy as np

jax.random.PRNGKey(0)


random_seed = np.random.randint(0, 2**31 - 1)
jax.random.PRNGKey(random_seed)



import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sparse

sparse_counts = sparse.random(20, 100, density=0.01, format="csr", dtype=np.float32)
        model = SPF(
            sparse_counts,
            small_vocab,
            keywords_dict,
            residual_topics=2,
            batch_size=5,
        )
        assert model.D == 20
        assert sparse_counts.nnz / (20 * 100) <= 0.01

dtm = sparse.random(10, 50, density=0.1, format="csr", dtype=np.float32)
vocab = np.array([f"word_{i}" for i in range(50)])

model1 = PF(dtm, vocab, num_topics=3, batch_size=5)
model1.train_step(num_steps=3, lr=0.01, random_seed=42)
