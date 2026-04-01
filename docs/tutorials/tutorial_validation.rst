.. _tutorial_validation:

================================================================================
Tutorial: Model Validation & Evaluation
================================================================================

How to assess the quality of your trained topic models.

**Duration**: ~10 minutes
**Prerequisites**: :doc:`tutorial_training`

Validation Approaches
=====================

Three complementary approaches to validate models:

1. **Qualitative**: Manual inspection of topics
2. **Quantitative**: Metrics (coherence, perplexity)
3. **Downstream**: Performance on actual tasks

The Coherence Metric
====================

Coherence measures if top words of a topic are semantically related.

.. code-block:: python

   coherence_df = model.compute_topic_coherence()
   coherence = coherence_df['coherence'].values
   print(f"Coherence per topic: {coherence}")
   print(f"Average: {coherence.mean():.3f}")

   # Topic diversity: are topics distinct?
   diversity = model.compute_topic_diversity()
   print(f"Topic diversity: {diversity:.3f}")
   # 1.0 = all unique words, 0.0 = identical top words

Interpreting values:

- **0.6+**: Excellent coherence (words form clear themes)
- **0.4-0.6**: Good (topics are interpretable)
- **0.2-0.4**: Fair (some coherence but noisy)
- **<0.2**: Poor (topics are incoherent)

Finding low-coherence topics:

.. code-block:: python

   worst_topics = np.argsort(coherence)[:5]
   top_words = model.return_top_words_per_topic(n=10)
   for topic_id in worst_topics:
       words = top_words[topic_id]
       print(f"Topic {topic_id} (coherence={coherence[topic_id]:.3f}):")
       print(f"  {', '.join(words)}")

Qualitative Inspection
======================

Manual topic interpretation:

.. code-block:: python

   def evaluate_topics_manually(model, num_to_show=10):
       """Inspect top words for each topic."""
       top_words = model.return_top_words_per_topic(n=20)

       ratings = {}

       for topic_id, words in top_words.items():
           print(f"\n=== Topic {topic_id} ===")
           print(f"Top words: {', '.join(words[:10])}")

           # Rate quality: 1=bad, 2=poor, 3=fair, 4=good, 5=excellent
           rating = input("Rate this topic (1-5, q=quit): ")
           if rating.lower() == 'q':
               break
           if rating.isdigit():
               ratings[topic_id] = int(rating)

       return ratings

   # Use it
   ratings = evaluate_topics_manually(model)
   avg_rating = np.mean(list(ratings.values()))
   print(f"\nAverage rating: {avg_rating:.1f} / 5")

Checklist for topic quality:

.. code-block::

   ✓ Top words form coherent theme
   ✓ You can give topic a meaningful label
   ✓ Topic isn't all stopwords or common terms
   ✓ Topic doesn't duplicate another topic
   ✓ Topic isn't a garbage catch-all

Comparative Evaluation
======================

Compare multiple model configurations:

.. code-block:: python

   results = {}

   # Try different numbers of topics
   for num_topics in [5, 10, 20, 50]:
       model = PF(counts, vocab, num_topics=num_topics, batch_size=32)
       model.train_step(num_steps=200, lr=0.01)

       coherence_df = model.compute_topic_coherence()
       coherence = coherence_df['coherence'].values
       results[num_topics] = {
           'coherence_mean': coherence.mean(),
           'coherence_std': coherence.std(),
           'diversity': model.compute_topic_diversity(),
           'model': model
       }

   # Display results
   print("Performance by number of topics:")
   for k, v in results.items():
       print(f"  K={k}: coherence={v['coherence_mean']:.3f} ± {v['coherence_std']:.3f}")

   # Pick best and visualize
   best_k = max(results, key=lambda x: results[x]['coherence_mean'])
   print(f"\nBest configuration: {best_k} topics")

Downstream Task Evaluation
===========================

If you have a downstream task, evaluate model performance there:

.. code-block:: python

   # Example: Use topics for document classification
   from sklearn.ensemble import RandomForestClassifier

   # Get document-topic representations
   doc_topics_result = model.return_topics()
   _, e_theta = doc_topics_result

   # Train classifier on topics
   clf = RandomForestClassifier()
   clf.fit(e_theta, labels)  # labels = ground truth

   # Evaluate
   accuracy = clf.score(doc_topics, labels)
   print(f"Classification accuracy: {accuracy:.3f}")

   # Compare with other models
   results['model_quality'] = accuracy

Topic Similarity Analysis
==========================

Are topics overlapping? Check similarity:

.. code-block:: python

   from sklearn.metrics.pairwise import cosine_similarity

   beta = model.return_beta()
   similarity = cosine_similarity(beta.values.T)
   np.fill_diagonal(similarity, 0)

   # Find similar pairs
   similar = np.where(similarity > 0.7)
   top_words = model.return_top_words_per_topic(n=5)
   for i, j in zip(similar[0], similar[1]):
       if i < j:
           print(f"Topic {i} and {j} are similar (sim={similarity[i, j]:.3f})")
           print(f"  Topic {i}: {', '.join(top_words[i])}")
           print(f"  Topic {j}: {', '.join(top_words[j])}")

Document Coverage
=================

Do all documents get meaningful topic assignments?

.. code-block:: python

   _, e_theta = model.return_topics()

   # Topic concentration per document
   doc_entropy = -np.sum(e_theta * np.log(e_theta + 1e-10), axis=1)
   max_probability = e_theta.max(axis=1)

   print(f"Document topic concentration:")
   print(f"  Max topic probability: {max_probability.mean():.3f} ± {max_probability.std():.3f}")
   print(f"  Entropy: {doc_entropy.mean():.3f} ± {doc_entropy.std():.3f}")

   # Low entropy = document in few topics (concentrated)
   # High entropy = document spread across topics (diffuse)

   # Are we getting good coverage?
   if max_probability.mean() < 0.3:
       print("Warning: Documents don't concentrate on topics")
       print("  → Consider increasing num_topics or more training")

Visualization for Validation
=============================

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(2, 2, figsize=(12, 10))

   # 1. Coherence distribution
   coherence_df = model.compute_topic_coherence()
   coherence = coherence_df['coherence'].values
   axes[0, 0].hist(coherence, bins=20, edgecolor='black')
   axes[0, 0].set_xlabel('Coherence')
   axes[0, 0].set_ylabel('Number of topics')
   axes[0, 0].set_title('Topic Coherence Distribution')
   axes[0, 0].axvline(coherence.mean(), color='red', linestyle='--', label='Mean')
   axes[0, 0].legend()

   # 2. Topic prevalence (or use built-in: model.plot_topic_prevalence())
   _, e_theta = model.return_topics()
   avg_topics = e_theta.mean(axis=0)
   axes[0, 1].bar(range(len(avg_topics)), avg_topics)
   axes[0, 1].set_xlabel('Topic ID')
   axes[0, 1].set_ylabel('Average intensity')
   axes[0, 1].set_title('Topic Prevalence')

   # 3. Document entropy
   doc_entropy = -np.sum(e_theta * np.log(e_theta + 1e-10), axis=1)
   axes[1, 0].hist(doc_entropy, bins=30, edgecolor='black')
   axes[1, 0].set_xlabel('Entropy')
   axes[1, 0].set_ylabel('Number of documents')
   axes[1, 0].set_title('Document Topic Dispersion')

   # 4. Top vs average coherence
   top_topics = np.argsort(coherence)[-5:]
   bottom_topics = np.argsort(coherence)[:5]
   axes[1, 1].barh(range(5), coherence[bottom_topics], alpha=0.5, label='Worst')
   axes[1, 1].barh(range(5, 10), coherence[top_topics], alpha=0.5, label='Best')
   axes[1, 1].set_yticks(range(10))
   axes[1, 1].set_yticklabels(list(bottom_topics) + list(top_topics))
   axes[1, 1].set_xlabel('Coherence')
   axes[1, 1].set_title('Best vs Worst Topics')
   axes[1, 1].legend()

   plt.tight_layout()
   plt.show()

Validation Checklist
====================

Before deploying a model:

✓ Average coherence > 0.4
✓ No garbage topics (all stopwords)
✓ Topics aren't highly overlapping
✓ Manual inspection: topics make sense
✓ Downstream task performance acceptable
✓ Coverage: documents get meaningful topics
✓ Reproducibility: same seed → same results

Red Flags
=========

**Model probably needs improvement if**:

- ❌ Most topics have low coherence (<0.3)
- ❌ Can't label most topics meaningfully
- ❌ Many topics are duplicates
- ❌ Some topics are all stopwords/garbage
- ❌ Downstream task performance is poor
- ❌ Many documents have flat topic distribution

**Next steps when validation fails**:

1. Try more training iterations
2. Adjust learning rate
3. Change number of topics
4. Improve data preprocessing
5. Try guided/seeded variant (SPF)
6. Add covariates if available (CPF)

See :doc:`tutorial_hyperparameters` for optimization strategies.

Validation Workflow
===================

.. code-block:: text

   1. Train model with initial config
   2. Compute coherence
   3. Visualize and inspect topics
   4. Check for duplicates
   5. Evaluate downstream performance

   If quality acceptable: ✓ Done
   If not:
   6. Adjust configuration
   7. Retrain and repeat from 2

Version Tracking
================

Keep records of model evaluations:

.. code-block:: python

   import json
   from datetime import datetime

   def save_evaluation(model_name, config, results):
       """Save model evaluation results."""
       eval_record = {
           'timestamp': datetime.now().isoformat(),
           'model_name': model_name,
           'config': config,
           'results': {
               'mean_coherence': float(results['coherence'].mean()),
               'std_coherence': float(results['coherence'].std()),
               'num_low_quality': results.get('low_quality_count', 0),
           }
       }

       with open('evaluations.json', 'a') as f:
           f.write(json.dumps(eval_record) + '\n')

   # Use it
   config = {'num_topics': 20, 'learning_rate': 0.01}
   save_evaluation('pf_model_v1', config, {'coherence': coherence})

Next Steps
==========

- Satisfied? Move to :doc:`tutorial_hyperparameters` for fine-tuning
- Need to optimize? See :doc:`../how_to_guides/index`
- Want production-ready? Check :doc:`../contributing_guide/index` for best practices
