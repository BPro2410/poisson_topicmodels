==========
User Guide
==========

This guide provides a gentle introduction to the functionality of ``PyPF``. PyPF is desined as an easy-to-use Python library for a general audience of practitions that 
are interested in applying text mining techniques to their own text analysis tasks. 
The workflow for all text mining tasks with PyPF can be divided in three steps: 

+ model initialization, 
+ model training and 
+ model evaluation. 

PyPF requires to prespecify the model input by the user. This way, we believe that user can pre-process their data with no limits to get the best results. 

Prerequisites
----------------

Lets perform a simple topic modeling task. Therefore, we will use plain Poisson Factorization to model
topic proportions across documents.

First, we load any kind of text data.

.. code-block:: python

    # Example documents
    documents = [
        "My smartphone's battery life is fantastic, lasts all day!",
        "The camera on my phone is incredible, takes crystal-clear photos.",
        "Love the smooth performance, but it overheats with heavy apps.",
        "This phone charges super fast, very convenient.",
        "My computer sometimes freezes, but a restart fixes it.",
        "Best laptop I’ve owned, powerful and reliable!"
    ]

The text data can be preprocessed in a format required by the user. Next we create all relevant information for the PF PyPF API, that is:

+ a counts matrix (scipy sparse object)
+ the vocabulary used for the counts matrix


.. code-block:: python

    from sklearn.feature_extraction.text import CountVectorizer
    import scipy.sparse as sparse

    cv = CountVectorizer(stop_words='english', min_df = 1)
    counts = sparse.csr_matrix(cv.fit_transform(documents), dtype = np.float32)

Creating a Model
----------------

All models are accessible through the main factory class:

.. code-block:: python

   from PyPF import topicmodels

   model = topicmodels("PF", counts, cv.vocab, num_topics=20, batch_size=128)

Training a Model
----------------

Each model implements a stochastic variational inference (SVI) training loop.

.. code-block:: python

   estimated_params = model.train_step(num_steps=1000, lr=0.01)

Inspecting Results
------------------

After training, you can inspect learned parameters:

.. code-block:: python

   topics, proportions = model.return_topics()
   beta = model.return_beta()

   print("Document topics:", topics)
   print("Top words for topic 0:", beta.iloc[:, 0].nlargest(10))

Specialized Models
------------------

- **SPF / CSPF** require a dictionary of seed keywords.
- **CPF / CSPF** require a covariate design matrix.
- **TBIP** requires author labels for documents.
