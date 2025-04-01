What is PyPF
============

PyPF is desined as an easy-to-use Python library for a general audience of practitions that 
are interested in applying text mining techniques to their own text analysis tasks. 
The workflow for all text mining tasks with PyPF can be divided in three steps: 

+ model initialization, 
+ model training and 
+ model evaluation. 

PyPF requires to prespecify the model input by the user. This way, we believe that user can pre-process their data with no limits to get the best results. 


A simple example
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


Next, we can start to call the PyPF API, select the hyperparameter and fit the required model.

.. code-block:: python

    from PyPF import topicmodels

    # We select the Poisson factorization model 'PF' and initialize the model first
    tm = topicmodels("PF", counts, vocab, num_topics = 10, batch_size = 1024)

    # Set the number of training steps and the learning rate
    estimated_params = tm.train_step(num_steps = 100, lr =0.01)


From thereon, it is as easy as it can be. Users can perform any type of post-analysis using the estimated parameters, or
return model specific outputs like topic-word intensities or document-topic proportions.

.. code-block:: python

    # Analyze the results
    topics, e_theta = tm.return_topics()
    betas = tm.return_beta()

To check model convergence, we advice so have a closer look at the convergence of the training loss, i.e.


.. code-block:: python

    tm.Metrics.loss


What models are supported in PyPF
---------------------------------

PyPF is engineered with high modularity to integrate various text mining techniques. 
As of today, it features a rich collection of Poisson factorization based text mining models and already supports:

+ PF: Basic Poisson factorization topic modeling (also a version including covariates, named 'CPF')
+ SPF `Seeded Poisson factorization <https://arxiv.org/abs/2503.02741>`_ (also a version including covariates, named 'CSPF')
+ TBIP: `Text-based ideal points <https://aclanthology.org/2020.acl-main.475/>`_
+ STBS: `Structual text-based scaling model <https://arxiv.org/abs/2410.11897>`_


