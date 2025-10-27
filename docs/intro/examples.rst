========
Examples
========

This section provides a minimal example of all the models featured in PyPF.

First we will create a corpus that will serve as the counts matrix for all subsequent analysis and import the PyPF package.


.. code-block:: python

    # Imports
    from PyPF import topicmodels
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    import scipy.sparse as sparse

    # Example documents
    documents = [
        "My smartphone's battery life is fantastic, lasts all day!",
        "The camera on my phone is incredible, takes crystal-clear photos.",
        "Love the smooth performance, but it overheats with heavy apps.",
        "This phone charges super fast, very convenient.",
        "The display is bright and vibrant, but scratches easily.",
        "Great phone, but the speaker volume is too low.",
        "My phone lags sometimes, but overall it's decent.",
        "Face unlock works instantly, super happy with it!",
        "Wish my phone had expandable storage, but still a great buy.",
        "The new update slowed my phone down, very frustrating.",
        "My laptop is super fast, perfect for multitasking.",
        "Love my PC’s performance, but the fan noise is annoying.",
        "This computer runs all my programs smoothly, no complaints!",
        "The keyboard feels great, but the battery drains fast.",
        "Amazing display, perfect for video editing!",
        "My laptop is lightweight and easy to carry everywhere.",
        "Upgraded my PC and it boots in seconds!",
        "Great for gaming, but gets hot after long sessions.",
        "My computer sometimes freezes, but a restart fixes it.",
        "Best laptop I’ve owned, powerful and reliable!"
    ]

    cv = CountVectorizer(stop_words='english', min_df = 1)
    counts = sparse.csr_matrix(cv.fit_transform(documents), dtype = np.float32)



PF: Poisson factorization
-------------------------

We first fit a Poisson factorization topic model for 2 topics.

.. code-block:: python

    # Initialize the model
    tm1 = topicmodels("PF", counts, vocab, num_topics = 2, batch_size = 1024)

    # Train the model
    estimated_params = tm1.train_step(num_steps = 100, lr =0.01)

    # Analyze results
    print(estimated_params) # Check all learned parameters for custom analysis
    print(tm1.Metrics.loss) # Check model convergence
    topics, e_theta = tm1.return_topics() # See topic distribution across the documents
    betas = tm1.return_beta() # See topic distribution across the vocabulary



CPF: Poisson factorization with covariables
-------------------------------------------

To fit covariate effects in the PF topic model, we first have to simulate covariates on document level.

.. code-block:: python

    # Create dummy effects
    covs = [1]*11 + [0]*9

    # Create a design matrix
    X_design_matrix = pd.DataFrame({'intercept' : np.repeat(1, len(documents)), 'var_infromative' : covs})


Next we fit the PF topic model with covariates


.. code-block:: python

    # Initialize the model
    tm2 = topicmodels("CPF", counts, vocab, num_topics = 2, batch_size = 1024, X_design_matrix = X_design_matrix)
    
    # Train the model
    estimated_params = tm2.train_step(num_steps = 100, lr = 0.01)

    # Analyze results
    print(estimated_params) # Check all learned parameters for custom analysis
    print(tm2.Metrics.loss) # Check model convergence
    topics, e_theta = tm2.return_topics() # See topic distribution across the documents
    betas = tm2.return_beta() # See topic distribution across the vocabulary
    cov_effects = tm2.return_covariate_effects() # Return covariate effects



SPF: Seeded Poisson factorization
---------------------------------

For the Seeded Poisson factorization (SPF) model, we are able to estimate guided topics.
Therefore, we define two topics a-priori.

.. code-block:: python

    # Define topic-specific seed words
    smartphone = {"smartphone", "iphone", "phone", "touch", "app"}
    pc = {"laptop", "keyboard", "desktop", "pc"}

    keywords = {"smartphone": smartphone, "pc": pc}

Now we can fit the topic model with the a-priori specified topics 'smartphone' and 'pc'.

.. code-block:: python
    
    # Initialize the model
    tm3 = topicmodels("SPF", counts, vocab, keywords, residual_topics = 0, batch_size = 1024)

    # Train the model
    estimated_params = tm3.train_step(num_steps = 100, lr = 0.01)

    # Analyze results
    print(estimated_params) # Check all learned parameters for custom analysis
    print(tm3.Metrics.loss) # Check model convergence
    topics, e_theta = tm2.return_topics() # See topic distribution across the documents
    betas = tm2.return_beta() # See topic distribution across the vocabulary
    



CSPF: Seeded Poisson factorization with covariables
---------------------------------------------------

For the SPF topic model including covariates, we just follow the process and include the simulated metainformation from the CPF model.

.. code-block:: python

    # Initialize the model
    tm4 = topicmodels("CSPF", counts, vocab, keywords, residual_topics = 0, batch_size = 1024, X_design_matrix = X_design_matrix)

    # Train the model
    estimated_params = tm4.train_step(num_steps = 100, lr = 0.01)

    # Analyze results
    print(estimated_params) # Check all learned parameters for custom analysis
    print(tm4.Metrics.loss) # Check model convergence
    topics, e_theta = tm2.return_topics() # See topic distribution across the documents
    betas = tm2.return_beta() # See topic distribution across the vocabulary
    cov_effects = tm2.return_covariate_effects() # Return covariate effects



TBIP: Text-based ideal points
-----------------------------

The text-based ideal point model requires additional information regarding authority upon initlialization.
To give a minimum example we also simulate artificial authors.


.. code-block:: python

    speaker = np.random.choice(['A', 'B', 'C'], size=len(documents), replace=True)

Next we follow the PyPF logic and initialize, train and analyze the TBIP model.

.. code-block:: python
    
    # Initialize the model
    tm5 = topicmodels("TBIP", counts, vocab, num_topics = 2, authors = speaker, batch_size = 1024)

    # Train the model
    estimated_params = tm5.train_step(num_steps = 1000, lr = 0.01)

    # Analyze results
    print(estimated_params) # Check all learned parameters for custom analysis
    print(tm5.Metrics.loss) # Check model convergence
    topics, e_theta = tm2.return_topics() # See topic distribution across the documents
    betas = tm2.return_beta() # See topic distribution across the vocabulary



TVTBIP: Time-variyng text-based ideal point model
-------------------------------------------------

Implemented but minimal example to be added soon.


ETM: Embedded Topic Model
-------------------------

tbd.


SBTS: Structual text-based scaling model
----------------------------------------

tbd.