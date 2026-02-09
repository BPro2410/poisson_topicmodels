.. _intro_poisson_factorization:

Poisson Factorization
=====================

Since all the featured models in topicmodels are based on Poisson Factorization (PF), we give a short introduction into PF.

Poisson factorization (PF) is a powerful statistical tool used for modeling count data, and it has garnered significant attention in various fields due to its flexibility and robustness. At its core, Poisson factorization leverages the Poisson distribution's capacity to model the frequency of events occurring within a fixed interval of time or space. The Poisson distribution expresses the probability of a given number of events happening in a fixed interval of time or space and is defined as:

.. math::

    P(X = k) = \frac{\lambda^k  e^{-\lambda}}{k!}

where :math:`\lambda` is the average number of occurrences within the interval, :math:`k` is the number of occurrences, :math:`e` is the base of the natural logarithm, and :math:`k!` is the factorial of :math:`k`.

Poisson factorization extends the use of the Poisson distribution by modeling a matrix of count data, often referred to as the frequency data matrix. This matrix can represent various types of data, such as word frequencies in documents or user activity in online platforms.

Matrix Decomposition
--------------------

Given a matrix :math:`Y` of dimensions :math:`M \times N`, where :math:`Y_{ij}` represents the count of events for the :math:`i`-th row and :math:`j`-th column, Poisson factorization aims to decompose this matrix into two lower-dimensional matrices :math:`W` and :math:`H`:

.. math::

    Y \approx WH

Here, :math:`W` is an :math:`M \times K` matrix and :math:`H` is a :math:`K \times N` matrix, where :math:`K` is the latent dimension. The decomposition is typically performed such that:

.. math::

     Y_{ij} \sim \text{Poisson}(W_i H_j)

Advantages of Poisson Factorization
-----------------------------------

PF offers several benefits that make it a preferred choice for numerous applications:

1. **Scalability:** It efficiently handles large-scale datasets, making it suitable for big data applications.
2. **Sparsity Handling:** It naturally accommodates sparse data, which is common in real-world scenarios like text and social network data.
3. **Interpretability:** The factors :math:`W` and :math:`H` can often be interpreted meaningfully, providing insights into the underlying patterns in the data.
4. **Extensibility:** It can be extended to incorporate additional information, enhancing modeling capabilities.

Application in Text Mining
--------------------------

Text mining is a crucial task in text analysis, aiming to uncover latent factors within a collection of documents. Poisson factorization has proven to be particularly effective for this purpose due to its ability to model word frequencies accurately. In text mining, the count matrix :math:`Y` represents the frequency of words in documents. Poisson factorization helps in identifying the latent topics by decomposing this matrix into topic and word distributions.

For example, if :math:`Y` is a :math:`D \times V` matrix (where :math:`D` is the number of documents and :math:`V` is the number of unique words), PF decomposes it into non-negative latent matrices :math:`\theta` (document-topic distribution) and :math:`\beta` (topic-word distribution):

.. math::

    Y_{dv} \sim \text{Poisson}\left(\sum_{k = 1}^{K}\theta_{dk} \beta_{kv}\right)

Conclusion
----------

Poisson factorization is a versatile and robust tool for modeling count data, with significant advantages for various applications. Its ability to handle large, sparse datasets while providing meaningful decompositions makes it particularly suitable for topic modeling tasks in text classification. By leveraging Poisson factorization, researchers and practitioners can uncover latent topics in document collections, enhancing the understanding and organization of textual data.
