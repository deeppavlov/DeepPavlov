=================
Popularity Ranker
=================

Popularity Ranker re-ranks results obtained via :doc:`TF-IDF Ranker <tfidf_ranking>` using information about
the number of article views. The number of Wikipedia articles views is an open piece of information which can be
obtained via `Wikimedia REST API <https://wikimedia.org/api/rest_v1/>`_.
We assigned a mean number of views for the period since 2017/11/05 to 2018/11/05 to each article in our
English Wikipedia database `enwiki20180211 <http://files.deeppavlov.ai/datasets/wikipedia/enwiki.tar.gz>`_.

The inner algorithm of Popularity Ranker is a Logistic Regression classifier based on 3 features:

- tfidf score of the article
- popularity of the article
- multiplication of two above features

The classifier is trained on `SQuAD-v1.1`_ train set.

Quick Start
===========

Before using the model make sure that all required packages are installed running the command:

.. code:: bash

    python -m deeppavlov install en_ranker_pop_wiki

Building the model

.. code:: python

    from deeppavlov import build_model

    ranker = build_model('en_ranker_pop_wiki', download=True)

Inference

.. code:: python

    result = ranker(['Who is Ivan Pavlov?'])
    print(result[:5])

Output

::

    >> ['Ivan Pavlov', 'Vladimir Bekhterev', 'Classical conditioning', 'Valentin Pavlov', 'Psychology']

Text for the output titles can be further extracted with :class:`~deeppavlov.vocabs.wiki_sqlite.WikiSQLiteVocab` class.


Configuration
=============

Default ranker config is
:config:`doc_retrieval/en_ranker_pop_wiki.json <doc_retrieval/en_ranker_pop_wiki.json>`

Running the Ranker
==================

.. note::

    About **17 GB of RAM** required.

Interacting
-----------

When interacting, the ranker returns document titles of the relevant
documents.

Run the following to interact with the ranker:

.. code:: bash

    python -m deeppavlov interact en_ranker_pop_wiki -d


Available Data and Pretrained Models
====================================

Available information about Wikipedia articles popularity is downloaded to ``~/.deeppavlov/downloads/odqa/popularities.json``
and pre-trained logistic regression classifier is downloaded to ``~/.deeppavlov/models/odqa/logreg_3features.joblib`` by default.


References
==========

.. target-notes::

.. _`SQuAD-v1.1`: https://arxiv.org/abs/1606.05250
