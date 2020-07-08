=============
TF-IDF Ranker
=============

This is an implementation of a document ranker based on tf-idf vectorization.
The ranker implementation is based on `DrQA`_ project.
The default ranker implementation takes a batch of queries as input and returns 25 document titles sorted via relevance.

Quick Start
===========

Before using the model make sure that all required packages are installed running the command:

.. code:: bash

    python -m deeppavlov install en_ranker_tfidf_wiki

Training and building (if you have your own data)

.. code:: python

    from deeppavlov import configs, train_model
    ranker = train_model(configs.doc_retrieval.en_ranker_tfidf_wiki, download=True)

Building (if you don't have your own data)

.. code:: python

    from deeppavlov import build_model, configs

    ranker = build_model(configs.doc_retrieval.en_ranker_tfidf_wiki, download=True)

Inference

.. code:: python

    result = ranker(['Who is Ivan Pavlov?'])
    print(result[:5])

Output

::

    >> ['Ivan Pavlov (lawyer)', 'Ivan Pavlov', 'Pavlovian session', 'Ivan Pavlov (film)', 'Vladimir Bekhterev']

Text for the output titles can be further extracted with :class:`~deeppavlov.vocabs.wiki_sqlite.WikiSQLiteVocab` class.


Configuration
=============

Default ranker config for **English** language is
:config:`doc_retrieval/en_ranker_tfidf_wiki.json <doc_retrieval/en_ranker_tfidf_wiki.json>`

Default ranker config for **Russian** language is
:config:`doc_retrieval/ru_ranker_tfidf_wiki.json <doc_retrieval/ru_ranker_tfidf_wiki.json>`

Running the Ranker
==================

.. note::

    About **16 GB of RAM** required.

.. _ranker_training:

Training
--------

Run the following to fit the ranker on **English** Wikipedia:

.. code:: bash

    python -m deppavlov train en_ranker_tfidf_wiki

Run the following to fit the ranker on **Russian** Wikipedia:

.. code:: bash

    python -m deeppavlov train ru_ranker_tfidf_wiki

As a result of ranker training, a SQLite database and tf-idf matrix are created.

Interacting
-----------

When interacting, the ranker returns document titles of the relevant
documents.

Run the following to interact with the **English** ranker:

.. code:: bash

    python -m deeppavlov interact en_ranker_tfidf_wiki -d

Run the following to interact with the **Russian** ranker:

.. code:: bash

    python -m deeppavlov ru_ranker_tfidf_wiki -d

Available Data and Pretrained Models
====================================

Wikipedia DB is downloaded to ``~/.deeppavlov/downloads/odqa`` and pre-trained tfidf matrices are downloaded
to ``~/.deeppavlov/models/odqa`` folder by default.

enwiki.db
---------

**enwiki.db** SQLite database consists of **5180368** Wikipedia articles
and is built by the following steps:

#. Download a Wikipedia dump file. We took the latest
   `enwiki dump <https://dumps.wikimedia.org/enwiki/20180201>`__
   (from 2018-02-11)
#. Unpack and extract the articles with `WikiExtractor`_
   (with ``--json``, ``--no-templates``, ``--filter_disambig_pages``
   options)
#. Build a database during :ref:`ranker_training`.

enwiki_tfidf_matrix.npz
-------------------------

**enwiki_tfidf_matrix.npz** is a full Wikipedia tf-idf matrix of
size **hash_size x number of documents** which is
|2**24| x 5180368. This matrix is built with
:class:`~deeppavlov.models.vectorizers.hashing_tfidf_vectorizer.HashingTfIdfVectorizer` class.

ruwiki.db
---------

**ruwiki.db** SQLite database consists of **1463888 Wikipedia
articles**
and is built by the following steps:

#. Download a Wikipedia dump file. We took the latest
   `ruwiki dump <https://dumps.wikimedia.org/ruwiki/20180401>`__
   (from 2018-04-01)
#. Unpack and extract the articles with
   `WikiExtractor <https://github.com/attardi/wikiextractor>`__
   (with ``--json``, ``--no-templates``, ``--filter_disambig_pages``
   options)
#. Build a database during :ref:`ranker_training`.

ruwiki_tfidf_matrix.npz
-------------------------

**ruwiki_tfidf_matrix.npz** is a full Wikipedia tf-idf matrix of
size **hash_size x number of documents** which is
|2**24| x 1463888. This matrix is built with
:class:`~deeppavlov.models.vectorizers.hashing_tfidf_vectorizer.HashingTfIdfVectorizer` class.
class.

Comparison
==========

Scores for **TF-IDF Ranker** model:

+------------------------------------------------------------------------------+----------------+-----------------+
| Model                                                                        | Dataset        |   Recall@5      |
+------------------------------------------------------------------------------+----------------+-----------------+
| :config:`enwiki20180211 <doc_retrieval/en_ranker_tfidf_wiki.json>`           |                |       75.6      |
+------------------------------------------------------------------------------+                +-----------------+
| :config:`enwiki20161221 <doc_retrieval/en_ranker_tfidf_enwiki20161221.json>` |  SQuAD (dev)   |       76.2      |
+------------------------------------------------------------------------------+                +-----------------+
| `DrQA`_ enwiki20161221                                                       |                |       77.8      |
+------------------------------------------------------------------------------+----------------+-----------------+


References
==========

.. target-notes::

.. _`DrQA`: https://github.com/facebookresearch/DrQA/
.. _`WikiExtractor`: https://github.com/attardi/wikiextractor

.. |2**24| replace:: 2\ :sup:`24`

