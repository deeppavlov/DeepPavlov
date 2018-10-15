=============
TF-IDF Ranker
=============

This is an implementation of a document ranker based on tf-idf vectorization.
The ranker implementation is based on `DrQA`_ project.
The default ranker implementation takes a batch of queries as input and returns 5 document ids as output.

::

    :: Who is Ivan Pavlov?
    >> ['Ivan Pavlov (lawyer)', 'Ivan Pavlov', 'Pavlovian session', 'Ivan Pavlov (film)', 'Vladimir Bekhterev']

Text for the output ids can be further extracted with :class:`~deeppavlov.vocabs.wiki_sqlite.WikiSQLiteVocab` class.

Configuration
=============

Default ranker config for **English** language is
:config:`doc_retrieval/en_ranker_tfidf_wiki.json <doc_retrieval/en_ranker_tfidf_wiki.json>`

Default ranker config for **Russian** language is
:config:`doc_retrieval/ru_ranker_tfidf_wiki.json <doc_retrieval/ru_ranker_tfidf_wiki.json>`

Running the Ranker
==================

.. note::

    Training and inferring the ranker requires ~16 GB RAM.

.. _ranker_training:

Training
--------

Run the following to fit the ranker on **English** Wikipedia:

.. code:: bash

    cd deeppavlov/
    python deep.py train configs/doc_retrieval/en_ranker_tfidf_wiki.json

Run the following to fit the ranker on **Russian** Wikipedia:

.. code:: bash

    cd deeppavlov/
    python deep.py train configs/doc_retrieval/ru_ranker_tfidf_wiki.json

Interacting
-----------

When interacting, the ranker returns document titles of the relevant
documents.

Run the following to interact with the **English** ranker:

.. code:: bash

    cd deeppavlov/
    python deep.py interact configs/doc_retrieval/en_ranker_tfidf_wiki.json -d

Run the following to interact with the **Russian** ranker:

.. code:: bash

    cd deeppavlov/
    python deep.py interact configs/doc_retrieval/ru_ranker_tfidf_wiki.json -d

As a result of ranker training, a SQLite database and tf-idf matrix are created.

Available Data and Pretrained Models
====================================

Wikipedia DB and pretrained tfidf matrices are downloaded in
``deeppavlov/download/odqa`` folder by default.

enwiki.db
---------

**enwiki.db** SQLite database consists of **5159530** Wikipedia articles
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


+-----------------------------------------------------------------+----------------+----------------------+-----------------+
| Model                                                           | Dataset        |  Wiki dump           |  Recall (top 5) |
+-----------------------------------------------------------------+----------------+----------------------+-----------------+
| :config:`DeepPavlov <doc_retrieval/en_ranker_tfidf_wiki.json>`  | SQuAD (dev)    |  enwiki (2018-02-11) |       75.6      |
+-----------------------------------------------------------------+----------------+----------------------+-----------------+
| `DrQA`_                                                         | SQuAD (dev)    |  enwiki (2016-12-21) |       77.8      |
+-----------------------------------------------------------------+----------------+----------------------+-----------------+


References
==========

.. target-notes::

.. _`DrQA`: https://github.com/facebookresearch/DrQA/
.. _`WikiExtractor`: https://github.com/attardi/wikiextractor

.. |2**24| replace:: 2\ :sup:`24`

