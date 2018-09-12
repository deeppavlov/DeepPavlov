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

Config
======

Default ranker config for **English** language is
:config:`ranking/en_ranker_tfidf_wiki.json <ranking/en_ranker_tfidf_wiki.json>`

Default ranker config for **Russian** language is
:config:`ranking/ru_ranker_tfidf_wiki.json <ranking/ru_ranker_tfidf_wiki.json>`

Config Structure
----------------

-  **dataset_iterator** - downloads Wikipidia DB, creates batches for
   ranker fitting

   -  **data_dir** - a directory to download DB to
   -  **data_url** - an URL to download Wikipedia DB from
   -  **shuffle** - whether to perform shuffling when iterating over DB or not

-  **chainer** - pipeline manager

   -  **in** - pipeline input data (questions)
   -  **out** - pipeline output data (Wikipedia articles ids)

-  **tfidf_ranker** - the ranker class

   -  **top_n** - a number of document to return (when n=1 the most
      relevant document is returned)
   -  **in** - ranker input data (queries)
   -  **out** - ranker output data (Wikipedia articles ids)
   -  **fit_on_batch** - pass method to a vectorizer
   -  **vectorizer** - a vectorizer class

      -  **fit_on_batch** - fit the vectorizer on batches of Wikipedia articles
      -  **save_path** - a path to serialize a vectorizer to
      -  **load_path** - a path to load a vectorizer from
      -  **tokenizer** - a tokenizer class

         -  **lemmas** - whether to lemmatize tokens or not
         -  **ngram_range** - ngram range for **vectorizer** features

-  **train** - parameters for vectorizer fitting

   -  **validate_best**- is ingnored, any value
   -  **test_best** - is ignored, any value
   -  **batch_size** - how many Wikipedia articles should return
      the **dataset_iterator** in a single batch

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
    python deep.py train deeppavlov/configs/ranking/en_ranker_tfidf_wiki.json

Run the following to fit the ranker on **Russian** Wikipedia:

.. code:: bash

    cd deeppavlov/
    python deep.py train deeppavlov/configs/ranking/ru_ranker_tfidf_wiki.json

Interacting
-----------

When interacting, the ranker returns document titles of the relevant
documents.

Run the following to interact with the **English** ranker:

.. code:: bash

    cd deeppavlov/
    python deep.py interact deeppavlov/configs/ranking/en_ranker_tfidf_wiki.json -d

Run the following to interact with the **Russian** ranker:

.. code:: bash

    cd deeppavlov/
    python deep.py interact deeppavlov/configs/ranking/ru_ranker_tfidf_wiki.json -d

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
#. Build a database with the help of `DrQA
   script <https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/build_db.py>`__.

enwiki_tfidf_matrix.npz
-------------------------

**enwiki_tfidf_matrix.npz** is a full Wikipedia tf-idf matrix of
size **hash_size x number of documents** which is
**2**24 x 5159530**. This matrix is built with
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
#. Build a database with the help of `DrQA
   script <https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/build_db.py>`__.

ruwiki_tfidf_matrix.npz
-------------------------

**ruwiki_tfidf_matrix.npz** is a full Wikipedia tf-idf matrix of
size **hash_size x number of documents** which is
**2**24 x 1463888**. This matrix is built with
:class:`~deeppavlov.models.vectorizers.hashing_tfidf_vectorizer.HashingTfIdfVectorizer` class.
class.

Comparison
==========

Scores for **TF-IDF Ranker** model:


+-------------------------------------------------------+----------------+----------------------+-----------------+
| Model                                                 | Dataset        |  Wiki dump           |  Recall (top 5) |
+-------------------------------------------------------+----------------+----------------------+-----------------+
| :config:`DeepPavlov <odqa/en_ranker_tfidf_wiki.json>` | SQuAD (dev)    |  enwiki (2018-02-11) |       75.6      |
+-------------------------------------------------------+----------------+----------------------+-----------------+
| `DrQA`_                                               | SQuAD (dev)    |  enwiki (2016-12-21) |       77.8      |
+-------------------------------------------------------+----------------+----------------------+-----------------+


References
==========

.. target-notes::

.. _`DrQA`: https://github.com/facebookresearch/DrQA/
.. _`WikiExtractor`: https://github.com/attardi/wikiextractor

