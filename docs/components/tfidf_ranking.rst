TFIDF ranker
============

This is an implementation of a document ranker based on tfidf
vectorization. The ranker implementation
is based on `DrQA <https://github.com/facebookresearch/DrQA>`__
project.

Config
~~~~~~

Default ranker config for **English** language is
``deeppavlov/configs/ranking/en_ranker_tfidf_wiki.json``

Default ranker config for **Russian** language is
``deeppavlov/configs/ranking/ru_ranker_tfidf_wiki.json``

The ranker config for **English** language can be found at
``deeppavlov/configs/odqa/en_ranker_prod.json``

The ranker config for **Russian** language can be found at
``deeppavlov/configs/odqa/ru_ranker_prod.json``

-  **dataset\_iterator** - downloads Wikipidia DB, creates batches for
   ranker fitting

   -  ***data*\ dir\_** - a directory to download DB to
   -  ***data*\ url\_** - an URL to download Wikipedia DB from
   -  ***shuffle*** - whether to perform shuffling when iterating over
      DB or not

-  **chainer** - pipeline manager

   -  ***in*** - pipeline input data (questions)
   -  ***out*** - pipeline output data (Wikipedia articles ids and
      scores of the articles)

-  **tfidf\_ranker** - the ranker class

   -  **top\_n** - a number of document to return (when n=1 the most
      relevant document is returned)
   -  ***in*** - ranker input data (questions)
   -  ***out*** - ranker output data (Wikipedia articles ids)
   -  ***fit*\ on\_batch\_** - fit the ranker on batches of Wikipedia
      articles
   -  ***vectorizer*** - a vectorizer class

      -  ***fit*\ on\_batch\_** - fit the vectorizer on batches of
         Wikipedia articles
      -  ***save*\ path\_** - a path to serialize a vectorizer to
      -  ***load*\ path\_** - a path to load a vectorizer from
      -  ***tokenizer*** - a tokenizer class

         -  ***lemmas*** - whether to lemmatize tokens or not
         -  ***ngram*\ range\_** - ngram range for vectorizer features

-  **train** - parameters for vectorizer fitting

   -  ***validate*\ best\_**- is ingnored, any value
   -  ***test*\ best\_** - is ignored, any value
   -  ***batch*\ size\_** - how many Wikipedia articles should return
      the dataset iterator in a single batch

Running the ranker
------------------

**Training and infering the rannker on English Wikipedia requires 16 GB
RAM**

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

When interacted the ranker returns document titles of the relevant
documents.

Run the following to interact the **English** ranker:

.. code:: bash

    cd deeppavlov/
    python deep.py interact deeppavlov/configs/ranking/en_ranker_tfidf_wiki.json -d

Run the following to interact the **Russian** ranker:

.. code:: bash

    cd deeppavlov/
    python deep.py interact deeppavlov/configs/ranking/ru_ranker_tfidf_wiki.json -d

Pretrained models
-----------------

Wikipedia DB and pretrained tfidf matrices are downloaded in
``deeppavlov/download/odqa`` by default.

enwiki.db
~~~~~~~~~

**enwiki.db** SQLite database consists of **5159530 Wikipedia
articles**
and is built by the following steps:

#. Download a Wikipedia dump file. We took the latest
   `enwiki <https://dumps.wikimedia.org/enwiki/20180201>`__
   (from 2018-02-11)
#. Unpack and extract the articles with
   `WikiExtractor <https://github.com/attardi/wikiextractor>`__
   (with ``--json``, ``--no-templates``, ``--filter_disambig_pages``
   options)
#. Build a database with the help of `DrQA
   script <https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/build_db.py>`__.

enwiki\_tfidf\_matrix.npz
~~~~~~~~~~~~~~~~~~~~~~~~~

**enwiki\_tfidf\_matrix.npz** is a full Wikipedia tf-idf matrix of
size ``hash_size x number of documents`` which is
``2**24 x 5159530``. This matrix is built with
``deeppavlov/models/vectorizers/hashing_tfidf_vectorizer.HashingTfidfVectorizer``
class.

ruwiki.db
~~~~~~~~~

**ruwiki.db** SQLite database consists of **1463888 Wikipedia
articles**
and is built by the following steps:

#. Download a Wikipedia dump file. We took the latest
   `ruwiki <https://dumps.wikimedia.org/ruwiki/20180401>`__
   (from 2018-04-01)
#. Unpack and extract the articles with
   `WikiExtractor <https://github.com/attardi/wikiextractor>`__
   (with ``--json``, ``--no-templates``, ``--filter_disambig_pages``
   options)
#. Build a database with the help of `DrQA
   script <https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/build_db.py>`__.

ruwiki\_tfidf\_matrix.npz
~~~~~~~~~~~~~~~~~~~~~~~~~

**ruwiki\_tfidf\_matrix.npz** is a full Wikipedia tf-idf matrix of
size ``hash_size x number of documents`` which is
``2**24 x 1463888``. This matrix is built with
``deeppavlov/models/vectorizers/hashing_tfidf_vectorizer.HashingTfidfVectorizer``
class.

References
----------

#. https://github.com/facebookresearch/DrQA

