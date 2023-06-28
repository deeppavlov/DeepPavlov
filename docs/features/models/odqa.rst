=================================================
Open Domain Question Answering Model on Wikipedia
=================================================

Task definition
===============

**Open Domain Question Answering (ODQA)** is a task to find an exact answer
to any question in **Wikipedia** articles. Thus, given only a question, the system outputs
the best answer it can find.
The default ODQA implementation takes a batch of queries as input and returns the best answer.

Quick Start
===========

The example below is given for basic ODQA config :config:`en_odqa_infer_wiki <odqa/en_odqa_infer_wiki.json>`.
Check what :ref:`other ODQA configs <odqa_configuration>` are available and simply replace `en_odqa_infer_wiki`
with the config name of your preference.

Before using the model make sure that all required packages are installed running the command:

.. code:: bash

    python -m deeppavlov install en_odqa_infer_wiki

Training (if you have your own data)

.. code:: python

    from deeppavlov import train_evaluate_model_from_config

    train_evaluate_model_from_config('en_ranker_tfidf_wiki', download=True)
    train_evaluate_model_from_config('qa_squad2_bert', download=True)

Building

.. code:: python

    from deeppavlov import build_model

    odqa = build_model('en_odqa_infer_wiki', download=True)

Inference

.. code:: python

    result = odqa(['What is the name of Darth Vader\'s son?'])
    print(result)

Output:

::

    >> Luke Skywalker


Languages
=========

There are pretrained **ODQA** models for **English** and **Russian**
languages in :doc:`DeepPavlov </index/>`.

Models
======

English ODQA version consists of the following components:

- TF-IDF ranker (based on `DrQA`_), which defines top-N most relevant paragraphs in TF-IDF index;
- `Binary Passage Retrieval`_ (BPR) ranker, which defines top-K most relevant in binary index;
- a database of paragraphs (by default, from Wikipedia) which finds N + K most relevant paragraph text by IDs, defined by TF-IDF and BPR ranker;
- Reading Comprehension component, which finds answers in paragraphs and defines answer confidences.

Russian ODQA version performs retrieval only with TF-IDF index.

Binary Passage Retrieval is resource-efficient the method of building a dense passage index. The dual encoder (with BERT or other Tranformer as backbone) is trained on question answering dataset (Natural Questions in our case) to maximize dot product of question and passage with answer embeddings and minimize otherwise. The question or passage embeddings are obtained the following way: vector of BERT CLS-token is fed into a dense layer followed by a hash function which turns dense vector into binary one.

Running ODQA
============

.. note::

    About **22 GB of RAM** required.
    It is possible to run on a 16 GB machine, but than swap size should be at least 8 GB.

Training
--------

**ODQA ranker** and **ODQA reader** should be trained separately.
Read about training the **ranker** :ref:`here <ranker_training>`.
Read about training the **reader** in our separate [reader tutorial]<SQuAD.ipynb#4.-Train-the-model-on-your-data>.

Interacting
-----------

When interacting, the **ODQA** model returns a plain answer to the user's
question.

Run the following to interact with **English ODQA**:

.. code:: bash

    python -m deeppavlov interact en_odqa_infer_wiki -d

Run the following to interact with **Russian ODQA**:

.. code:: bash

    python -m deeppavlov interact ru_odqa_infer_wiki -d

Configuration
=============

.. _odqa_configuration:

The **ODQA** configs suit only model inferring purposes. For training purposes use
the :ref:`ranker configs <ranker_training>` and the [reader tutorial]<SQuAD.ipynb#4.-Train-the-model-on-your-data>
accordingly.

There are several ODQA configs available:

+----------------------------------------------------------------------------------------+-------------------------------------------------+
|                                                                                        |                                                 |
|                                                                                        |                                                 |
| Config                                                                                 | Description                                     |
+----------------------------------------------------------------------------------------+-------------------------------------------------+
|:config:`en_odqa_infer_wiki <odqa/en_odqa_infer_wiki.json>`                             | Basic config for **English** language. Consists |
|                                                                                        | of of Binary Passage Retrieval, TF-IDF          |
|                                                                                        | retrieval and reader.                           |
+----------------------------------------------------------------------------------------+-------------------------------------------------+
|:config:`ru_odqa_infer_wiki <odqa/ru_odqa_infer_wiki.json>`                             | Basic config for **Russian** language. Consists |
|                                                                                        | of TF-IDF ranker and reader.                    |
+----------------------------------------------------------------------------------------+-------------------------------------------------+
|:config:`en_odqa_pop_infer_wiki <odqa/en_odqa_pop_infer_wiki.json>`                     | Extended config for **English** language.       |
|                                                                                        | Consists of of Binary Passage Retrieval, TF-IDF |
|                                                                                        | retrieval, popularity ranker and reader.        |
+----------------------------------------------------------------------------------------+-------------------------------------------------+

Comparison
==========

Scores for **ODQA** models:

+-----------------------------------------------------------------------+------+----------------------+----------------------+------+------+------+
| Model                                                                 | Lang |       Dataset        | Number of paragraphs |  F1  |  EM  | RAM  |
+-----------------------------------------------------------------------+------+----------------------+----------------------+------+------+------+
|:config:`DeppPavlov <odqa/en_odqa_infer_wiki.json>`                    |  En  |  Natural Questions   |         200          | 41.7 | 33.8 | 10.4 |
+-----------------------------------------------------------------------+      |                      +----------------------+------+------+------+
|:config:`DeppPavlov <odqa/en_odqa_pop_infer_wiki.json>`                |      |                      |         200          | 41.7 | 33.8 | 10.4 |
+-----------------------------------------------------------------------+      +                      +----------------------+------+------+------+
|`DPR`_                                                                 |      |                      |         100          |  -   | 41.5 | 64.6 |
+-----------------------------------------------------------------------+------+----------------------+----------------------+------+------+------+
|:config:`DeepPavlov with RuBERT reader <odqa/ru_odqa_infer_wiki.json>` |  Ru  |  SDSJ Task B (dev)   |         100          | 58.9 | 42.6 | 13.1 |  
+-----------------------------------------------------------------------+------+----------------------+----------------------+------+------+------+

EM stands for "exact-match accuracy". Metrics are counted for top 100 and top 200 paragraphs, extracted by retrieval module.

References
==========

.. target-notes::

.. _`DrQA`: https://github.com/facebookresearch/DrQA/
.. _`Binary Passage Retrieval`: https://arxiv.org/abs/2106.00882
.. _`DPR`: https://arxiv.org/abs/2004.04906


