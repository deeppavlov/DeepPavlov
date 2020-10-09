=================================================
Open Domain Question Answering Skill on Wikipedia
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

    from deeppavlov import configs, train_evaluate_model_from_config

    train_evaluate_model_from_config(configs.doc_retrieval.en_ranker_tfidf_wiki, download=True)
    train_evaluate_model_from_config(configs.squad.multi_squad_noans, download=True)

Building

.. code:: python

    from deeppavlov import build_model, configs

    odqa = build_model(configs.odqa.en_odqa_infer_wiki, download=True)

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

The architecture of **ODQA** skill is modular and consists of two models,
a **ranker** and a **reader**. The **ranker** is based on `DrQA`_ proposed by Facebook Research
and the **reader** is based on `R-NET`_ proposed by Microsoft Research Asia
and its `implementation`_ by Wenxuan Zhou.

Running ODQA
============

.. note::

    About **24 GB of RAM** required.
    It is possible to run on a 16 GB machine, but than swap size should be at least 8 GB.

Training
--------

**ODQA ranker** and **ODQA reader** should be trained separately.
Read about training the **ranker** :ref:`here <ranker_training>`.
Read about training the **reader** in our separate :ref:`reader tutorial <reader_training>`.

Interacting
-----------

When interacting, the **ODQA** skill returns a plain answer to the user's
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
the :ref:`ranker configs <ranker_training>` and the :ref:`reader configs <reader_training>`
accordingly.

There are several ODQA configs available:

+----------------------------------------------------------------------------------------+-------------------------------------------------+
|                                                                                        |                                                 |
|                                                                                        |                                                 |
| Config                                                                                 | Description                                     |
+----------------------------------------------------------------------------------------+-------------------------------------------------+
|:config:`en_odqa_infer_wiki <odqa/en_odqa_infer_wiki.json>`                             | Basic config for **English** language. Consists |
|                                                                                        | of TF-IDF ranker and reader. Searches for an    |
|                                                                                        | answer in ``enwiki20180211`` Wikipedia dump.    |
+----------------------------------------------------------------------------------------+-------------------------------------------------+
|:config:`en_odqa_infer_enwiki20161221 <odqa/en_odqa_infer_enwiki20161221.json>`         | Basic config for **English** language. Consists |
|                                                                                        | of TF-IDF ranker and reader. Searches for an    |
|                                                                                        | answer in ``enwiki20161221`` Wikipedia dump.    |
+----------------------------------------------------------------------------------------+-------------------------------------------------+
|:config:`ru_odqa_infer_wiki <odqa/ru_odqa_infer_wiki.json>`                             | Basic config for **Russian** language. Consists |
|                                                                                        | of TF-IDF ranker and reader. Searches for an    |
|                                                                                        | answer in ``ruwiki20180401`` Wikipedia dump.    |
+----------------------------------------------------------------------------------------+-------------------------------------------------+
|:config:`en_odqa_pop_infer_enwiki20180211 <odqa/en_odqa_pop_infer_enwiki20180211.json>` | Extended config for **English** language.       |
|                                                                                        | Consists of TF-IDF Ranker, Popularity Ranker    |
|                                                                                        | and reader. Searches for an answer in           |
|                                                                                        | ``enwiki20180211`` Wikipedia dump.              |
+----------------------------------------------------------------------------------------+-------------------------------------------------+

Comparison
==========

Scores for **ODQA** skill:

+-------------------------------------------------------------------------------------+------+----------------------+----------------+---------------------+---------------------+
|                                                                                     |      |                      |                |   Ranker@5          |   Ranker@25         |
|                                                                                     |      |                      |                +----------+----------+-----------+---------+
| Model                                                                               | Lang |    Dataset           |   WikiDump     |  F1      |   EM     |   F1      |   EM    |
+-------------------------------------------------------------------------------------+------+----------------------+----------------+----------+----------+-----------+---------+
|:config:`DeppPavlov <odqa/en_odqa_infer_wiki.json>`                                  |      |                      | enwiki20180211 |  35.89   |  29.21   |  39.96    |  32.64  |
+-------------------------------------------------------------------------------------+      +                      +----------------+----------+----------+-----------+---------+
|:config:`DeepPavlov <odqa/en_odqa_infer_enwiki20161221.json>`                        |  En  |   SQuAD (dev)        |                | **37.83**|**31.26** |  41.86    |  34.73  |
+-------------------------------------------------------------------------------------+      +                      +                +----------+----------+-----------+---------+
|`DrQA`_                                                                              |      |                      |                |   \-     |  27.1    |   \-      |   \-    |
+-------------------------------------------------------------------------------------+      +                      +                +----------+----------+-----------+---------+
|`R3`_                                                                                |      |                      | enwiki20161221 |  37.5    |  29.1    |   \-      |   \-    |
+-------------------------------------------------------------------------------------+------+----------------------+----------------+----------+----------+-----------+---------+
|:config:`DeepPavlov with RuBERT reader <odqa/ru_odqa_infer_wiki_rubert.json>`        |      |                      |                | **42.02**|**29.56** |   \-      |   \-    |
+-------------------------------------------------------------------------------------+  Ru  +  SDSJ Task B (dev)   + ruwiki20180401 +----------+----------+-----------+---------+
|:config:`DeepPavlov <odqa/ru_odqa_infer_wiki.json>`                                  |      |                      |                |  28.56   |  18.17   |   \-      |   \-    |
+-------------------------------------------------------------------------------------+------+----------------------+----------------+----------+----------+-----------+---------+

EM stands for "exact-match accuracy". Metrics are counted for top 5 and top 25 documents returned by retrieval module.

References
==========

.. target-notes::

.. _`DrQA`: https://github.com/facebookresearch/DrQA/
.. _`R-NET`: https://www.microsoft.com/en-us/research/publication/mcr/
.. _`implementation`: https://github.com/HKUST-KnowComp/R-Net/
.. _`R3`: https://arxiv.org/abs/1709.00023


