=================================================
Open Domain Question Answering Skill on Wikipedia
=================================================

Task definition
===============

**Open Domain Question Answering (ODQA)** is a task to find an exact answer
to any question in **Wikipedia** articles. Thus, given only a question, the system outputs
the best answer it can find:

::

    :: What is the name of Darth Vader's son?
    >> Luke Skywalker

Languages
=========

There are pretrained **ODQA** models for **English** and **Russian**
languages in DeepPavlov :doc:`DeepPavlov </index/>`.

Models
======

The architecture of **ODQA** skill is modular and consists of two models,
a **ranker** and a **reader**. The **ranker** is based on `DrQA`_ proposed by Facebook Research
and the **reader** is based on `R-NET`_ proposed by Microsoft Research Asia
and its `implementation`_ by Wenxuan Zhou.

Running ODQA
============

**Tensorflow-1.8.0 with GPU support is required** to run this model.

**About 16 GB of RAM required**

.. note::

    TensorFlow 1.8 with GPU support is required to run this skill.

    About 16 GB of RAM required.

Training
--------

**ODQA ranker** and **ODQA reader** should be trained separately.
Read about training the **ranker** :ref:`here <ranker_training>`.
Read about training the **reader** in our separate :doc:`reader tutorial </apiref/models/squad>`.

Interacting
-----------

When interacting, the **ODQA** skill returns a plain answer to the user's
question.

Run the following to interact with **English ODQA**:

.. code:: bash

    cd deeppavlov/
    python deep.py interact deeppavlov/configs/odqa/en_odqa_infer_wiki.json -d

Run the following to interact with **Russian ODQA**:

.. code:: bash

    cd deeppavlov/
    python deep.py interact deeppavlov/configs/odqa/ru_odqa_infer_wiki.json -d

Configuration
=============

The **ODQA** configs suit only model inferring purposes. For training purposes use
the :ref:`ranker configs <ranker_training>` and the :ref:`reader configs <reader_training>`
accordingly.

Comparison
==========

Scores for **ODQA** skill:


+-----------------------+-----------------------------+----------------+-----------------------+--------+------+
| Model                                               | Dataset        |  Wiki dump            |  F1    |  EM  |
+-----------------------------------------------------+----------------+-----------------------+--------+------+
|:config:`DeepPavlov <odqa/en_odqa_infer_wiki.json>`  | SQuAD (dev)    |   enwiki (2018-02-11) |  28.0  | 22.2 |
+-----------------------------------------------------+----------------+-----------------------+--------+------+
|`DrQA`_                                              | SQuAD (dev)    |   enwiki (2016-12-21) |   \-   | 27.1 |
+-----------------------------------------------------+----------------+-----------------------+--------+------+


EM stands for "exact-match accuracy". Metrics are counted for top 5 documents returned by retrieval module.

References
==========

.. target-notes::

.. _`DrQA`: https://github.com/facebookresearch/DrQA/
.. _`R-NET`: https://www.microsoft.com/en-us/research/publication/mrc/
.. _`implementation`: https://github.com/HKUST-KnowComp/R-Net/


