Open Domain Question Answering Skill on Wikipedia
=================================================

Task definition
---------------

Open Domain Question Answering (ODQA) is a task to find an exact
answer to any question in
Wikipedia articles. Thus, given only a question, the system outputs
the best answer it can find:

Question:

    What is the name of Darth Vader's son?

Answer:

    Luke Skywalker

Languages
---------

There are pretrained ODQA models for **English** and **Russian**
languages in DeepPavlov.

Models
------

The architecture of ODQA skill is modular and consists of two models,
a ranker and a reader. The ranker is based on
DrQa proposed by Facebook Research (`Reading Wikipedia to Answer
Open-Domain Questions <https://arxiv.org/abs/1704.00051>`__)
and the reader is based on R-Net proposed by Microsoft Research Asia
(`"R-NET: Machine Reading Comprehension with Self-matching
Networks" <https://www.microsoft.com/en-us/research/publication/mrc/>`__)
and its `implementation <https://github.com/HKUST-KnowComp/R-Net>`__
by Wenxuan Zhou.

Running ODQA
------------

**Tensorflow-1.8.0 with GPU support is required** to run this model.

**About 16 GB of RAM required**

Training
--------

The ODQA ranker and ODQA reader should be trained separately.
Read about training the **ranker** :doc:`here </userdocs/tfidf_ranking>`.
Read about training the **reader** in our separate `reader
tutorial <https://github.com/deepmipt/DeepPavlov/tree/master/deeppavlov/models/squad>`__.

Interacting
-----------

When interacted, the ODQA model returns a plain answer to the user's
question.

Run the following to interact **English** ODQA:

.. code:: bash

    cd deeppavlov/
    python deep.py interact deeppavlov/configs/odqa/en_odqa_infer_wiki.json -d

Run the following to interact the ranker:

.. code:: bash

    cd deeppavlov/
    python deep.py interact deeppavlov/configs/odqa/ru_odqa_infer_wiki.json -d

Configuration
-------------

The ODQA configs suit only model inferring purposes. The `ranker
config <#the-ranker-config>`__ should be used for ranker training
and the :doc:`reader config </userdocs/squad>`
should be used for reader training.

References
----------

#. https://github.com/facebookresearch/DrQA
#. https://github.com/HKUST-KnowComp/R-Net

