Ranking and paraphrase identification
=====================================

This library model solves the tasks of ranking and paraphrase identification based on semantic similarity
which is trained with siamese neural networks. The trained network can retrieve the response
closest semantically to a given context from some database or answer whether two sentences are paraphrases or not.
It is possible to build automatic semantic FAQ systems with such neural architectures.

Training and inference models on predifined datasets
----------------------------------------------------

BERT Ranking
~~~~~~~~~~~~

Before using models make sure that all required packages are installed running the command for TensorFlow:

.. code:: bash

    python -m deeppavlov install ranking_ubuntu_v2_torch_bert_uncased


To train the interaction-based (accurate, slow) model on the `Ubuntu V2`_ from command line:

::

    python -m deeppavlov train ranking_ubuntu_v2_bert_uncased [-d]

To train the representation-based (accurate, fast) model on the `Ubuntu V2`_ from command line:

::

    python -m deeppavlov train ranking_ubuntu_v2_bert_sep [-d]

Further the trained representation-based model can be run for inference over the provided response base
(~500K in our case) from command line:

::

    python -m deeppavlov interact ranking_ubuntu_v2_bert_sep_interact [-d]

Statistics on the models quality are available :doc:`here </features/overview>`.

Ranking
~~~~~~~

To use Sequential Matching Network (SMN) or Deep Attention Matching Network (DAM) or
Deep Attention Matching Network with Universal Sentence Encoder (DAM-USE-T)
on the `Ubuntu V2`_ for inference, please run one of the following commands:

::

    python -m deeppavlov interact -d ranking_ubuntu_v2_mt_word2vec_smn
    python -m deeppavlov interact -d ranking_ubuntu_v2_mt_word2vec_dam_transformer

Now a user can enter a dialog consists of 10 context sentences and several (>=1) candidate response sentences separated by '&'
and then get the probability that the response is proper continuation of the dialog:

::

    :: & & & & & & & & bonhoeffer  whar drives do you want to mount what &  i have an ext3 usb drive  & look with fdisk -l & hello there & fdisk is all you need
    >> [0.9776373  0.05753616 0.9642599 ]

Paraphrase identification
~~~~~~~~~~~~~~~~~~~~~~~~~

Paraphraser.ru dataset
~~~~~~~~~~~~~~~~~~~~~~

Before using the model make sure that all required packages are installed running the command:

.. code:: bash

    python -m deeppavlov install paraphraser_bert

To train the model on the `paraphraser.ru`_ dataset one can use the following code in python:

.. code:: python

    from deeppavlov import configs, train_model

    para_model = train_model('paraphraser_bert', download=True)


Paraphrase identification
~~~~~~~~~~~~~~~~~~~~~~~~~

**train.csv**: the same as for ranking.

**valid.csv**, **test.csv**: each line in the file contains ``context``, ``response`` and ``label`` separated by the tab key. ``label`` is
binary, i.e. 1 or 0 corresponding to the correct or incorrect ``response`` for the given ``context``.
Instead of ``response`` and ``context`` it can be simply two phrases which are paraphrases or non-paraphrases as indicated by the ``label``.

Classification metrics on the valid and test dataset parts (the parameter ``metrics`` in the JSON configuration file)
such as ``f1``, ``acc`` and ``log_loss``  can be calculated.

.. _`paraphraser.ru`: https://paraphraser.ru
.. _`Ubuntu V2`: https://github.com/rkadlec/ubuntu-ranking-dataset-creator
