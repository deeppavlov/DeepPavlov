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

Before using models make sure that all required packages are installed running the command:

.. code:: bash

    python -m deeppavlov install ranking_ubuntu_v2_torch_bert_uncased


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
