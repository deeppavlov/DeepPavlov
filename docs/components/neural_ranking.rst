Ranking and paraphrase identification
=====================================

This library component solves the tasks of ranking and paraphrase identification based on semantic similarity
which is trained with siamese neural networks. The trained network can retrieve the response
closest semantically to a given context from some database or answer whether two sentences are paraphrases or not.
It is possible to build automatic semantic FAQ systems with such neural architectures.

Ranking
-------

Before using the model make sure that all required packages are installed running the command:

.. code:: bash

    python -m deeppavlov install ranking_insurance

To train the model on the `InsuranceQA V1`_ dataset one can use the following code in python:

.. code:: python

    from deeppavlov import configs, train_model

    rank_model = train_model(configs.ranking.ranking_insurance, download=True)

To train from command line:

::

    python -m deeppavlov train deeppavlov/configs/ranking/ranking_insurance.json [-d]

As an example of configuration file see
:config:`ranking_insurance.json <ranking/ranking_insurance.json>`.


To use the model pre-trained on the `InsuranceQA V1`_ dataset for
inference one can use the following code in python:

.. code:: python

    from deeppavlov import build_model, configs

    rank_model = build_model(configs.ranking.ranking_insurance, download=True)
    rank_model(['how much to pay for auto insurance?'])

To run the model for inference from command line:

::

    python -m deeppavlov interact deeppavlov/configs/ranking/ranking_insurance_interact.json [-d]

Then a user can enter a context and get responses:

::

    :: how much to pay for auto insurance?
    >> ['the cost of auto insurance be based on several factor include your driving record , claim history , type of vehicle , credit score where you live and how far you travel to and from work I will recommend work with an independent agent who can shop several company find the good policy for you', 'there be not any absolute answer to this question rate for auto insurance coverage can vary greatly from carrier to carrier and from area to area contact local agent in your area find out about coverage availablity and pricing within your area look for an agent that you be comfortable working with as they will be the first last point of contact in most instance', 'the cost of auto insurance coverage for any vehicle or driver can vary greatly thing that effect your auto insurance rate be geographical location , vehicle , age (s) of driver (s) , type of coverage desire , motor vehicle record of all driver , credit rating of all driver and more contact a local agent get a quote a quote cost nothing but will let you know where your rate will']


Paraphrase identification
-------------------------

Before using the model make sure that all required packages are installed running the command:

.. code:: bash

    python -m deeppavlov install paraphrase_ident_qqp

To train the model on the `Quora Question Pairs`_ dataset one can use the following code in python:

.. code:: python

    from deeppavlov import configs, train_model

    para_model = train_model(configs.ranking.paraphrase_ident_qqp, download=True)

To train from command line:

::

    python -m deeppavlov train deeppavlov/configs/ranking/paraphrase_ident_qqp.json [-d]

As an example of configuration file see
:config:`paraphrase_ident_qqp.json <ranking/paraphrase_ident_qqp.json>`.


To use the model pre-trained on the `Quora Question Pairs`_ dataset for
inference, one can use the following code in python:

.. code:: python

    from deeppavlov import build_model, configs

    para_model = build_model(configs.ranking.paraphrase_ident_qqp_interact, download=True)
    para_model(['How can I be a good geologist?&What should I do to be a great geologist?'])

To use the model for inference from command line:

::

    python -m deeppavlov interact deeppavlov/configs/ranking/paraphrase_ident_qqp_interact.json [-d]

Now a user can enter two sentences and the model will make a prediction whether these sentences are paraphrases or not.

::

    :: How can I be a good geologist?&What should I do to be a great geologist?
    >> This is a paraphrase.

.. _`InsuranceQA V1`: https://github.com/shuzi/insuranceQA
.. _`Quora Question Pairs`: https://www.kaggle.com/c/quora-question-pairs/data
