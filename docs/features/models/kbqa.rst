Knowledge Base Question Answering (KBQA)
========================================

Description
-----------

The Knowledge Base Question Answering model uses Wikidata to answer question. To find the answer the following
models are used:
:doc:`NER model </features/models/ner>` performs entity discovery. In a given question it finds a substring which
is an entity, possible mentioned in a Knowledge Base.
:doc:`Classification model </features/models/ner>` classifies the question into a set of predefined relations from
Wikidata.
Substring extracted by the NER model is used for entity linking. Entity linking preforms matching the substring
with one of the Wikidata entities. Matching is based on Levenshtein distance between the substring and an entity
description. The result of the matching procedure is a set of candidate entities. The reset is search of the
entity among this set with one of the top-k relations predicted by classification model.


Use the model
-------------

At the moment the model that solves KBQA task is available only for Russian language.
Any pre-trained model can be used for inference from both Command Line Interface (CLI) and Python. Before using the
model make sure that all required packages are installed using the command:

.. code:: bash

    python -m deeppavlov install kbqa_rus

To use a pre-trained model from CLI use the following command:

.. code:: bash

    python deeppavlov/deep.py interact kbqa_rus [-d]

where ``kbqa_rus`` is the name of the config and ``-d`` is an optional download key. The key ``-d`` is used
to download the pre-trained model along with embeddings and all other files needed to run the model. Also command
``download`` is possible,


Available config:

.. table::
    :widths: auto

    +-----------------------------------------------+-------------------+-----------------+------------+------------+
    | Model                                         | Dataset           | Embeddings Size | Model Size |  Wikidata  |
    +===============================================+===================+=================+============+============+
    | :config:`kbqa_rus <kbqa/kbqa_rus.json>`       | Simple Questions  |     7.7 GB      |   8.9 MB   |   1.1 GB   |
    |                                               | and Zero-Shot IE  |                 |            |            |
    +-----------------------------------------------+-------------------+-----------------+------------+------------+


Models can be used from Python using the following code:

.. code:: python

    from deeppavlov import configs, build_model

    kbqa_model = build_model(configs.kbqa.kbqa_rus, download=True)
    kbqa_model(['Когда родился Пушкин?'])
    >>> ["1799-05-26"]
