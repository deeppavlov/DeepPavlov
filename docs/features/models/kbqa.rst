Knowledge Base Question Answering (KBQA)
========================================

Description
-----------

There are three models for KBQA in DeepPavlov library: model for solving of complex questions in English, model for solving of simple question in Russian and model for solving of simple questions in Russian using syntactic parser.

The Complex Knowledge Base Question Answering model uses Wikidata to answer complex questions. Types of questions which the model answers:

Complex questions with numerical values:
"What position did Angela Merkel hold on November 10, 1994?"

Complex question where the answer is number or date:
"When did Jean-Paul Sartre move to Le Havre?"

Questions with counting of answer entities:
"How many sponsors are for Juventus F.C.?"

Questions with ordering of answer entities by ascending or descending of some parameter:
"Which country has highest individual tax rate?"

Simple questions:
"What is crew member Yuri Gagarin's Vostok?"

To find the answer the following
models are used:

BERT model for prediction of query template type. Model performs classification of questions into 8 classes correponding to 8 query template types.

BERT entity detection model for extraction of entity substrings from the questions. 

Substring extracted by the entity detection model is used for entity linking. Entity linking preforms matching the substring
with one of the Wikidata entities. Matching is based on Levenshtein distance between the substring and an entity
title. The result of the matching procedure is a set of candidate entities. The reset is search of the
entity among this set with one of the top-k relations predicted by classification model.

BiGRU model for ranking of candidate relations.

BERT model for ranking of candidate relation paths.

Query generator model is used to fill query template with candidate entities and relations (to find valid combinations of entities and relations for query template). Query Generation model uses Wikidata HDT file. Query Generation Online model uses Wikidata Query Service.

Model for simple question answering in Russian uses the following models: :doc:`NER model </features/models/ner>` performs entity discovery. In a given question it finds a substring which is an entity, possible mentioned in a Knowledge Base. :doc:`Classification model </features/models/ner>` classifies the question into a set of predefined relations from Wikidata. Substring extracted by the NER model is used for entity linking. Entity linking preforms matching the substring with one of the Wikidata entities. Matching is based on Levenshtein distance between the substring and an entity description. The result of the matching procedure is a set of candidate entities. The reset is search of the entity among this set with one of the top-k relations predicted by classification model.

Model for simple question answering with syntactic parser uses UDPipe for parsing of syntactic tree to extract candidate entity and relation substrings.

Use the model
-------------

Any pre-trained model can be used for inference from both Command Line Interface (CLI) and Python. Before using the
model make sure that all required packages are installed using the command:

.. code:: bash

    python -m deeppavlov install kbqa_cq
    python -m deeppavlov install kbqa_cq_online
    python -m deeppavlov install kbqa_rus
    python -m deeppavlov install kbqa_tree

To use a pre-trained model from CLI use the following command:

.. code:: bash

    python deeppavlov/deep.py interact kbqa_сq [-d]
    python deeppavlov/deep.py interact kbqa_cq_online [-d]
    python deeppavlov/deep.py interact kbqa_rus [-d]
    python deeppavlov/deep.py interact kbqa_tree [-d]

where ``kbqa_cq`` and others are the names of configs and ``-d`` is an optional download key. The key ``-d`` is used
to download the pre-trained model along with embeddings and all other files needed to run the model. Also command
``download`` is possible,



KBQA model for complex question answering can be used from Python using the following code:

.. code:: python

    from deeppavlov import configs, build_model

    kbqa_model = build_model(configs.kbqa.kbqa_cq, download=True)
    kbqa_model(['When did Jean-Paul Sartre move to Le Havre?'])
    >>> ["1931-01-01"]
    kbqa_model(['How many sponsors are for Juventus F.C.?'])
    >>> [4]

KBQA model for simple question answering in Russian can be used from Python using the following code:

.. code:: python

    from deeppavlov import configs, build_model

    kbqa_model = build_model(configs.kbqa.kbqa_rus, download=True)
    kbqa_model(['Когда родился Пушкин?'])
    >>> ["1799-05-26"]
