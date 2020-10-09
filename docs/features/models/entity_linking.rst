Entity Linking
========================================

Entity linking is the task of mapping words from text (e.g. names of persons, locations and organizations) to entities from the target knowledge base (Wikidata in our case).

Entity Linking systems are available for English and Russian languages.

Entity Linking component performs the following steps:

* the substring, detected with :config:`NER (English) <ner/ner_ontonotes_bert_probas.json>` or :config:`NER (Russian) <ner/ner_rus_bert_probas.json>`, is fed to TfidfVectorizer and the resulting sparse vector is converted to dense one
* `Faiss <https://github.com/facebookresearch/faiss>`__ library is used to find k nearest neighbours for tf-idf vector in the matrix where rows correspond to tf-idf vectors of words in entity titles
* entities are ranked by number of relations in Wikidata (number of outgoing edges of nodes in the knowledge graph)
* :config:`BERT (English) <classifiers/entity_ranking_bert_eng_no_mention.json>` or :config:`BERT (Russian) <classifiers/entity_ranking_bert_rus_no_mention.json>` is used for entities ranking by entity description and by sentence that mentions the entity

Use the model
-------------

Pre-trained model can be used for inference from both Command Line Interface (CLI) and Python. Before using the model make sure that all required packages are installed using the command:

For English version:

.. code:: bash

    python -m deeppavlov install entity_linking_eng

To use a pre-trained model from CLI use the following command:

.. code:: bash

    python -m deeppavlov interact entity_linking_eng -d
    >>> The city stands on the River Thames in the south-east of England, at the head of its 50-mile (80 km) estuary leading to the North Sea.
    >>> (['the river thames', 'the north sea', 'england'], [[4, 5, 6], [30, 31, 32], [13]], ['Q19686', 'Q1693', 'Q21'])

For Russian version:

.. code:: bash

    python -m deeppavlov install entity_linking_rus

To use a pre-trained model from CLI use the following command:

.. code:: bash

    python -m deeppavlov interact entity_linking_rus -d
    >>> Москва — столица России, город федерального значения, административный центр Центрального федерального округа и центр Московской области.
    >>> (['москва', 'россии', 'центрального федерального округа', 'московской области'], [[0], [3], [11, 12, 13], [16, 17]], ['Q649', 'Q159', 'Q190778', 'Q1749'])

Entity Linking model can be used from Python using the following code:

.. code:: python

    from deeppavlov import configs, build_model

    el_model = build_model(configs.kbqa.entity_linking_rus, download=True)
    el_model(['Москва — столица России, город федерального значения, административный центр Центрального федерального округа и центр Московской области.'])
