Entity Linking
========================================

Entity linking is the task of mapping of words in the text, corresponding to names of persons, locations and organizations, to entities in the target knowledge base (Wikidata in our case).

Entity Linking component consists of the following steps:

* the substring, detected with :config:`NER <ner/ner_rus_bert_probas.json>`, is fed into TfidfVectorizer with the following parameters: analyzer="char_wb", ngram_range=(2, 2), max_features=500, and the vector is converted to dense
* `Faiss <https://github.com/facebookresearch/faiss>`__ library is used to find k nearest neighbours for tf-idf vector in the matrix where rows correspond to tf-idf vectors of words in entity titles
* Entities are ranked by number of relations in Wikidata (number of outgoing edges of nodes in the knowledge graph)
* :config:`BERT <classifiers/entity_ranking_bert_rus_no_mention.json>` is used for ranking of entities by description of the entity and the sentence with the mention of the entity.

Use the model
-------------

Entity Linking model is available for Russian language. Pre-trained model can be used for inference from both Command Line Interface (CLI) and Python. Before using the model make sure that all required packages are installed using the command:

.. code:: bash

    python -m deeppavlov install entity_linking_rus

To use a pre-trained model from CLI use the following command:

.. code:: bash

    python deeppavlov/deep.py interact entity_linking_rus [-d]

where ``entity_linking_rus`` and others are the names of configs and ``-d`` is an optional download key. The key ``-d`` is used
to download the pre-trained model along with embeddings and all other files needed to run the model. Also command
``download`` is possible,


Entity Linking model can be used from Python using the following code:

.. code:: python

    from deeppavlov import configs, build_model

    el_model = build_model(configs.kbqa.entity_linking_rus, download=True)
    el_model(['Москва — столица России, город федерального значения, административный центр Центрального федерального округа и центр Московской области.'])
    >>> [['Q649', 'Q159', 'Q190778', 'Q1749']]
