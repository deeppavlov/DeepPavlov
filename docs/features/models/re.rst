Relation Extraction (RE)
==============================

Relation extraction is the task of detecting and classifying the relationship between to entities in text.
DeepPavlov provides the document-level relation extraction meaning that the relation can be detected between the entities that are not in one sentence.
Currently, RE is available only for English language.

RE Component Architecture
-----------------------

Train the model
-----------------------

The RE model can be trained using the following command:

.. code:: bash

    python3.6 -m deeppavlov train re_docred


Use the trained RE model
-----------------------

Currently, we provide RE trained on `DocRED<https://www.aclweb.org/anthology/P19-1074/>`__ corpus.
We based our model on the `Adaptive Thresholding and Localized Context Pooling<https://arxiv.org/pdf/2010.11304.pdf>`__ model and used NER entity tags as additional input.

Model input:

- text document as a list ok tokens
- list of entity information
    - positions of all mentions of the first entity
    - positions of all mentions of the second entity
    - NER tag of the first entity
    - NER tag of the second entity
E.g., if there is a document "Barack Obama is married to Michelle Obama, born Michelle Robinson.", where we want to figure out whether there is a relation between entity 1 ("Barack Obama") and entity 2 ("Michelle Obama" / "Michelle Robinson"), the input for the model will be the following:

``
    ["Barack", "Obama", "is", "married", "to", "Michelle", "Obama", "," "born", "Michelle", "Robinson", "."]
    [[(0, 2)], [(5, 7), (9, 11)], "PER", "PER"]
``

As NER tags, we adapted the used in the DocRED corpus, which are, in turn, inherited from `Tjong Kim Sang and De Meulder (2003)<https://aclanthology.org/W03-0419/>`__ :

+-------+----------------------------------------------------------------------------------------------+
|PER    | People, including fictional                                                                  |
+-------+----------------------------------------------------------------------------------------------+
|ORG    | Companies, universities, institutions, politi- cal or religious groups, etc.                 |
+-------+----------------------------------------------------------------------------------------------+
|LOC    | Geographically defined locations, including mountains, waters, etc.                          |
        | Politically defined locations, including LOC countries, cities, states, streets, etc.        |
        | Facilities, including buildings, museums, stadiums, hospitals, factories, airports, etc.x    |
+-------+----------------------------------------------------------------------------------------------+
|TIME   | Absolute or relative dates or periods.                                                       |
+-------+----------------------------------------------------------------------------------------------+
|NUM    | Percents, money, quantities                                                                  |
+-------+----------------------------------------------------------------------------------------------+
|MISC   | Products, including vehicles, weapons, etc.                                                  |
        | Events, including elections, battles, sporting MISC events, etc. Laws, cases, languages, etc |
+-------+----------------------------------------------------------------------------------------------+

The trained model weights can be loaded with the following command:

.. code:: bash

    python3.6 -m deeppavlov download re_docred

The trained model can be used for inference with the following code:

.. code:: python

    from deeppavlov import configs, build_model
    re = build_model(configs.relation_extraction.re_docred, download=False)

    sentence_tokens = ["Barack", "Obama", "is", "married", "to", "Michelle", "Obama", "," "born", "Michelle", "Robinson", "."]
    entity_info = [[(0, 2)], [(5, 7), (9, 11)], "PER", "PER"]
    re([[sentence_tokens, entity_info]])
    >> [["spouse"]]
