Entity Extraction
========================================
Entity Detection is the task of identifying entity mentions in text with corresponding entity types.
Entity Detection configs are available in :config:`English <entity_extraction/entity_detection_en.json>` and :config:`Russian <entity_extraction/entity_detection_ru.json>` languages. These configs support entity detection in texts longer than 512 tokens.

Use the model
-------------

Pre-trained model can be used for inference from both Command Line Interface (CLI) and Python. Before using the model make sure that all required packages are installed using the command:

For English version:

.. code:: bash

    python -m deeppavlov install entity_detection_en

To use a pre-trained model from CLI use the following command:

.. code:: bash

    python -m deeppavlov interact entity_detection_en -d
    >>> Forrest Gump is a comedy-drama film directed by Robert Zemeckis and written by Eric Roth.
    >>> ([['forrest gump', 'robert zemeckis', 'eric roth']], [[(0, 12), (48, 63), (79, 88)]], [[[0, 1], [10, 11], [15, 16]]], [['WORK_OF_ART', 'PERSON', 'PERSON']], [[(0, 89)]], [['Forrest Gump is a comedy-drama film directed by Robert Zemeckis and written by Eric Roth.']], [[0.8997, 0.9979, 0.9979]])
    
The output elements:

* entity substrings
* entity offsets (indices of start and end symbols of entities in text)
* entity positions (indices of entity tokens in text)
* entity tags
* sentences offsets
* list of sentences in text 

For Russian version:

.. code:: bash

    python -m deeppavlov install entity_linking_ru

To use a pre-trained model from CLI use the following command:

.. code:: bash

    python -m deeppavlov interact entity_linking_ru -d
    >>> Москва — столица России, центр Центрального федерального округа и центр Московской области.
    >>> ([['москва', 'россии', 'центрального федерального округа', 'московской области']], [[(0, 6), (17, 23), (31, 63), (72, 90)]], [[[0], [3], [6, 7, 8], [11, 12]]], [['CITY', 'COUNTRY', 'LOC', 'LOC']], [[(0, 91)]], [['Москва — столица России, центр Центрального федерального округа и центр Московской области.']], [[0.8359, 0.938, 0.9917, 0.9803]])
    
Entity Detection model can be used from Python using the following code:

.. code:: python

    from deeppavlov import build_model

    ed = build_model('entity_detection_en', download=True)
    ed(['Forrest Gump is a comedy-drama film directed by Robert Zemeckis and written by Eric Roth.'])

Entity Linking is the task of finding knowledge base entity ids for entity mentions in text. Entity Linking in DeepPavlov supports Wikidata and Wikipedia (for :config:`English <entity_extraction/entity_linking_en.json>` and :config:`Russian <entity_extraction/entity_linking_ru.json>`). Entity Linking component performs the following steps:

* extraction of candidate entities from SQLite database;
* candidate entities sorting by entity tags (if entity tags are provided);
* ranking of candidate entities by connections in Wikidata knowledge graph of candidate entities for different mentions;
* candidate entities ranking by context and descriptions using Transformer model `bert-small <https://huggingface.co/prajjwal1/bert-small>`__ in English config and `distilrubert-tiny <https://huggingface.co/DeepPavlov/distilrubert-tiny-cased-conversational-v1>`__.

Entity linking models in DeepPavlov are lightweight: English version requires 2.4 Gb RAM and 1.2 Gb GPU, Russian version 2.2 Gb RAM and 1.1 Gb GPU.

Entity Extraction configs perform subsequent Entity Detection and Entity Linking of extracted entity mentions.
Entity Extraction configs are available for :config:`English <entity_extraction/entity_extraction_en.json>` and :config:`Russian <entity_extraction/entity_extraction_ru.json>`.

Use the model
-------------

For English version:

.. code:: bash

    python -m deeppavlov install entity_extraction_en

To use a pre-trained model from CLI use the following command:

.. code:: bash

    python -m deeppavlov interact entity_extraction_en -d
    >>> Forrest Gump is a comedy-drama film directed by Robert Zemeckis and written by Eric Roth.
    >>> (['forrest gump', 'robert zemeckis', 'eric roth'], ['WORK_OF_ART', 'PERSON', 'PERSON'], [(0, 12), (48, 63), (79, 88)], ['Q134773', 'Q187364', 'Q942932'], [(1.0, 110, 1.0), (1.0, 73, 1.0), (1.0, 37, 0.95)], ['Forrest Gump', 'Robert Zemeckis', 'Eric Roth'])

For Russian version:

.. code:: bash

    python -m deeppavlov install entity_extraction_ru

To use a pre-trained model from CLI use the following command:

.. code:: bash

    python -m deeppavlov interact entity_extraction_ru -d
    >>> Москва — столица России, центр Центрального федерального округа и центр Московской области.
    >>> (['москва', 'россии', 'центрального федерального округа', 'московской области'], ['CITY', 'COUNTRY', 'LOC', 'LOC'], [(0, 6), (17, 23), (31, 63), (72, 90)], ['Q649', 'Q159', 'Q190778', 'Q1697'], [(1.0, 134, 1.0), (1.0, 203, 1.0), (0.97, 24, 0.28), (0.9, 30, 1.0)], ['Москва', 'Россия', 'Центральный федеральный округ', 'Московская область'])

Entity Linking model can be used from Python using the following code:

.. code:: python

    from deeppavlov import build_model

    entity_extraction = build_model('entity_extraction_en', download=True)
    entity_extraction(['Forrest Gump is a comedy-drama film directed by Robert Zemeckis and written by Eric Roth.'])
