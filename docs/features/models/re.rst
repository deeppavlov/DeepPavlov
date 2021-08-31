Relation Extraction (RE)
==============================

Relation extraction is the task of detecting and classifying the relationship between two entities in text.
DeepPavlov provides the document-level relation extraction meaning that the relation can be detected between the entities that are not in one sentence.
Currently, RE is available for English and Russian languages.

- :config:`RE model <relation_extraction/re_docred.json>` for English language trained on `DocRED <https://www.aclweb.org/anthology/|P19-1074/>`__ corpus based on Wikipedia.
- :config:`RE model <relation_extraction/re_rured.json>` for Russian language trained on `RuRED <http://www.dialog-21.ru/media/5093/gordeevdiplusetal-031.pdf>`__ corpus based on the Lenta.ru news corpus.

English RE model
----------------------------

The English RE model can be trained using the following command:

.. code:: bash

    python -m deeppavlov train re_docred

The trained model weights can be loaded with the following command:

.. code:: bash

    python -m deeppavlov download re_docred

The trained model can be used for inference with the following code:

.. code:: python

    from deeppavlov import configs, build_model
    re = build_model(configs.relation_extraction.re_docred, download=False)

    sentence_tokens = [["Barack", "Obama", "is", "married", "to", "Michelle", "Obama", ",", "born", "Michelle", "Robinson", "."]]
    entity_pos = [[[(0, 2)], [(5, 7), (9, 11)]]]
    entity_tags = [["PER", "PER"]]
    pred = re_model(sentence_tokens, entity_pos, entity_tags)
    >> [['P26'], ['spouse']]

**Model Input**:

- list of tokens of a text document
- list of entities positions (i.e. all start and end positions of both entities' mentions)
- list of NER tags of both entities.

As NER tags, we adapted the used in the DocRED corpus, which are, in turn, inherited from `Tjong Kim Sang and De Meulder(2003) <https://aclanthology.org/W03-0419/>`__

.. raw:: html

   <details>
   <summary><a>The whole list of 6 English NER tags</a></summary>

+-------+------------------------------------------------------------------------------------------------+
|PER    | People, including fictional                                                                    |
+-------+------------------------------------------------------------------------------------------------+
|ORG    | Companies, universities, institutions, political or religious groups, etc.                     |
+-------+------------------------------------------------------------------------------------------------+
|LOC    | Geographically defined locations, including mountains, waters, etc.                            |
|       | Politically defined locations, including countries, cities, states, streets, etc.              |
|       | Facilities, including buildings, museums, stadiums, hospitals, factories, airports, etc.       |
+-------+------------------------------------------------------------------------------------------------+
|TIME   | Absolute or relative dates or periods.                                                         |
+-------+------------------------------------------------------------------------------------------------+
|NUM    | Percents, money, quantities                                                                    |
+-------+------------------------------------------------------------------------------------------------+
|MISC   | Products, including vehicles, weapons, etc.                                                    |
|       | Events, including elections, battles, sporting MISC events, etc. Laws, cases, languages, etc   |
+-------+------------------------------------------------------------------------------------------------+

.. raw:: html

   </details>


**Model Output**: one or several of the 97 relations found between the given entities; relation id in `Wikidata <https://www.wikidata.org/wiki/Wikidata:Main_Page>`__ (e.g. 'P26') and relation name ('spouse').

.. raw:: html

   <details>
   <summary><a>The whole list of English relation</a></summary>

+----------------+-----------------------------------------------------+
|Relation id     |  Relation                                           |
+----------------+-----------------------------------------------------+
|P6              |  head of government                                 |
+----------------+-----------------------------------------------------+
|P17             |  country                                            |
+----------------+-----------------------------------------------------+
|P19             |  place of birth                                     |
+----------------+-----------------------------------------------------+
|P20             |  place of death                                     |
+----------------+-----------------------------------------------------+
|P22             |  father                                             |
+----------------+-----------------------------------------------------+
|P25             |  mother                                             |
+----------------+-----------------------------------------------------+
|P26             |  spouse                                             |
+----------------+-----------------------------------------------------+
|P27             |  country of citizenship                             |
+----------------+-----------------------------------------------------+
|P30             |  continent                                          |
+----------------+-----------------------------------------------------+
|P31             |  instance of                                        |
+----------------+-----------------------------------------------------+
|P35             |  head of state                                      |
+----------------+-----------------------------------------------------+
|P36             |  capital                                            |
+----------------+-----------------------------------------------------+
|P37             |  official language                                  |
+----------------+-----------------------------------------------------+
|P39             |  position held                                      |
+----------------+-----------------------------------------------------+
|P40             |  child                                              |
+----------------+-----------------------------------------------------+
|P50             |  author                                             |
+----------------+-----------------------------------------------------+
|P54             |  member of sports team                              |
+----------------+-----------------------------------------------------+
|P57             |  director                                           |
+----------------+-----------------------------------------------------+
|P58             |  screenwriter                                       |
+----------------+-----------------------------------------------------+
|P69             |  educated at                                        |
+----------------+-----------------------------------------------------+
|P86             |  composer                                           |
+----------------+-----------------------------------------------------+
|P102            |  member of political party                          |
+----------------+-----------------------------------------------------+
|P108            |  employer                                           |
+----------------+-----------------------------------------------------+
|P112            |  founded by                                         |
+----------------+-----------------------------------------------------+
|P118            |  league                                             |
+----------------+-----------------------------------------------------+
|P123            |  publisher                                          |
+----------------+-----------------------------------------------------+
|P127            |  owned by                                           |
+----------------+-----------------------------------------------------+
|P131            |  located in the administrative territorial entity   |
+----------------+-----------------------------------------------------+
|P136            |  genre                                              |
+----------------+-----------------------------------------------------+
|P137            |  operator                                           |
+----------------+-----------------------------------------------------+
|P140            |  religion                                           |
+----------------+-----------------------------------------------------+
|P150            |  contains administrative territorial entity         |
+----------------+-----------------------------------------------------+
|P155            |  follows                                            |
+----------------+-----------------------------------------------------+
|P156            |  followed by                                        |
+----------------+-----------------------------------------------------+
|P159            |  headquarters location                              |
+----------------+-----------------------------------------------------+
|P161            |  cast member                                        |
+----------------+-----------------------------------------------------+
|P162            |  producer                                           |
+----------------+-----------------------------------------------------+
|P166            |  award received                                     |
+----------------+-----------------------------------------------------+
|P170            |  creator                                            |
+----------------+-----------------------------------------------------+
|P171            |  parent taxon                                       |
+----------------+-----------------------------------------------------+
|P172            |  ethnic group                                       |
+----------------+-----------------------------------------------------+
|P175            |  performer                                          |
+----------------+-----------------------------------------------------+
|P176            |  manufacturer                                       |
+----------------+-----------------------------------------------------+
|P178            |  developer                                          |
+----------------+-----------------------------------------------------+
|P179            |  series                                             |
+----------------+-----------------------------------------------------+
|P190            |  sister city                                        |
+----------------+-----------------------------------------------------+
|P194            |  legislative body                                   |
+----------------+-----------------------------------------------------+
|P205            |  basin country                                      |
+----------------+-----------------------------------------------------+
|P206            |  located in or next to body of water                |
+----------------+-----------------------------------------------------+
|P241            |  military branch                                    |
+----------------+-----------------------------------------------------+
|P264            |  record label                                       |
+----------------+-----------------------------------------------------+
|P272            |  production company                                 |
+----------------+-----------------------------------------------------+
|P276            |  location                                           |
+----------------+-----------------------------------------------------+
|P279            |  subclass of                                        |
+----------------+-----------------------------------------------------+
|P355            |  subsidiary                                         |
+----------------+-----------------------------------------------------+
|P361            |  part of                                            |
+----------------+-----------------------------------------------------+
|P364            |  original language of work                          |
+----------------+-----------------------------------------------------+
|P400            |  platform                                           |
+----------------+-----------------------------------------------------+
|P403            |  mouth of the watercourse                           |
+----------------+-----------------------------------------------------+
|P449            |  original network                                   |
+----------------+-----------------------------------------------------+
|P463            |  member of                                          |
+----------------+-----------------------------------------------------+
|P488            |  chairperson                                        |
+----------------+-----------------------------------------------------+
|P495            |  country of origin                                  |
+----------------+-----------------------------------------------------+
|P527            |  has part                                           |
+----------------+-----------------------------------------------------+
|P551            |  residence                                          |
+----------------+-----------------------------------------------------+
|P569            |  date of birth                                      |
+----------------+-----------------------------------------------------+
|P570            |  date of death                                      |
+----------------+-----------------------------------------------------+
|P571            |  inception                                          |
+----------------+-----------------------------------------------------+
|P576            |  dissolved, abolished or demolished                 |
+----------------+-----------------------------------------------------+
|P577            |  publication date                                   |
+----------------+-----------------------------------------------------+
|P580            |  start time                                         |
+----------------+-----------------------------------------------------+
|P582            |  end time                                           |
+----------------+-----------------------------------------------------+
|P585            |  point in time                                      |
+----------------+-----------------------------------------------------+
|P607            |  conflict                                           |
+----------------+-----------------------------------------------------+
|P674            |  characters                                         |
+----------------+-----------------------------------------------------+
|P676            |  lyrics by                                          |
+----------------+-----------------------------------------------------+
|P706            |  located on terrain feature                         |
+----------------+-----------------------------------------------------+
|P710            |  participant                                        |
+----------------+-----------------------------------------------------+
|P737            |  influenced by                                      |
+----------------+-----------------------------------------------------+
|P740            |  location of formation                              |
+----------------+-----------------------------------------------------+
|P749            |  parent organization                                |
+----------------+-----------------------------------------------------+
|P800            |  notable work                                       |
+----------------+-----------------------------------------------------+
|P807            |  separated from                                     |
+----------------+-----------------------------------------------------+
|P840            |  narrative location                                 |
+----------------+-----------------------------------------------------+
|P937            |  work location                                      |
+----------------+-----------------------------------------------------+
|P1001           |  applies to jurisdiction                            |
+----------------+-----------------------------------------------------+
|P1056           |  product or material produced                       |
+----------------+-----------------------------------------------------+
|P1198           |  unemployment rate                                  |
+----------------+-----------------------------------------------------+
|P1336           |  territory claimed by                               |
+----------------+-----------------------------------------------------+
|P1344           |  participant of                                     |
+----------------+-----------------------------------------------------+
|P1365           |  replaces                                           |
+----------------+-----------------------------------------------------+
|P1366           |  replaced by                                        |
+----------------+-----------------------------------------------------+
|P1376           |  capital of                                         |
+----------------+-----------------------------------------------------+
|P1412           |  languages spoken, written or signed                |
+----------------+-----------------------------------------------------+
|P1441           |  present in work                                    |
+----------------+-----------------------------------------------------+
|P3373           |  sibling                                            |
+----------------+-----------------------------------------------------+

.. raw:: html

   </details>


.. raw:: html

   <details>
   <summary><a>Some details on DocRED corpus English RE model was trained on</a></summary>

The English RE model was trained on DocRED English corpus. It was constructed from Wikipedia and Wikidata and is now the largest human-annotated dataset for document-level RE from plain text.

As the original DocRED test dataset containes only unlabeled data, while we want to have labeled one in order to perform evaluation, we decided to:
1. merge train and dev data (= labeled data)
2. split them into new train, dev and test dataset

Currently, there are two types of possible splittings provided:

- user can set the relative size of dev and test data (e.g. 1/7)
- user can set the absolute size of dev and test data (e.g. 2000 samples)

In our experiment, we set the absolute size of dev and test data == 150 initial documents. It resulted in approximately 3500 samples.

We additionally generate negative samples if it was necessary to have the following proportions:
- for train set: negative samples are twice as many as positive ones
- for dev & test set: negative samples are the same amount as positive ones

+----------------+----------------+----------------+
| Train          | Dev            | Test           |
+----------------+----------------+----------------+
| 130650         | 3406           |3545            |
+----------------+----------------+----------------+

+----------------+----------------+----------------+----------------+----------------+----------------+
| Train Positive | Train Negative | Dev Positive   | Dev Negative   | Test Positive  | Test Negative  |
+----------------+----------------+----------------+----------------+----------------+----------------+
| 44823          | 89214          | 1239           | 1229           | 1043           | 1036           |
+----------------+----------------+----------------+----------------+----------------+----------------+

.. raw:: html

   </details>

Russian RE model
----------------------------

The Russian RE model can be trained using the following command:

.. code:: bash

    python -m deeppavlov train re_rured

The trained model weights can be loaded with the following command:

.. code:: bash

    python -m deeppavlov download re_rured

The trained model can be used for inference with the following code:

.. code:: python

    from deeppavlov import configs, build_model
    model = build_model(configs.relation_extraction.re_rured)

    sentence_tokens = [["Илон", "Маск", "живет", "в", "Сиэттле", "."]]
    entity_pos = [[[(0, 2)], [(4, 6)]]]
    entity_tags = [["PERSON", "CITY"]]
    pred = model(sentence_tokens, entity_pos, entity_tags)
    >> [['P551'], ['место жительства']]

**Model Input**:

- list of tokens of a text document
- list of entities positions (i.e. all start and end positions of both entities' mentions)
- list of NER tags of both entities.

.. raw:: html

   <details>
   <summary><a>Full list of 29 Russian NER tags</a></summary>

+-------------+------------------------+------------------------------------------------------------------------+
| NER tag     | Description                                                                                     |
+-------------+------------------------+------------------------------------------------------------------------+
| WORK_OF_ART | name of work of art                                                                             |
+-------------+------------------------+------------------------------------------------------------------------+
| NORP        | affiliation                                                                                     |
+-------------+------------------------+------------------------------------------------------------------------+
| GROUP       | unnamed groups of people and companies                                                          |
+-------------+------------------------+------------------------------------------------------------------------+
| LAW         | law name                                                                                        |
+-------------+------------------------+------------------------------------------------------------------------+
| NATIONALITY | names of nationalities                                                                          |
+-------------+------------------------+------------------------------------------------------------------------+
| EVENT       | event name                                                                                      |
+-------------+------------------------+------------------------------------------------------------------------+
| DATE        | date value                                                                                      |
+-------------+------------------------+------------------------------------------------------------------------+
| CURRENCY    | names of currencies                                                                             |
+-------------+------------------------+------------------------------------------------------------------------+
| GPE         | geo-political entity                                                                            |
+-------------+------------------------+------------------------------------------------------------------------+
| QUANTITY    | quantity value                                                                                  |
+-------------+------------------------+------------------------------------------------------------------------+
| FAMILY      | families as a whole                                                                             |
+-------------+------------------------+------------------------------------------------------------------------+
| ORDINAL     | ordinal value                                                                                   |
+-------------+------------------------+------------------------------------------------------------------------+
| RELIGION    | names of religions                                                                              |
+-------------+------------------------+------------------------------------------------------------------------+
| CITY        | Names of cities, towns, and villages                                                            |
+-------------+------------------------+------------------------------------------------------------------------+
| MONEY       | money name                                                                                      |
+-------------+------------------------+------------------------------------------------------------------------+
| AGE         | people's and object's ages                                                                      |
+-------------+------------------------+------------------------------------------------------------------------+
| LOCATION    | location name                                                                                   |
+-------------+------------------------+------------------------------------------------------------------------+
| PERCENT     | percent value                                                                                   |
+-------------+------------------------+------------------------------------------------------------------------+
| BOROUGH     | Names of sub-city entities                                                                      |
+-------------+------------------------+------------------------------------------------------------------------+
| PERSON      | person name                                                                                     |
+-------------+------------------------+------------------------------------------------------------------------+
| REGION      |  Names of sub-country entities                                                                  |
+-------------+------------------------+------------------------------------------------------------------------+
| COUNTRY     | Names of countries                                                                              |
+-------------+------------------------+------------------------------------------------------------------------+
| PROFESSION  | Professions and people of these professions.                                                    |
+-------------+------------------------+------------------------------------------------------------------------+
| ORGANIZATION| organization name                                                                               |
+-------------+------------------------+------------------------------------------------------------------------+
| FAC         | building name                                                                                   |
+-------------+------------------------+------------------------------------------------------------------------+
| CARDINAL    | cardinal value                                                                                  |
+-------------+------------------------+------------------------------------------------------------------------+
| PRODUCT     | product name                                                                                    |
+-------------+------------------------+------------------------------------------------------------------------+
| TIME        | time value                                                                                      |
+-------------+------------------------+------------------------------------------------------------------------+
| STREET      | street name                                                                                     |
+-------------+------------------------+------------------------------------------------------------------------+

.. raw:: html

   </details>

**Model Output**: one or several of the 30 relations found between the given entities; a Russian relation name (e.g. "участник") or an English one, if Russian one is unavailable, and, if applicable, its id in `Wikidata <https://www.wikidata.org/wiki/Wikidata:Main_Page>`__ (e.g. 'P710').

.. raw:: html

   <details>
   <summary><a>Full list of Russian relation</a></summary>

+----------------------------+-------------------+---------------------------------+
| Relation                   | Relation id       | Russian relation                |
+----------------------------+-------------------+---------------------------------+
| MEMBER                     | P710              | участник                        |
+----------------------------+-------------------+---------------------------------+
| WORKS_AS                   | P106              | род занятий                     |
+----------------------------+-------------------+---------------------------------+
| WORKPLACE                  |                   |                                 |
+----------------------------+-------------------+---------------------------------+
| OWNERSHIP                  | P1830             | владеет                         |
+----------------------------+-------------------+---------------------------------+
| SUBORDINATE_OF             | -                 | -                               |
+----------------------------+-------------------+---------------------------------+
| TAKES_PLACE_IN             | P276              | местонахождение                 |
+----------------------------+-------------------+---------------------------------+
| EVENT_TAKES_PART_IN        | P1344             | участвовал в                    |
+----------------------------+-------------------+---------------------------------+
| SELLS_TO                   | -                 | -                               |
+----------------------------+-------------------+---------------------------------+
| ALTERNATIVE_NAME           | -                 | -                               |
+----------------------------+-------------------+---------------------------------+
| HEADQUARTERED_IN           | P159              | расположение штаб-квартиры      |
+----------------------------+-------------------+---------------------------------+
| PRODUCES                   | P1056             | продукция                       |
+----------------------------+-------------------+---------------------------------+
| ABBREVIATION               | -                 | -                               |
+----------------------------+-------------------+---------------------------------+
| DATE_DEFUNCT_IN            | P576              | дата прекращения существования  |
+----------------------------+-------------------+---------------------------------+
| SUBEVENT_OF                | P361              | часть от                        |
+----------------------------+-------------------+---------------------------------+
| DATE_FOUNDED_IN            | P571              | дата основания/создания/возн-я  |
+----------------------------+-------------------+---------------------------------+
| DATE_TAKES_PLACE_ON        | P585              | момент времени                  |
+----------------------------+-------------------+---------------------------------+
| NUMBER_OF_EMPLOYEES_FIRED  | -                 | -                               |
+----------------------------+-------------------+---------------------------------+
| ORIGINS_FROM               | P495              | страна происхождения            |
+----------------------------+-------------------+---------------------------------+
| ACQUINTANCE_OF             | -                 | -                               |
+----------------------------+-------------------+---------------------------------+
| PARENT_OF                  | P40               | дети                            |
+----------------------------+-------------------+---------------------------------+
| ORGANIZES                  | P664              | организатор                     |
+----------------------------+-------------------+---------------------------------+
| FOUNDED_BY                 | P112              | основатель                      |
+----------------------------+-------------------+---------------------------------+
| PLACE_RESIDES_IN           | P551              | место жительства                |
+----------------------------+-------------------+---------------------------------+
| BORN_IN                    | P19               | место рождения                  |
+----------------------------+-------------------+---------------------------------+
| AGE_IS                     | -                 | -                               |
+----------------------------+-------------------+---------------------------------+
| RELATIVE                   | -                 | -                               |
+----------------------------+-------------------+---------------------------------+
| NUMBER_OF_EMPLOYEES        | P1128             | число сотрудников               |
+----------------------------+-------------------+---------------------------------+
| SIBLING                    | P3373             | брат/сестра                     |
+----------------------------+-------------------+---------------------------------+
| DATE_OF_BIRTH              | P569              | дата рождения                   |
+----------------------------+-------------------+---------------------------------+

.. raw:: html

   </details>

.. raw:: html

   <details>
   <summary><a>Some details on RuRED corpus Russian RE model was trained on</a></summary>

In case of RuRED we used the train, dev and test sets from the original RuRED setting. We additionally generate negative samples if it was necessary to have the following proportions:

- for train set: negative samples are twice as many as positive ones
- for dev & test set: negative samples are the same amount as positive ones

+---------------+---------------+----------------+
| Train         | Dev           | Test           |
+---------------+---------------+----------------+
| 12855         | 1076          |1072            |
+---------------+---------------+----------------+

+---------------+----------------+----------------+----------------+----------------+----------------+
| Train Positive| Train Negative | Dev Positive   | Dev Negative   | Test Positive  | Test Negative  |
+---------------+----------------+----------------+----------------+----------------+----------------+
| 4285          | 8570           | 538            | 538            | 536            | 536            |
+---------------+----------------+----------------+----------------+----------------+----------------+

.. raw:: html

   </details>


RE Model Architecture
-----------------------
We based our model on the `Adaptive Thresholding and Localized Context Pooling <https://arxiv.org/pdf/2010.11304.pdf>`__ model and used NER entity tags as additional input. Two core ideas of this model are:

- Adaptive Threshold

The usual global threshold for converting the RE classifier output probability to relation label is replaced with a learnable one. A new threshold class that learns an entities-dependent threshold value is introduced and learnt as all other classes. During prediction the positive classes (= relations that are hold in the sample indeed) are claimed to be the classes with higher logins that the TH class, while all others are negative ones.

- Localised Context Pooling

The embedding of each entity pair is enhanced with an additional local context embedding related to both entities. Such representation, which is attended to the relevant context in the document, is useful to decide the relation for exactly this entity pair. For incorporating the context information the attention heads are directly used.

