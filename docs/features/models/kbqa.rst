Knowledge Base Question Answering (KBQA)
========================================

Description
-----------

There are three models for KBQA in DeepPavlov library:

* :config:`kbqa_cq <kbqa/kbqa_cq.json>` - for answering complex questions in English

* :config:`kbqa_cq <kbqa/kbqa_cq_online.json>` - for answering complex questions in English using Wikidata Query Service

* :config:`kbqa_rus <kbqa/kbqa_cq_rus.json>` - for answering complex questions in Russian

The Complex Knowledge Base Question Answering model uses Wikidata to answer complex questions. Types of questions which the model answers:

* Complex questions with numerical values:
    "What position did Angela Merkel hold on November 10, 1994?"

* Complex question where the answer is number or date:
    "When did Jean-Paul Sartre move to Le Havre?"

* Questions with counting of answer entities:
    "How many sponsors are for Juventus F.C.?"

* Questions with ordering of answer entities by ascending or descending of some parameter:
    "Which country has highest individual tax rate?"

* Simple questions:
    "What is crew member Yuri Gagarin's Vostok?"

To find the answer the following
models are used:

* BERT model for prediction of query template type. Model performs classification of questions into 8 classes correponding to 8 query template types.

* BERT entity detection model for extraction of entity substrings from the questions. 

* Substring extracted by the entity detection model is used for entity linking. Entity linking performs matching the substring
  with one of the Wikidata entities. Matching is based on Levenshtein distance between the substring and an entity
  title. The result of the matching procedure is a set of candidate entities. The reset is search of the
  entity among this set with one of the top-k relations predicted by classification model.

* BiGRU model for ranking of candidate relations.

* BERT model for ranking of candidate relation paths.

* Query generator model is used to fill query template with candidate entities and relations (to find valid combinations of entities and relations for query template). Query Generation model uses Wikidata HDT file. Query Generation Online model uses Wikidata Query Service.

Use the model
-------------

Any pre-trained model can be used for inference from both Command Line Interface (CLI) and Python. Before using the
model make sure that all required packages are installed using the command:

.. code:: bash

    python -m deeppavlov install kbqa_cq
    python -m deeppavlov install kbqa_cq_online
    python -m deeppavlov install kbqa_cq_rus

To use a pre-trained model from CLI use the following command:

.. code:: bash

    python deeppavlov/deep.py interact kbqa_сq [-d]
    python deeppavlov/deep.py interact kbqa_cq_online [-d]
    python deeppavlov/deep.py interact kbqa_cq_rus [-d]

where ``kbqa_cq`` and others are the names of configs and ``-d`` is an optional download key. The key ``-d`` is used
to download the pre-trained model along with embeddings and all other files needed to run the model. Also command
``download`` is possible,



KBQA model for complex question answering can be used from Python using the following code:

.. code:: python

    from deeppavlov import configs, build_model

    kbqa_model = build_model(configs.kbqa.kbqa_cq, download=True)
    kbqa_model(['What is in the village of Negev that has diplomatic relations with the Czech Republic?'])
    >>> ["Israel"]
    kbqa_model(['Magnus Carlsen is a part of what sport?'])
    >>> ["chess"]
    kbqa_model(['How many sponsors are for Juventus F.C.?'])
    >>> [4]

In the models mentioned above lite version of Wikidata is used. Full version of Wikidata can be downloaded from http://www.rdfhdt.org/datasets/. Examples of questions which the model can answer with the following version of Wikidata:

.. code:: python

    from deeppavlov import configs, build_model

    kbqa_model = build_model(configs.kbqa.kbqa_cq, download=True)
    kbqa_model(['When did Jean-Paul Sartre move to Le Havre?'])
    >>> ["1931-01-01"]
    kbqa_model(['What position did Angela Merkel hold on November 10, 1994?'])
    >>> ["Bundesminister"]

KBQA model for complex question answering in Russian can be used from Python using the following code:

.. code:: python

    from deeppavlov import configs, build_model

    kbqa_model = build_model(configs.kbqa.kbqa_cq_rus, download=True)
    kbqa_model(['Когда родился Пушкин?'])
    >>> ["1799-05-26"]

Train models
------------

Models, trained for complex question answering:

* :config:`query_pr <classifiers/query_pr.json>` - classification model for prediction of query template type

* :config:`entity_detection <ner/ner_lcquad_bert_ent_and_type.json>` - sequence tagging model for detection of entity and entity types substrings in the question

* :config:`rel_ranking <ranking/rel_ranking.json>` - model for ranking of candidate relations for the question

* :config:`rel_ranking_bert <classifiers/rel_ranking_bert.json>` - model for ranking of candidate relation paths for the question

Training of Query Prediction
----------------------------

The dataset consists of three csv files: train.csv, valid.csv and test.csv. Each line in this file contains question and corresponding query template type, for example:

"What is the longest river in the UK?", 6

Training of Entity Detection
----------------------------

The dataset is a pickle file. The dataset must be split into three parts: train, test, and validation. Each part is a list of tuples of question tokens and tags for each token. An example of training sample:

('What', 'is', 'the', 'complete', 'list', 'of', 'records', 'released', 'by', 'Jerry', 'Lee', 'Lewis', '?'], ['O-TAG', 'O-TAG', 'O-TAG', 'O-TAG', 'T-TAG', 'T-TAG', 'T-TAG', 'O-TAG', 'O-TAG', 'E-TAG', 'E-TAG', 'E-TAG', 'O-TAG'])

"T-TAG" corresponds to tokens of entity types, "E-TAG" - for entities, "O-TAG" - for other tokens.

Training of Relation and Path Ranking
-------------------------------------

The dataset for relation ranking consists of two xml files (train and test sets). Each sample contains a question, a relation title and a label (1 if the relation corresponds to the question and 0 otherwise). An example of training sample:

.. code:: xml

    <paraphrase>
       <value name="text_1">Is it true that the total shots in career of Rick Adduono is equal to 1?</value>
       <value name="text_2">total shots in career</value>
       <value name="class">1</value>
    </paraphrase>

The dataset for path ranking is similar to the dataset for relation ranking. If the path from the grounded entity in the question and the answer consists of two relations, relation titles are separated with "#":

.. code:: xml

    <paraphrase>
       <value name="text_1">When did Thomas Cromwell end his position as Lord Privy Seal?</value>
       <value name="text_2">position held # end time</value>
       <value name="class">1</value>
    </paraphrase>

Adding new SPARQL queries templates
-----------------------------------------
Templates can be added to sparql_queries.json file, which is a dictionary, where keys are template numbers, and values are templates with additional information.
An example of a template::

    {"query_template": "SELECT ?obj WHERE { wd:E1 p:R1 ?s . ?s ps:R1 ?obj . ?s ?p ?x filter(contains(?x, N)) }",
     "property_types": {"?p": "qualifier"},
     "rank_rels": ["wiki", "do_not_rank", "do_not_rank"],
     "rel_types": ["no_type", "statement", "qualifier"],
     "filter_rels": [false],
     "rel_dirs": ["forw"],
     "query_sequence": [1, 2, 3],
     "entities_and_types_num": [1, 0],
     "entities_and_types_select": "1 0",
     "syntax_structure": {"gr_ent": 1, "types": 0, "mod_ent": 0, "q_ent": 0, "count": false, "order": false},
     "return_if_found": true,
     "template_num": "0",
     "alternative_templates": []}

* "query_template" is the template of the SPARQL query
* "property_types" defines the types of unknown relations in the template
* "rank_rels" is a list which defines whether to rank relations, in this example "p:R1" relations we extract from Wikidata for "wd:E1" entities and rank with RelRanker, "ps:R1" and "?p" relations we do not extract and rank
* "rel_types" - direct, statement or qualifier relations
* "filter_rels" (only for online version of KBQA) - whether candidate rels will be enumerated in the "filter" expression in the query, for example
  "SELECT ?ent WHERE { ?ent wdt:P31 wd:Q4022 . ?ent ?p1 wd:Q90 } filter(?p1 = wdt:P131 || ?p1 = wdt:P17)"
* "rel_dirs" - "forw" if the relation connects the subject and unknown object, for example, "wd:Q649 wdt:P17 ?p", "backw" if the relation connects the unknown object and the subject, for example "?p wdt:P17 wd:Q159"
* "query_sequence" (only for offline version of KBQA) - the sequence in which the triplets will be extracted from Wikidata hdt file
* "entities_and_types_num" - numbers of entities and types extracted from the question, which this template can contain
* "entities_and_types_select" - the dictionary where keys are number of entities and types extracted from the question and values are indices of entities and types which should be filled in the template (because we can extract more entities and types than the template contains)
* "syntax_structure" - information about syntactic structure of questions corresponding to this query
* "return_if_found" - parameter for the cycle which iterates over all possible combinations of entities, relations and types, if "true" - return if the first valid combination is found, if "false" - consider all combinations
* "template_num" - the number of template
* alternative_templates - numbers of alternative templates to use if the answer was not found with the current template
