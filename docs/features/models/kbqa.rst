Knowledge Base Question Answering (KBQA)
========================================

Overview
-----------

The knowledge base:

* is a comprehensive repository of information about a given domain or a number of domains,
* reflects the ways we model knowledge about a given subject or subjects, in terms of concepts, entities, properties, and relationships,
* enables us to use this structured knowledge where appropriate, e.g., answering factoid questions

Currently, we support Wikidata as a Knowledge Base (Knowledge Graph). In the future, we will expand support for more knowledge graphs, as well as outline the steps needed to support a custom knowledge base.

The question answerer:

* validates questions against a preconfigured list of question templates, disambiguates entities using Entity Linking, and answers questions asked in natural language,
* can be used with Wikidata (English, Russian) and (in the future versions) with custom knowledge graphs.

Built-In Models
------------------

Currently, we provide two built-in models for KBQA in DeepPavlov library:

* :config:`kbqa_cq_en <kbqa/kbqa_cq_en.json>` - for answering complex questions over Wikidata in English,

* :config:`kbqa_cq_ru <kbqa/kbqa_cq_ru.json>` - for answering complex questions over Wikidata in Russian,

These configs use local Wikidata dump in hdt format (3.7 Gb on disk).

+--------------------------------------------------+-----------+-----------+
| Model                                            |  RAM, Gb  |  GPU, Gb  |
+==================================================+===========+===========+
| :config:`kbqa_cq_en <kbqa/kbqa_cq_en.json>`      |    3.5    |    4.3    |
+--------------------------------------------------+-----------+-----------+
| :config:`kbqa_cq_ru <kbqa/kbqa_cq_ru.json>`      |    6.9    |    6.5    |
+--------------------------------------------------+-----------+-----------+

The Knowledge Base Question Answering model uses Wikidata to answer complex questions. Here are some of the most popular types of questions supported by the model:

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

The following models are used to find the answer:

* BERT model for prediction of query template type. Model performs classification of questions into 8 classes correponding to 8 query template types,

* BERT entity detection model for extraction of entity substrings from the questions, 

* Substring extracted by the entity detection model is used for entity linking. Entity linking performs matching the substring
  with one of the Wikidata entities. Matching is based on Levenshtein distance between the substring and an entity
  title. The result of the matching procedure is a set of candidate entities. The reset is search of the
  entity among this set with one of the top-k relations predicted by classification model,

* BERT model for ranking of candidate relations and candidate relation paths,

* Query generator model is used to fill query template with candidate entities and relations (to find valid combinations of entities and relations for query template). Query Generation model uses Wikidata HDT file. Query Generation Online model uses Wikidata Query Service.

How Do I: Using KBQA In CLI & Python
------------------------------------

Any pre-trained model in DeepPavlov Library can be used for inference from both Command Line Interface (CLI) and Python. Before using the model make sure that all required packages are installed using the command:

.. code:: bash

    python -m deeppavlov install kbqa_cq_en
    python -m deeppavlov install kbqa_cq_ru

To use a pre-trained model from CLI use the following command:

.. code:: bash

    python deeppavlov/deep.py interact kbqa_сq_en [-d]
    python deeppavlov/deep.py interact kbqa_cq_ru [-d]

where ``kbqa_cq_en`` and others are the names of configs and ``-d`` is an optional download key. The key ``-d`` is used
to download the pre-trained model along with embeddings and all other files needed to run the model. You can also use command ``download``.

KBQA model for complex question answering can be used from Python using the following code:

.. code:: python

    from deeppavlov import build_model

    kbqa_model = build_model('kbqa_cq_en', download=True)
    kbqa_model(['What is the currency of Sweden?'])
    >>> ["Swedish krona"]
    kbqa_model(['Magnus Carlsen is a part of what sport?'])
    >>> ["chess"]
    kbqa_model(['How many sponsors are for Juventus F.C.?'])
    >>> [4]
    kbqa_model(['When did Jean-Paul Sartre move to Le Havre?'])
    >>> ["1931-01-01"]
    kbqa_model(['What position did Angela Merkel hold on November 10, 1994?'])
    >>> ["Bundesminister"]

KBQA model for complex question answering in Russian can be used from Python using the following code:

.. code:: python

    from deeppavlov import build_model

    kbqa_model = build_model('kbqa_cq_ru', download=True)
    kbqa_model(['Когда родился Пушкин?'])
    >>> ["1799-05-26"]

How Do I: Train KBQA Model
--------------------------
Here are the models we've trained for complex question answering:

* :config:`query_pr <classifiers/query_pr.json>` - classification model for prediction of query template type,

* :config:`entity_detection <ner/ner_ontonotes_bert.json>` - sequence tagging model for detection of entity and entity types substrings in the question,

* :config:`rel_ranking <ranking/rel_ranking_bert_en.json>` - model for ranking of candidate relations and candidate_relation_paths for the question,

How Do I: Train Query Prediction Model
--------------------------------------

The dataset (in pickle format) is a dict of three keys: "train", "valid" and "test". The value by each key is the list of samples, an example of a sample:

"What is the longest river in the UK?", 6

How Do I: Train Entity Detection Model
--------------------------------------

The dataset is a pickle file. The dataset must be split into three parts: train, test, and validation. Each part is a list of tuples of question tokens and tags for each token. An example of training sample::

 (['What', 'is', 'the', 'complete', 'list', 'of', 'records', 'released', 'by', 'Jerry', 'Lee', 'Lewis', '?'],
  ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'I-PER', 'I-PER', 'O'])

The tags of tokens correspond to BIO-markup.

How Do I: Train Relation and Path Ranking Models
------------------------------------------------

The dataset (in pickle format) is a dict of three keys: "train", "valid" and "test". The value by each key is the list of samples, an example of a sample::

 (['What is the Main St. Exile label, which Nik Powell co-founded?', ['record label', 'founded by']], '1')
 
The sample contains the question, relations in the question and label (1 - if the relations correspond to the question, 0 - otherwise).

How Do I: Adding Templates For New SPARQL Queries
-------------------------------------------------
Templates can be added to sparql_queries.json file, which is a dictionary, where keys are template numbers, and values are templates with additional information.
An example of a template::

    {
        "query_template": "SELECT ?obj WHERE { wd:E1 p:R1 ?s . ?s ps:R1 ?obj . ?s ?p ?x filter(contains(?x, N)) }",
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
        "alternative_templates": []
     }

* ``query_template`` is the template of the SPARQL query,
* ``property_types`` defines the types of unknown relations in the template,
* ``rank_rels`` is a list which defines whether to rank relations, in this example **p:R1** relations we extract from Wikidata for **wd:E1** entities and rank with rel_ranker, **ps:R1** and **?p** relations we do not extract and rank,
* ``rel_types`` - direct, statement or qualifier relations,
* ``filter_rels`` (only for online version of KBQA) - whether candidate rels will be enumerated in the **filter** expression in the query, for example,
  **SELECT ?ent WHERE { ?ent wdt:P31 wd:Q4022 . ?ent ?p1 wd:Q90 } filter(?p1 = wdt:P131 || ?p1 = wdt:P17)**,
* ``rel_dirs`` - **forw** if the relation connects the subject and unknown object, for example, **wd:Q649 wdt:P17 ?p**, **backw** if the relation connects the unknown object and the subject, for example **?p wdt:P17 wd:Q159**,
* ``query_sequence`` (only for offline version of KBQA) - the sequence in which the triplets will be extracted from Wikidata hdt file,
* ``entities_and_types_num`` - numbers of entities and types extracted from the question, which this template can contain,
* ``entities_and_types_select`` - the dictionary where keys are number of entities and types extracted from the question and values are indices of entities and types which should be filled in the template (because we can extract more entities and types than the template contains),
* ``syntax_structure`` - information about syntactic structure of questions corresponding to this query,
* ``return_if_found`` - parameter for the cycle which iterates over all possible combinations of entities, relations and types, if **true** - return if the first valid combination is found, if **false** - consider all combinations,
* ``template_num`` - the number of template,
* alternative_templates - numbers of alternative templates to use if the answer was not found with the current template.

Advanced: Using Wiki Parser As Standalone Service For KBQA
------------------------------------------------------------------------------
Default configuration for KBQA was designed to use all of the supporting models together as a part of the KBQA pipeline. However, there might be a case when you want to work with some of these models in addition to KBQA.

For example, you might want to use Wiki Parser component to directly run SPARQL queries against your copy of Wikidata. To support these usecase, starting with this release you can also deploy supporting models as standalone components.

Config :config:`wiki_parser <kbqa/wiki_parser.json>` can be used as service with the following command:

.. code:: bash

    python -m deeppavlov riseapi wiki_parser [-d] [-p <port>]
    
Arguments of the annotator are ``parser_info`` (what we want to extract from Wikidata) and ``query``.

Examples of queries:

To extract triplets for entities, the ``query`` argument should be the list of entities ids and ``parser_info`` - list of "find\_triplets" strings.

.. code:: python

    requests.post(wiki_parser_url, json = {"parser_info": ["find_triplets"], "query": ["Q159"]}).json()


To extract all relations of the entities, the ``query`` argument should be the list of entities ids and ``parser_info`` - list of "find\_rels" strings.

.. code:: python

    requests.post(wiki_parser_url, json = {"parser_info": ["find_rels"], "query": ["Q159"]}).json()


To execute SPARQL queries, the ``query`` argument should be the list of tuples with the info about SPARQL queries and ``parser_info`` - list of "query\_execute" strings.

Let us consider an example of the question "What is the deepest lake in Russia?" with the corresponding SPARQL query
``SELECT ?ent WHERE { ?ent wdt:P31 wd:T1 . ?ent wdt:R1 ?obj . ?ent wdt:R2 wd:E1 } ORDER BY ASC(?obj) LIMIT 5``

Arguments:

* ``what_return``: ["?obj"],
* ``query_seq``: [["?ent", "P17", "Q159"], ["?ent", "P31", "Q23397"], ["?ent", "P4511", "?obj"]],
* ``filter_info``: [],
* ``order_info``: order\_info(variable='?obj', sorting_order='asc').

.. code:: python

    requests.post("wiki_parser_url", json = {"parser_info": ["query_execute"], "query": [[["?obj"], [["Q159", "P36", "?obj"]], [], [], True]]}).json()


To find labels for entities ids, the ``query`` argument should be the list of entities ids and ``parser_info`` - list of "find\_label" strings.

.. code:: python

    requests.post(wiki_parser_url, json = {"parser_info": ["find_label"], "query": [["Q159", ""]]}).json()


In the example in the list ["Q159", ""] the second element which is an empty string can be the string with the sentence.

To use Entity Linking service in KBQA, in the :config:`kbqa_cq_en <kbqa/kbqa_cq_en.json>` you should replace :config:`entity linking component <kbqa/kbqa_cq_en.json#L24>` with API Requester component in the following way::

    {
        "class_name": "api_requester",
        "id": "entity_linker",
        "url": "entity_linking_url",
        "out": ["entity_ids"],
        "param_names": ["entity_substr", "template_found"]
     }
    
To use Wiki Parser service in KBQA, in the :config:`kbqa_cq_en <kbqa/kbqa_cq_en.json>` you should replace :config:`wiki parser component <kbqa/kbqa_cq_en.json#L28>` with API Requester component in the following way::

    {
        "class_name": "api_requester",
        "id": "wiki_p",
        "url": "wiki_parser_url",
        "out": ["wiki_parser_output"],
        "param_names": ["parser_info", "query"]
     }

.. warning::
    Don't forget to replace the ``url`` parameter values in the above examples with correct URLs
