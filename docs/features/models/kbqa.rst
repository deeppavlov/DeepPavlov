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

Currently, we provide three built-in models for KBQA in DeepPavlov library:

* :config:`kbqa_cq <kbqa/kbqa_cq.json>` - for answering complex questions over Wikidata in English,

* :config:`kbqa_rus <kbqa/kbqa_cq_rus.json>` - for answering complex questions over Wikidata in Russian,

* :config:`kbqa_cq_online <kbqa/kbqa_cq_online.json>` - for answering complex questions in English over Wikidata using Wikidata Query Service.

The first two models are very similar to each other, and they allow you to deploy them together with local copy of Wikidata on-premises or in the cloud. The third model is lightweight as it allows you to skip downloading entire Wikidata and use the existing Wikidata APIs instead. 

.. note:: 
   We recommend you to use the lightweight model for quick experiments as well as academic research, and full models in production to avoid dependencies on the public Wikidata APIs.

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

* BiGRU model for ranking of candidate relations,

* BERT model for ranking of candidate relation paths,

* Query generator model is used to fill query template with candidate entities and relations (to find valid combinations of entities and relations for query template). Query Generation model uses Wikidata HDT file. Query Generation Online model uses Wikidata Query Service.

How Do I: Using KBQA In CLI & Python
------------------------------------

Any pre-trained model in DeepPavlov Library can be used for inference from both Command Line Interface (CLI) and Python. Before using the model make sure that all required packages are installed using the command:

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
to download the pre-trained model along with embeddings and all other files needed to run the model. You can also use command ``download``.

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

How Do I: Train KBQA Model
--------------------------
Here are the models we've trained for complex question answering:

* :config:`query_pr <classifiers/query_pr.json>` - classification model for prediction of query template type,

* :config:`entity_detection <ner/ner_lcquad_bert_ent_and_type.json>` - sequence tagging model for detection of entity and entity types substrings in the question,

* :config:`rel_ranking <ranking/rel_ranking.json>` - model for ranking of candidate relations for the question,

* :config:`rel_ranking_bert <classifiers/rel_ranking_bert.json>` - model for ranking of candidate relation paths for the question.

How Do I: Train Query Prediction Model
--------------------------------------

The dataset consists of three csv files: train.csv, valid.csv and test.csv. Each line in this file contains question and corresponding query template type, for example::

"What is the longest river in the UK?", 6

How Do I: Train Entity Detection Model
--------------------------------------

The dataset is a pickle file. The dataset must be split into three parts: train, test, and validation. Each part is a list of tuples of question tokens and tags for each token. An example of training sample::

 (['What', 'is', 'the', 'complete', 'list', 'of', 'records', 'released', 'by', 'Jerry', 'Lee', 'Lewis', '?'],
  ['O-TAG', 'O-TAG', 'O-TAG', 'O-TAG', 'T-TAG', 'T-TAG', 'T-TAG', 'O-TAG', 'O-TAG', 'E-TAG', 'E-TAG', 'E-TAG', 'O-TAG'])

``T-TAG`` corresponds to tokens of entity types, ``E-TAG`` - for entities, ``O-TAG`` - for other tokens.

How Do I: Train Relation and Path Ranking Models
------------------------------------------------

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
* ``rank_rels`` is a list which defines whether to rank relations, in this example **p:R1** relations we extract from Wikidata for **wd:E1** entities and rank with RelRanker, **ps:R1** and **?p** relations we do not extract and rank,
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

Advanced: Using Entity Linking and Wiki Parser As Standalone Services For KBQA
------------------------------------------------------------------------------
Default configuration for KBQA was designed to use all of the supporting models together as a part of the KBQA pipeline. However, there might be a case when you want to work with some of these models in addition to KBQA.

For example, you might want to use Entity Linking as an annotator in your `Deepy-based <https://github.com/deepmipt/assistant-base>`_ multiskill AI Assistant. Or, you might want to use Wiki Parser component to directly run SPARQL queries against your copy of Wikidata. To support these usecase, starting with this release you can also deploy supporting models as standalone components.

Config :config:`kbqa_entity_linking <kbqa/kbqa_entity_linking.json>` can be used as service with the following command:

.. code:: bash

    python -m deeppavlov riseapi kbqa_entity_linking [-d] [-p <port>]
    
Arguments:

* ``entity_substr`` - batch of lists of entity substrings for which we want to find ids in Wikidata,
* ``template`` - template of the sentence (if the sentence with the entity matches of one of templates),
* ``context`` - text with the entity.

.. code:: python

    import requests

    payload = {"entity_substr": [["Forrest Gump"]], "template": [""], "context": ["Who directed Forrest Gump?"]}
    response = requests.post(entity_linking_url, json=payload).json()
    print(response)

    
    
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

To use Entity Linking service in KBQA, in the :config:`kbqa_cq_sep <kbqa/kbqa_cq_sep.json>` you should use add to ``pipe`` API Requester component::

    {
        "class_name": "api_requester",
        "id": "linker_entities",
        "url": "entity_linking_url",
        "out": ["entity_ids"],
        "param_names": ["entity_substr", "template_found"]
     }

and replace ``linker_entities`` parameter value of the :config:`query_generator <kbqa/kbqa_cq_sep.json#L71>` component with ``#linker_entities``::

    "linker_entities": "#linker_entities",
    
To use Wiki Parser service in KBQA, in the :config:`kbqa_cq_sep <kbqa/kbqa_cq_sep.json>` you should add to ``pipe`` API Requester component::

    {
        "class_name": "api_requester",
        "id": "wiki_p",
        "url": "wiki_parser_url",
        "out": ["wiki_parser_output"],
        "param_names": ["parser_info", "query"]
     }

and replace ``wiki_parser`` parameter value of the :config:`query_generator <kbqa/kbqa_cq_sep.json#L75>` and :config:`rel_ranking_bert_infer <kbqa/kbqa_cq_sep.json#L88>` components with ``#wiki_p``::

    "wiki_parser": "#wiki_p",

.. warning::
    Don't forget to replace the ``url`` parameter values in the above examples with correct URLs