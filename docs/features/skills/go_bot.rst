Go-Bot Framework
################

Overview
********

Go-Bot is an ML-driven framework designed to enable development of the goal-oriented skills for DeepPavlov Dream AI Assistant Platform.

These goal-oriented skills can be written in Python (enabling using their corresponding Go-Bot-trained models natively) or in any other programming language (requiring running their corresponding Go-Bot-trained models as microservices).

To build a Go-Bot-based goal-oriented skill, you need to provide Go-Bot framework with a dataset (in RASA v1 or DSTC2 formats), train model, download it, and then use it by either calling them natively from Python or by rising them as microservices and then calling them via its standard DeepPavlov REST API.

Currently, we support two different approaches to define domain model and behavior of a given goal-oriented skill - using either a subset of the v1 of the RASA DSLs (domain.yml, nlu.md, stories.md) or a DSTC2 format. As of the latest release, the following subset of functionality is supported:

* Intents
* Slots (simple slots requiring custom classifiers for custom data types)
* Stories (w/o 1:1 mapping between intents and responses)
* Templated Responses (w/o variables)
* **Form-Filling** (basic, added in **v0.14 release**)

In the future, we will expand support for RASA DSLs where appropriate to enable backward compatibility, add integration with the upcoming Intent Catcher component available as part of the DeepPavlov component library, and so on.

To experiment with the Go-Bot you can follow tutorials for using RASA DSLs, or pick one of the two available pre-trained models designed around the DSTSC2 dataset (English).

RASA DSLs Format Support
************************

Overview
========
While DSTC-2 schema format is quite rich, preparing this kind of dataset with all required annotations might be challenging. To simplify the process of building goal-oriented bots using DeepPavlov technology, in `v0.12.0 <https://deeppavlov.ai/blog/tpost/58y1cugd7b-deeppavlov-library-0120-release>`_ we have introduced a (limited) support for defining them using RASA DSLs.

.. note::
  DSLs, known as Domain-Specific Languages, provide a rich mechanism to define the behavior, or "the what", while the underlying system uses the parser to transform these definitions into commands that implement this behavior, or "the how" using the system's components.

RASA.ai is an another well-known Open Source Conversational AI Framework. Their approach to defining the domain model and behavior of the goal-oriented bots is quite simple for building simple goal-oriented bots. In this section you will learn how to use key parts of RASA DSLs (configuration files) to build your own goal-oriented skill based on the DeepPavlov's Go-Bot framework.



While there are several configuration files used by the RASA platform, each with their own corresponding DSL (mostly re-purposed Markdown and YAML), for now only three essential files: ``stories.md``,
``nlu.md``, ``domain.yml`` are supported by the DeepPavlov Go-Bot Framework.

These files allows you to define user stories that match intents and bot actions, intents with slots and entities, as well as the training data for the NLU components.

.. note::
   As mentioned in our `blog post <https://deeppavlov.ai/blog/tpost/58y1cugd7b-deeppavlov-library-0120-release>`__, **this is the very beginning of our work** focused on supporting RASA DSLs as a way to configure DeepPavlov-based goal-oriented chatbots.

Currently, only a subset of the functionality in these files is supported by now.

stories.md
^^^^^^^^^^

``stories.md`` is a mechanism used to teach your chatbot how to respond
to user messages. It allows you to control your chatbot's dialog
management.

The full RASA functionality is described in the `original
documentation <https://rasa.com/docs/rasa/core/stories/>`__.

The format supported by DeepPavlov is the subset of features described
in `"What makes up a story"
section <https://rasa.com/docs/rasa/core/stories/#what-makes-up-a-story>`__.

The original format features are: *User Messages*, *Actions*, *Events*,
*Checkpoints*, *OR Statements*, *End-to-End Story Evaluation Format*.

-  We **do support** all the functionality of User Messages format
   feature.

-  We **do support only** utterance actions of the Actions format
   feature. Custom actions are **not supported yet**.

-  We **do partially support** Form Filling (starting with v0.14.0 release).

-  We **do not support** Events, Checkpoints and OR Statements format
   features.

format
""""""

see the `original
documentation <https://rasa.com/docs/rasa/core/stories/>`__ for the
detailed ``stories.md`` format description.

Stories file is a markdown file of the following format:

.. code:: md

   ## story_title (not used by algorithm, but useful to work with for humans)
   * user_action_label{"1st_slot_present_in_action": "slot1_value", .., "Nth_slot_present_in_action": "slotN_value"}
     - system_respective_utterance
   * another_user_action_of_the_same_format
     - another_system_response
   ...

   ## another_story_title
   ...
    
   ## formfilling dialogue
   * greet
     - form{"name": "zoo_form"}
     - utter_api_call


nlu.md
^^^^^^

``nlu.md`` represents an NLU model of your chatbot. It allows you to
provide training examples that show how your chatbot should
understand user messages, and then train a model through these
examples.

We do support the format described in the `Markdown
format <https://rasa.com/docs/rasa/nlu/training-data-format/#markdown-format>`__
section of the original RASA documentation with the following
limitations:

-  an extended entities annotation format
   (``[<entity-text>]{"entity": "<entity name>", "role": "<role name>", ...}``)
   is **not supported**
-  *synonyms*, *regex features* and *lookup tables* format features are
   **not supported**

format
""""""

see the `original
documentation <https://rasa.com/docs/rasa/nlu/training-data-format/>`__
on the RASA NLU markdown format for the detailed ``nlu.md`` format
description.

NLU file is a markdown file of the following format:

.. code:: md

   ## intent:possible_user_action_label_1
     - An example of user text that has the possible_user_action_label_1 action label
     - Another example of user text that has the possible_user_action_label_1 action label
     ...
   
   ## intent:possible_user_action_label_N
     - An example of user text that has the (possible_user_action_label_N)[action_label] action label
     <!-- Slotfilling dataset is provided as an inline markup of user texts -->
    ...

   
domain.yml
^^^^^^^^^^

``domain.yml`` helps you to define the universe your chatbot lives in:
what user inputs it expects to get, what actions it should be able to
predict,
how to respond, and what information to store.

The format supported by DeepPavlov is the same as the described in the
`original documentation <https://rasa.com/docs/rasa/core/domains/>`__
with the following limitations:

-  only textual slots are allowed
-  only slot classes are allowed as entity classes
-  only textual response actions are allowed with currently no variables
   support

format
""""""

see the `original
documentation <https://rasa.com/docs/rasa/core/domains/>`__ on the RASA
Domains YAML config format for the detailed ``domain.yml`` format
description.

Domain file is a YAML file of the following format:

.. code:: yaml

   # slots section lists the possible slot names (aka slot types) 
   # that are used in the domain (i.e. relevant for bot's tasks)
   # currently only type: text is supported
   slots:
     slot1_name:
       type: text
       ...
     slotN_name:
       type: text

   # entities list now follows the slots list 2nd level keys 
   # and is present to support upcoming features. Stay tuned for updates with this!
   entities:
     - slot1_name
     ...
     - slotN_name

   # intents section lists the intents that can appear in the stories
   # being kept together they do describe the user-side part of go-bot's experience
   intents:
     - user_action_label
     - another_user_action_of_the_same_format
     ...

   # responses section lists the system response templates.
   # Despite system response' titles being usually informative themselves
   #   (one could even find them more appropriate when no actual "Natural Language" is needed 
   #    (e.g. for buttons actions in bot apps))
   # It is though extremely useful to be able to serialize the response title to text. 
   # That's what this section content is needed for.
   responses:
     system_utterance_1:
       - text: "The text that system responds with"
     another_system_response:
       - text: "Here some text again"

   forms:
     zoo_form:
       animal:
         - type: from_entity
           entity: animal

How Do I: Build Go-Bot Skill with RASA DSLs (v1)
================================================

Tutorials
^^^^^^^^^

We encourage you to explore the tutorials below to get better understanding of how to build basic and more advanced goal-oriented skills with these RASA DSLs:

* `Original Tutorial Notebook Featuring Simple and DSTC2-based Skills <https://github.com/deepmipt/DeepPavlov/blob/master/examples/gobot_md_yaml_configs_tutorial.ipynb>`_

* `Tutorial Notebook Featuring Harvesters Maintenance Go-Bot Skill from Deepy 3000 Demo <https://colab.research.google.com/drive/1BdTnDsytEABOU7RbNRQqIVE-rBHOv0kM?usp=sharing>`_


How Do I: Integrate Go-Bot-based Goal-Oriented Skill into DeepPavlov Deepy
============================================================================

To integrate your Go-Bot-based goal-oriented skill into your Multiskill AI Assistant built using DeepPavlov Conversational AI Stack, follow the following instructions:

1. Clone `Deepy repository <https://github.com/deepmipt/assistant-base>`_
2. Replace ``docker-compose.yml`` in the root of the repository and ``pipeline_conf.json`` in the ``/agent/`` subdirectory with the corresponding files from the `deepy_gobot_base <https://github.com/deepmipt/assistant-base/tree/main/assistant_dists/deepy_gobot_base>`_ **Deepy Distribution**
3. Clone the second `Tutorial Notebook <https://colab.research.google.com/drive/1BdTnDsytEABOU7RbNRQqIVE-rBHOv0kM?usp=sharing>`_
4. Change its ``domain.yml``, ``nlu.md``, and ``stories.md`` based on your project needs with your custom **intents**, **slots**, **forms**, and write your own **stories**
5. Train the go-bot model in your copy of the Tutorial Notebook
6. Download and put saved data from your copy of the Tutorial Notebook into the `Harvesters Maintenance Go-Bot Skill <https://github.com/deepmipt/assistant-base/tree/main/skills/harvesters_maintenance_gobot_skill>`_ 
7. [Optional] Unless you need a Chit-Chat skill remove `it <https://github.com/deepmipt/assistant-base/tree/main/skills/program-y>`_ from at both the ``/agent/pipeline_conf.json`` and from ``docker-compose.yml``
8. Use ``docker-compose up --build`` command to build and run your DeepPavlov-based Multiskill AI Assistant

.. note::
   In the coming version of the DeepPavlov Library we will provide a more comprehensive update to the documentation to further simplify the process of building goal-oriented skills with DeepPavlov Conversational AI technology stack. Stay tuned!

How Do I: Use Form-Filling in Go-Bot Skill with RASA DSLs (v1)
================================================================

Tutorials
^^^^^^^^^

Follow this tutorial to experiment with the Form-Filling functionality in Go-Bot-based goal-oriented skills built using RASA DSLs (v1):

* `Tutorial Notebook Featuring Basic Form-Filling <https://github.com/deepmipt/DeepPavlov/blob/feature/gobot_naive_formfilling/examples/gobot_formfilling_tutorial.ipynb>`_


DSTC2 Format Support
**********************

Overview
==========

The DeepPavlov Go-Bot Framework is based on [1]_ which introduces
Hybrid Code Networks (HCNs) that combine an RNN with domain-specific
knowledge and system action templates. Originally, the DSTC2 format was used for the dataset to train a Go-Bot-based goal-oriented skills upon.

|alt text| **Diagram 1.** Dotted lines correspond to unrequired
(optional) modules, black squares to trained models, trapezes are
modules that depend on a dataset and must be provided by software
developer.

Here is a simple example of interaction with a trained goal-oriented skill
(available for download):

.. note::

    Models that rely on fasttext word embeddings will require 8.5 GB of disk space, those that use only one-hot encoding of words will require less than 300Mb of disk space.

.. code:: bash

    >> Hello, welcome to the Cambridge restaurant system. You can ask for restaurants by area, price range or food type. How may I help you?
    x::hi, i want cheap food
    2018-05-16 12:28:48.459 INFO in 'deeppavlov.models.go_bot.bot'['bot'] at line 346: Made api_call with {'pricerange': 'cheap'}, got 22 results.
    >> The lucky star serves chinese food in the cheap price range.
    x::is there anything with french food?
    2018-05-16 12:29:02.795 INFO in 'deeppavlov.models.go_bot.bot'['bot'] at line 346: Made api_call with {'pricerange': 'cheap', 'food': 'french'}, got 0 results.
    >> Sorry there is no french restaurant in the cheap price range.
    x::ok, maybe there is something in expensive pricerange?
    2018-05-16 12:29:23.337 INFO in 'deeppavlov.models.go_bot.bot'['bot'] at line 346: Made api_call with {'pricerange': 'expensive', 'food': 'french'}, got 2 results.
    >> Cote serves french food in the expensive price range.
    x::in what part of town is it?
    2018-05-16 12:29:31.193 INFO in 'deeppavlov.models.go_bot.bot'['bot'] at line 346: Made api_call with {'pricerange': 'expensive', 'food': 'french'}, got 2 results.
    >> Restaurant two two is a nice place in the north of town serving tasty french food.
    x::ok, give me their phone number
    >> The phone number of restaurant two two is 01223 351880.
    x::address and post code too
    >> Restaurant two two is on 22 chesterton road chesterton, c.b 4, 3 a.x.
    x::ok, thank you!
    >> Restaurant two two serves french food in the expensive price range.
    x::and if i wanted an italian restuarant?
    2018-05-16 12:30:15.98 INFO in 'deeppavlov.models.go_bot.bot'['bot'] at line 346: Made api_call with {'pricerange': 'expensive', 'food': 'italian'}, got 4 results.
    >> Frankie and bennys serves italian food in the expensive price range.
    x::more cheaply priced?
    2018-05-16 12:30:36.560 INFO in 'deeppavlov.models.go_bot.bot'['bot'] at line 346: Made api_call with {'pricerange': 'cheap', 'food': 'italian'}, got 5 results.
    >> Da vinci pizzeria serves italian food in the cheap price range.
    x::ok, give me their phone number and address, please
    >> The phone number of da vinci pizzeria is 01223 351707 and it is on 20 milton road chesterton.
    x::thanks
    >> Da vinci pizzeria serves italian food in the cheap price range.
    x::bye
    >> You are welcome!


Quick Demo
============

To quickly try out the Go-Bot capabilities you can use one of the two available pretrained models for DSTC2 dataset (English). Check them out by running this code:

.. code:: python

    from deeppavlov import build_model, configs

    bot1 = build_model(configs.go_bot.gobot_dstc2, download=True)

    bot1(['hi, i want restaurant in the cheap pricerange'])
    bot1(['bye'])

    bot2 = build_model(configs.go_bot.gobot_dstc2_best, download=True)

    bot2(['hi, i want chinese restaurant'])
    bot2(['bye'])

If some required packages are missing, install all the requirements by running in command line:

.. code:: bash

   python -m deeppavlov install gobot_dstc2

How Do I: Build Go-Bot with DSTC2
===================================
DSTC is a set of competitions originally known as "Dialog State Tracking Challenges" (DSTC, for short). First challenge
was organized in 2012-2013. Starting as an initiative to provide a common testbed for the task of Dialog State Tracking,
the first Dialog State Tracking Challenge (DSTC) was organized in 2013, followed by DSTC2&3 in 2014, DSTC4 in 2015,
and DSTC5 in 2016. Given the remarkable success of the first five editions, and understanding both, the complexity
of the dialog phenomenon and the interest of the research community in a wider variety of dialog related problems,
the DSTC rebranded itself as "Dialog System Technology Challenges" for its sixth edition. Then, DSTC6 and DSTC7 have
been completed in 2017 and 2018, respectively.

DSTC-2 released a large number of training dialogs related to restaurant search. Compared to DSTC (which was in the bus
timetables domain), DSTC 2 introduced changing user goals, tracking 'requested slots' as well as the new Restaurants domain.

Historically, DeepPavlov's Go-Bot used this DSTC-2 approach to defining domain model and behavior of the goal-oriented bots.
In this section you will learn how to use this approach to build a DSTC-2-based Go-Bot.

Requirements
^^^^^^^^^^^^

**TO TRAIN** a go\_bot model you should have:

1. (*optional, but recommended*) pretrained named entity recognition model (NER)

   - config :config:`configs/ner/slotfill_dstc2.json <ner/slotfill_dstc2.json>` is recommended
2. (*optional, but recommended*) pretrained intents classifier model

   - config :config:`configs/classifiers/intents_dstc2_big.json <classifiers/intents_dstc2_big.json>` is recommended
3. (*optional*) any sentence (word) embeddings for english

   - fasttext embeddings can be downloaded

      - via link https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip
      - or using deeppavlov with :code:`python3 -m deeppavlov download <path_to_config>`,
        where ``<path_to_config>`` is one of the :config:`provided config files <go_bot>`.

**TO INFER** from a go\_bot model you should **additionally** have:

4. pretrained vocabulary of dataset utterance tokens

   - it is trained in the same config as go\_bot model

5. pretrained goal-oriented bot model
   
   - config :config:`configs/go_bot/gobot_dstc2.json <go_bot/gobot_dstc2.json>` is recommended
   - ``slot_filler`` section of go\_bot's config should match NER's configuration
   - ``intent_classifier`` section of go\_bot's config should match classifier's configuration

Configs
^^^^^^^

For a working exemplary config see
:config:`configs/go_bot/gobot_dstc2.json <go_bot/gobot_dstc2.json>` (model without embeddings).

A minimal model without ``slot_filler``, ``intent_classifier`` and ``embedder`` is configured
in :config:`configs/go_bot/gobot_dstc2_minimal.json <go_bot/gobot_dstc2_minimal.json>`.

The best state-of-the-art model (with attention mechanism, relies on ``embedder`` and
does not use bag-of-words) is configured in
:config:`configs/go_bot/gobot_dstc2_best.json <go_bot/gobot_dstc2_best.json>`.

Usage example
^^^^^^^^^^^^^

To interact with a pretrained go\_bot model using commandline run:

.. code:: bash

    python -m deeppavlov interact <path_to_config> [-d]

where ``<path_to_config>`` is one of the :config:`provided config files <go_bot>`.

You can also train your own model by running:

.. code:: bash

    python -m deeppavlov train <path_to_config> [-d]

The ``-d`` parameter downloads

   - data required to train your model (embeddings, etc.);
   - a pretrained model if available (provided not for all configs). 

**Pretrained for DSTC2** models are available for

   - :config:`configs/go_bot/gobot_dstc2.json <go_bot/gobot_dstc2.json>` and
   - :config:`configs/go_bot/gobot_dstc2.json <go_bot/gobot_dstc2_best.json>`.

After downloading required files you can use the configs in your python code.
To infer from a pretrained model with config path equal to ``<path_to_config>``:

.. code:: python

    from deeppavlov import build_model

    CONFIG_PATH = '<path_to_config>'
    model = build_model(CONFIG_PATH)

    utterance = ""
    while utterance != 'exit':
        print(">> " + model([utterance])[0])
        utterance = input(':: ')

Config parameters
^^^^^^^^^^^^^^^^^

To configure your own pipelines that contain a ``"go_bot"`` component, refer to documentation for :class:`~deeppavlov.models.go_bot.bot.GoalOrientedBot` and :class:`~deeppavlov.models.go_bot.network.GoalOrientedBotNetwork` classes.

Datasets
========

.. _dstc2_dataset:

DSTC2
^^^^^

The Hybrid Code Network model was trained and evaluated on a modification of a dataset from Dialogue State Tracking
Challenge 2 [2]_. The modifications were as follows:

-  **new turns with api calls**

   -  added api\_calls to restaurant database (example:
      ``{"text": "api_call area=\"south\" food=\"dontcare\" pricerange=\"cheap\"", "dialog_acts": ["api_call"]}``)

-  **new actions**

   -  bot dialog actions were concatenated into one action (example:
      ``{"dialog_acts": ["ask", "request"]}`` ->
      ``{"dialog_acts": ["ask_request"]}``)
   -  if a slot key was associated with the dialog action, the new act
      was a concatenation of an act and a slot key (example:
      ``{"dialog_acts": ["ask"], "slot_vals": ["area"]}`` ->
      ``{"dialog_acts": ["ask_area"]}``)

-  **new train/dev/test split**

   -  original dstc2 consisted of three different MDP policies, the original train
      and dev datasets (consisting of two policies) were merged and
      randomly split into train/dev/test

-  **minor fixes**

   -  fixed several dialogs, where actions were wrongly annotated
   -  uppercased first letter of bot responses
   -  unified punctuation for bot responses

See :class:`deeppavlov.dataset_readers.dstc2_reader.DSTC2DatasetReader` for implementation.

Your data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dialogs
"""""""

If your model uses DSTC2 and relies on ``"dstc2_reader"``
(:class:`~deeppavlov.dataset_readers.dstc2_reader.DSTC2DatasetReader`),
all needed files, if not present in the
:attr:`DSTC2DatasetReader.data_path <deeppavlov.dataset_readers.dstc2_reader.DSTC2DatasetReader.data_path>` directory,
will be downloaded from web.

If your model needs to be trained on different data, you have several ways of
achieving that (sorted by increase in the amount of code):

1. Use ``"dialog_iterator"`` in dataset iterator config section and
   ``"dstc2_reader"`` in dataset reader config section
   (**the simplest, but not the best way**):

   -  set ``dataset_reader.data_path`` to your data directory;
   -  your data files should have the same format as expected in
      :meth:`DSTC2DatasetReader.read() <deeppavlov.dataset_readers.dstc2_reader.DSTC2DatasetReader.read>`
      method.

2. Use ``"dialog_iterator"`` in dataset iterator config section and
   ``"your_dataset_reader"`` in dataset reader config section (**recommended**):

   -  clone :class:`deeppavlov.dataset_readers.dstc2_reader.DSTC2DatasetReader` to
      ``YourDatasetReader``;
   -  register as ``"your_dataset_reader"``;
   -  rewrite so that it implements the same interface as the origin.
      Particularly, ``YourDatasetReader.read()`` must have the same output as
      :meth:`DSTC2DatasetReader.read() <deeppavlov.dataset_readers.dstc2_reader.DSTC2DatasetReader.read>`.
   
      -  ``train`` — training dialog turns consisting of tuples:
      
         -  first tuple element contains first user's utterance info
            (as dictionary with the following fields):

            -  ``text`` — utterance string
            -  ``intents`` — list of string intents, associated with user's utterance
            -  ``db_result`` — a database response *(optional)*
            -  ``episode_done`` — set to ``true``, if current utterance is
               the start of a new dialog, and ``false`` (or skipped) otherwise *(optional)*

         -  second tuple element contains second user's response info

            -  ``text`` — utterance string
            -  ``act`` — an act, associated with the user's utterance

      -  ``valid`` — validation dialog turns in the same format
      -  ``test`` — test dialog turns in the same format

3. Use your own dataset iterator and dataset reader (**if 2. doesn't work for you**):

   -  your ``YourDatasetIterator.gen_batches()`` class method output should match the
      input format for chainer from
      :config:`configs/go_bot/gobot_dstc2.json <go_bot/gobot_dstc2.json>`.

Templates
"""""""""

You should provide a maping from actions to text templates in the format

.. code:: text

    action1<tab>template1
    action2<tab>template2
    ...
    actionN<tab>templateN

where filled slots in templates should start with "#" and mustn't contain whitespaces.

For example,

.. code:: text

    bye You are welcome!
    canthear  Sorry, I can't hear you.
    expl-conf_area  Did you say you are looking for a restaurant in the #area of town?
    inform_area+inform_food+offer_name  #name is a nice place in the #area of town serving tasty #food food.

It is recommended to use ``"DefaultTemplate"`` value for ``template_type`` parameter.


Database (Optional)
=====================

If your dataset doesn't imply any api calls to an external database, just do not set
``database`` and ``api_call_action`` parameters and skip the section below.

Otherwise, you should

1. provide sql table with requested items or
2. construct such table from provided in train samples ``db_result`` items.
   This can be done with the following script:


   .. code:: bash

       python -m deeppavlov train configs/go_bot/database_<your_dataset>.json

   where ``configs/go_bot/database_<your_dataset>.json`` is a copy
   of ``configs/go_bot/database_dstc2.json`` with configured
   ``save_path``, ``primary_keys`` and ``unknown_value``.

Comparison
************

Scores for different modifications of our bot model and comparison with existing benchmarks:

+-----------------------------------+------+------------------------------------------------------------------------------------+---------------+-----------+---------------+
| Dataset                           | Lang | Model                                                                              | Metric        | Test      | Downloads     |
+===================================+======+====================================================================================+===============+===========+===============+
| `DSTC 2`_                         | En   | :config:`basic bot <go_bot/gobot_dstc2_minimal.json>`                              | Turn Accuracy | 0.380     | 10 Mb         |
+ (:ref:`modified <dstc2_dataset>`) +      +------------------------------------------------------------------------------------+               +-----------+---------------+
|                                   |      | :config:`bot with slot filler <go_bot/gobot_dstc2.json>`                           |               | 0.542     | 400 Mb        |
+                                   +      +------------------------------------------------------------------------------------+               +-----------+---------------+
|                                   |      | :config:`bot with slot filler, intents & attention <go_bot/gobot_dstc2_best.json>` |               | **0.553** | 8.5 Gb        |
+-----------------------------------+      +------------------------------------------------------------------------------------+               +-----------+---------------+
| `DSTC 2`_                         |      | Bordes and Weston (2016) [3]_                                                      |               | 0.411     | --            |
+                                   +      +------------------------------------------------------------------------------------+               +-----------+---------------+
|                                   |      | Eric and Manning (2017) [4]_                                                       |               | 0.480     | --            |
+                                   +      +------------------------------------------------------------------------------------+               +-----------+---------------+
|                                   |      | Perez and Liu (2016) [5]_                                                          |               | 0.487     | --            |
+                                   +      +------------------------------------------------------------------------------------+               +-----------+---------------+
|                                   |      | Williams et al. (2017) [1]_                                                        |               | **0.556** | --            |
+-----------------------------------+------+------------------------------------------------------------------------------------+---------------+-----------+---------------+

.. _`DSTC 2`: http://camdial.org/~mh521/dstc/

References
************

.. [1] `Jason D. Williams, Kavosh Asadi, Geoffrey Zweig "Hybrid Code
    Networks: practical and efficient end-to-end dialog control with
    supervised and reinforcement learning" –
    2017 <https://arxiv.org/abs/1702.03274>`_

.. [2] `Dialog State Tracking Challenge 2
    dataset <http://camdial.org/~mh521/dstc/>`_

.. [3] `Antoine Bordes, Y-Lan Boureau & Jason Weston "Learning end-to-end
    goal-oriented dialog" - 2017 <https://arxiv.org/abs/1605.07683>`_

.. [4] `Mihail Eric, Christopher D. Manning "A Copy-Augmented
    Sequence-to-Sequence Architecture Gives Good Performance on
    Task-Oriented Dialogue" - 2017 <https://arxiv.org/abs/1701.04024>`_

.. [5] `Fei Liu, Julien Perez "Gated End-to-end Memory Networks" -
    2016 <https://arxiv.org/abs/1610.04211>`_


.. |alt text| image:: ../../_static/gobot_diagram.png
