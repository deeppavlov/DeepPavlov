Go-Bot Framework
################

Overview
********

Go-Bot is an ML-driven framework designed to enable development of the goal-oriented skills for
DeepPavlov Dream AI Assistant Platform.

These goal-oriented skills can be written in Python (enabling using their corresponding Go-Bot-trained models natively)
or in any other programming language (requiring running their corresponding Go-Bot-trained models as microservices).

To build a Go-Bot-based goal-oriented skill, you need to provide Go-Bot framework with a dataset (in RASA v1 format),
train model, download it, and then use it by either calling them natively from Python or by rising them as microservices
and then calling them via its standard DeepPavlov REST API.

Currently we support a subset of the v1 of the RASA DSLs (domain.yml, nlu.md, stories.md) to define domain model and
behavior of a given goal-oriented skill. As of the latest release, the following subset of functionality is supported:

* Intents
* Slots (simple slots requiring custom classifiers for custom data types)
* Stories (w/o 1:1 mapping between intents and responses)
* Templated Responses (w/o variables)
* **Form-Filling** (basic, added in **v0.14 release**)

In the future, we will expand support for RASA DSLs where appropriate to enable backward compatibility, add integration
with the upcoming Intent Catcher component available as part of the DeepPavlov component library, and so on.

To experiment with the Go-Bot you can follow tutorials for using RASA DSLs.

RASA DSLs Format Support
************************

Overview
========
To simplify the process of building goal-oriented bots using DeepPavlov technology,
in `v0.12.0 <https://deeppavlov.ai/blog/tpost/58y1cugd7b-deeppavlov-library-0120-release>`_ we have introduced
a (limited) support for defining them using RASA DSLs.

.. note::
  DSLs, known as Domain-Specific Languages, provide a rich mechanism to define the behavior, or "the what", while the underlying system uses the parser to transform these definitions into commands that implement this behavior, or "the how" using the system's components.

RASA.ai is an another well-known Open Source Conversational AI Framework. Their approach to defining the domain model
and behavior of the goal-oriented bots is quite simple for building simple goal-oriented bots.
In this section you will learn how to use key parts of RASA DSLs (configuration files)
to build your own goal-oriented skill based on the DeepPavlov's Go-Bot framework.

While there are several configuration files used by the RASA platform, each with their own
corresponding DSL (mostly re-purposed Markdown and YAML), for now only three essential files: ``stories.md``,
``nlu.md``, ``domain.yml`` are supported by the DeepPavlov Go-Bot Framework.

These files allows you to define user stories that match intents and bot actions, intents with slots and entities,
as well as the training data for the NLU components.

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

We encourage you to explore the tutorials below to get better understanding of how to build basic and more advanced
goal-oriented skills with these RASA DSLs:

* `Original Tutorial Notebook Featuring Simple and DSTC2-based Skills <https://github.com/deepmipt/DeepPavlov/blob/master/examples/gobot_md_yaml_configs_tutorial.ipynb>`_

* `Tutorial Notebook Featuring Harvesters Maintenance Go-Bot Skill from Deepy 3000 Demo <https://colab.research.google.com/drive/1BdTnDsytEABOU7RbNRQqIVE-rBHOv0kM?usp=sharing>`_


How Do I: Integrate Go-Bot-based Goal-Oriented Skill into DeepPavlov Deepy
============================================================================

To integrate your Go-Bot-based goal-oriented skill into your Multiskill AI Assistant built
using DeepPavlov Conversational AI Stack, follow the following instructions:

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

To configure your own pipelines that contain a ``"go_bot"`` component, refer to documentation
for :class:`~deeppavlov.models.go_bot.bot.GoalOrientedBot`
and :class:`~deeppavlov.models.go_bot.network.GoalOrientedBotNetwork` classes.

Database (Optional)
=====================

If your dataset doesn't imply any api calls to an external database, just do not set
``database`` and ``api_call_action`` parameters and skip the section below.

Otherwise, you should

1. provide sql table with requested items or
2. construct such table from provided in train samples ``db_result`` items.

.. |alt text| image:: ../../_static/gobot_diagram.png
