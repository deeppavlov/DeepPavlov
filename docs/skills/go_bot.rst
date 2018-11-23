Dialogue Bot for goal-oriented task
===================================

The dialogue bot is based on `[1] <#references>`__ which introduces
Hybrid Code Networks (HCNs) that combine an RNN with domain-specific
knowledge and system action templates.

|alt text| **Diagram 1.** Dotted lines correspond to unrequired
(optional) modules, black squares to trained models, trapezes are
modules that depend on a dataset and must be provided by software
developer.

Here is a simple example of interaction with a trained dialogue bot
(available for download):

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

Usage
-----

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

Configs:
^^^^^^^^

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
--------

DSTC2
^^^^^

The Hybrid Code Network model was trained and evaluated on a modification of a dataset from Dialogue State Tracking
Challenge 2 `[2] <#references>`__. The modifications were as follows:

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
^^^^^^^^^

Dialogs
""""""'

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

Database (optional)
""""""""""""""""""'

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
----------

Scores for different modifications of our bot model:

+-----------------------------------------------+----------------------------------------------------------------------+----------------------------+
| Model                                         | Config                                                               | Test turn textual accuracy |
+===============================================+======================================================================+============================+
| basic bot                                     | :config:`gobot_dstc2_minimal.json <go_bot/gobot_dstc2_minimal.json>` | 0.3809                     |
+-----------------------------------------------+----------------------------------------------------------------------+----------------------------+
| bot with slot filler & fasttext embeddings    |                                                                      | 0.5317                     |
+-----------------------------------------------+----------------------------------------------------------------------+----------------------------+
| bot with slot filler & intents                | :config:`gobot_dstc2.json <go_bot/gobot_dstc2.json>`                 | 0.5248                     |
+-----------------------------------------------+----------------------------------------------------------------------+----------------------------+
| bot with slot filler & intents & embeddings   |                                                                      | 0.5145                     |
+-----------------------------------------------+----------------------------------------------------------------------+----------------------------+
| bot with slot filler & embeddings & attention | :config:`gobot_dstc2_best.json <go_bot/gobot_dstc2_best.json>`       | **0.5551**                 |
+-----------------------------------------------+----------------------------------------------------------------------+----------------------------+

There is another modification of DSTC2 dataset called dialog babi Task6
`[3] <#references>`__. It differs from ours in train/valid/test split and
intent/action labeling.

These are the test scores provided by Williams et al. (2017) `[1] <#references>`__
(can't be directly compared with above):

+----------------------------------------------------+------------------------------+
|                   Model                            | Test turn textual accuracy   |
+====================================================+==============================+
| Bordes and Weston (2016) `[4] <#references>`__     |   0.411                      |
+----------------------------------------------------+------------------------------+
| Perez and Liu (2016) `[5] <#references>`__         |   0.487                      |
+----------------------------------------------------+------------------------------+
| Eric and Manning (2017) `[6] <#references>`__      |   0.480                      |
+----------------------------------------------------+------------------------------+
| Williams et al. (2017) `[1] <#references>`__       |   0.556                      |
+----------------------------------------------------+------------------------------+

TODO: add dialog accuracies

References
----------

[1] `Jason D. Williams, Kavosh Asadi, Geoffrey Zweig "Hybrid Code
Networks: practical and efficient end-to-end dialog control with
supervised and reinforcement learning" –
2017 <https://arxiv.org/abs/1702.03274>`_

[2] `Dialog State Tracking Challenge 2
dataset <http://camdial.org/~mh521/dstc/>`_

[3] `The bAbI project <https://research.fb.com/downloads/babi/>`_

[4] `Antoine Bordes, Y-Lan Boureau & Jason Weston "Learning end-to-end
goal-oriented dialog" - 2017 <https://arxiv.org/abs/1605.07683>`_

[5] `Fei Liu, Julien Perez "Gated End-to-end Memory Networks" -
2016 <https://arxiv.org/abs/1610.04211>`_

[6] `Mihail Eric, Christopher D. Manning "A Copy-Augmented
Sequence-to-Sequence Architecture Gives Good Performance on
Task-Oriented Dialogue" - 2017 <https://arxiv.org/abs/1701.04024>`_


.. |alt text| image:: ../_static/gobot_diagram.png
