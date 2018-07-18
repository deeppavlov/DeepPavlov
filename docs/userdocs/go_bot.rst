Dialogue Bot for goal-oriented task
===================================

The dialogue bot is based on `[1] <#references>`__ which introduces
Hybrid Code Networks (HCNs) that combine an RNN with domain-specific
knowledge and system action templates.

|alt text| **Diagram 1.** Dotted lines correspond to unrequired
(optional) modules, black squares to trained models, trapezes are
modules that depend on a dataset and must be provided by software
developer.

Here is a simple example of interaction with a trained dialogue bot (can
be downloaded with ``deeppavlov/download.py``):

.. code::

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
   ``deeppavlov/configs/ner/ner_dstc2.json`` is recommended
2. (*optional, but recommended*) pretrained intents classifier model
   ``deeppavlov/configs/intents/intents_dstc2_big.json`` is recommended
3. (*optional*) downloaded english fasttext embeddings trained on wiki
   (https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip)\: fasttext embeddings can be loaded via
   ``python3 deeppavlov/download.py -all`` you can use any english embeddings of your choice, but edit go\_bot config
   accordingly.

**TO INFER** from a go\_bot model you should **additionaly** have:

4. pretrained vocabulary of dataset utterance tokens

-  it is trained in the same config as go\_bot model

5. pretrained goal-oriented bot model itself

-  config ``deeppavlov/configs/go_bot/gobot_dstc2.json`` is recommended
-  ``slot_filler`` section of go\_bot's config should match NER's configuration
-  ``intent_classifier`` section of go\_bot's config should match classifier's configuration
-  double-check that corresponding ``load_path`` point to NER and intent classifier model files

Config parameters:
^^^^^^^^^^^^^^^^^^

-  ``name`` always equals to ``"go_bot"``
-  ``word_vocab`` — vocabulary of tokens from context utterances

    +  ``name`` — ``"default_vocab"`` (for vocabulary's implementation see ``deeppavlov.core.data.vocab``)
    +  ``level`` — ``"token"``
    +  ``tokenizer`` — ``{ "name": "split_tokenizer" }``
    +  ``save_path`` — ``"vocabs/token.dict"``
    +  ``load_path`` — ``"vocabs/token.dict"``


-  ``template_path`` — map from actions to text templates for response generation
-  ``template_type`` — type of templates to use (``"BaseTemplate"`` by default) *(optional)*
-  ``database`` – database that will be used during model inference to make "api\_call" action and
   get ``db_result`` *(optional)*

   +  ``name`` — ``"sqlite_database"`` from
      :doc:`deeppavlov.core.data.sqlite_database:Sqlite3Database </apiref/core/data>` or your implementation
   +  ``table_name`` – sqlite table name
   +  ``primary_keys`` – list of primary table keys' names
   +  ``keys`` – ordered list of table key names, if not set will be inferred from loaded database automatically
      *(optional, recommended not to be used)*
   +  ``unknown_value`` – value used to fill unknown column values (defaults to ``"UNK"``) *(optional)*
   +  ``save_path`` – path to database filename (will load to it, and save to it)


-  ``api_call_action`` – label of action that corresponds to database api call (the same label that is used
   to represent the action in your ``template_path`` file), during interaction it will be used to get ``db_result``
   from ``database`` *(optional)*
-  ``use_action_mask`` — if ``true``, action mask is applied to network output *(False, by default)*
-  ``tokenizer`` — one of tokenizers from ``deeppavlov.models.tokenizers`` module

   +  ``name`` — tokenizer name
   +  other arguments specific to your tokenizer


-  ``bow_embedder`` — ``deeppavlov.models.embedders.bow_embedder`` or ``null`` *(optional)*

   +  ``name`` — embedder name
   +  other arguments specific to your bag of words embedder


-  ``embedder`` — one of embedders from ``deeppavlov.models.embedders`` module *(optional)*

   +  ``name`` — embedder name (``"fasttext"`` recommended, see ``deeppavlov.models.embedders.fasttext_embedder``)
   +  other arguments specific to your embedder


-  ``tracker`` — dialogue state tracker from ``deeppavlov.models.trackers``

   +  ``name`` — tracker name (``"default_tracker"`` or ``"featurized_tracker"`` recommended)
   +  ``slot_vals`` — list of slots that should be tracked


-  ``network parameters`` - see :doc:`GoalOrientedBotNetwork </apiref/models/go_bot>` for details.
-  ``slot_filler`` — model that predicts slot values for a given utterance

   +  ``name`` — slot filler name (``"dstc_slotfilling"`` recommended, for implementation see ``deeppavlov.models.ner``)
   +  other slot filler arguments


-  ``intent_classifier`` — model that outputs intents probability distribution for a given utterance

   +  ``name`` — intent classifier name (``"intent_model"`` recommended, for implementation
      see ``deeppavlov.models.classifiers.intents``)
   +  classifier's other arguments


-  ``debug`` — whether to display debug output (defaults to ``false``) *(optional)*

For a working exemplary config see ``deeeppavlov/configs/go_bot/gobot_dstc2.json`` (model without embeddings).

A minimal model without ``slot_filler``, ``intent_classifier`` and ``embedder`` is configured
in ``deeeppavlov/configs/go_bot/gobot_dstc2_minimal.json``.

A full model (with fasttext embeddings) configuration is in ``deeeppavlov/configs/go_bot/gobot_dstc2_all.json``.

The best state-of-the-art model (with attention mechanism, relies on ``embedder`` and does not use bag-of-words) is
configured in ``deeeppavlov/configs/go_bot/gobot_dstc2_best.json``.

Usage example
^^^^^^^^^^^^^

Available **pretrained for DSTC2 dataset** models:

-  model for ``deeppavlov/configs/go_bot/gobot_dstc2.json``
-  model for ``deeppavlov/configs/go_bot/gobot_dstc2_best.json``

To use pretrained model you should firstly **download it** (if you haven't done it already
by ``python3 deeppavlov/download.py -all``):

.. code:: bash

    cd deeppavlov
    python3 deep.py interact path/to/config.json -d

To infer from a pretrained model with config path equal to ``path/to/config.json``:

.. code:: python

    from deeppavlov.core.commands.infer import build_model_from_config
    from deeppavlov.core.common.file import read_json

    CONFIG_PATH = 'path/to/config.json'
    model = build_model_from_config(read_json(CONFIG_PATH))

    utterance = ""
    while utterance != 'exit':
        print(">> " + model([utterance])[0])
        utterance = input(':: ')

To interact via command line use ``deeppavlov/deep.py`` script:

.. code:: bash

    cd deeppavlov
    python3 deep.py interact path/to/config.json

Training
--------

To train model with config path ``path/to/config.json`` you should firstly **download** all the needed data
(if you haven't done it already by ``python3 deeppavlov/download.py -all``):

.. code:: bash

    cd deeppavlov
    python3 deep.py train path/to/config.json -d

The script will download needed data (dataset, embeddings) for the particular model.

Config parameters
^^^^^^^^^^^^^^^^^

To be used for training, your config json file should include parameters:

-  ``dataset_reader``
-  ``name`` — ``"your_reader_here"`` for a custom dataset or ``"dstc2_v2_reader"`` to use DSTC2 (for implementation
   see ``deeppavlov.dataset_readers.dstc2_reader``)
-  ``data_path`` — a path to a dataset file, which in case of ``"dstc2_v2_reader"`` will be automatically downloaded
   from internet and placed to ``data_path`` directory
-  ``dataset_iterator`` — it should always be set to ``{"name": "dialog_iterator"}`` (for implementation
   see ``deeppavlov.dataset_iterators.dialog_iterator.py``)

See ``deeeppavlov/configs/go_bot/gobot_dstc2.json`` for details.

Train run
^^^^^^^^^

The easiest way to run the training is by using ``deeppavlov/deep.py`` script:

.. code:: bash

    cd deeppavlov
    python3 deep.py train path/to/config.json

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

   -  original dstc2 consisted of three different MDP polices, the original train and dev datasets (consisting of
      two polices) were merged and randomly split into train/dev/test

-  **minor fixes**

   -  fixed several dialogs, where actions were wrongly annotated
   -  uppercased first letter of bot responses
   -  unified punctuation for bot responses

Your data
^^^^^^^^^

Dialogs
'''''''

If your model uses DSTC2 and relies on ``dstc2_v2_reader`` (``DSTC2Version2DatasetReader``), all needed files, if not
present in the ``dataset_reader.data_path`` directory, will be downloaded from internet.

If your model needs to be trained on different data, you have several ways of achieving that (sorted by increase
in the amount of code):

1. Use ``"dialog_iterator"`` in dataset iterator config section and ``"dstc2_v2_reader"`` in dataset reader config
   section (**the simplest, but not the best way**):

   -  set ``dataset_iterator.data_path`` to your data directory;
   -  your data files should have the same format as expected in
      ``deeppavlov.dataset_readers.dstc2_reader:DSTC2Version2DatasetReader.read()`` function.


2. Use ``"dialog_iterator"`` in dataset iterator config section and ``"your_dataset_reader"`` in dataset reader config
   section (**recommended**):

   -  clone ``deeppavlov.dataset_readers.dstc2_reader:DSTC2Version2DatasetReader`` to ``YourDatasetReader``;
   -  register as ``"your_dataset_reader"``;
   -  rewrite so that it implements the same interface as the origin. Particularly, ``YourDatasetReader.read()`` must
      have the same output as ``DSTC2DatasetReader.read()``:
   -  ``train`` — training dialog turns consisting of tuples:

      -  first tuple element contains first user's utterance info (as dict with the following fields):

         -  ``text`` — utterance string
         -  ``intents`` — list of string intents, associated with user's utterance
         -  ``db_result`` — a database response *(optional)*
         -  ``episode_done`` — set to ``true``, if current utterance is the start of a new dialog, and ``false``
            (or skipped) otherwise *(optional)*

      -  second tuple element contains second user's response info

         -  ``text`` — utterance string
         -  ``act`` — an act, associated with the user's utterance

   -  ``valid`` — validation dialog turns in the same format
   -  ``test`` — test dialog turns in the same format


3. Use your own dataset iterator and dataset reader (**if 2. doesn't work for you**):

   -  your ``YourDatasetIterator.gen_batches()`` class method output should match the input format for chainer from
      ``configs/go_bot/gobot_dstc2.json``.

Templates
'''''''''

You should provide a maping from actions to text templates in the following format (and set ``template_type`` to
``"BaseTemplate"``, DSTC2 uses an extension of templates –``"DualTemplate"``, you will probably not need it):
``action_template``, where filled slots in templates should start with "#" and mustn't contain whitespaces.

For example,

.. code::

    bye You are welcome!
    canthear  Sorry, I can't hear you.
    expl-conf_area  Did you say you are looking for a restaurant in the #area of town?
    inform_area+inform_food+offer_name  #name is a nice place in the #area of town serving tasty #food food.

Database (optional)
'''''''''''''''''''

If your dataset doesn't imply any api calls to an external database, just do not set ``database`` and
``api_call_action`` parameters and skip the section below.

Otherwise, you should specify them and

1. provide sql table with requested items or
2. construct such table from provided in train samples ``db_result`` items. This can be done with the following script:


.. code:: bash

    cd deeppavlov
    python3 deep.py train configs/go_bot/database_yourdataset.json

where ``configs/go_bot/database_yourdataset.json`` is a copy of ``configs/go_bot/database_dstc2.json`` with configured
``save_path``, ``primary_keys`` and ``unknown_value``.

Comparison
----------

Scores for different modifications of our bot model:

+-------------------------------------------------+------------------------------------------------------------------------------------+------------------------------+
| Model                                           | Config                                                                             | Test turn textual accuracy   |
+=================================================+====================================================================================+==============================+
| basic bot                                       |  ``gobot_dstc2_minimal.json``                                                      | 0.3809                       |
+-------------------------------------------------+------------------------------------------------------------------------------------+------------------------------+
| bot with slot filler & fasttext embeddings      |                                                                                    | 0.5317                       |
+-------------------------------------------------+------------------------------------------------------------------------------------+------------------------------+
| bot with slot filler & intents                  |  ``gobot_dstc2.json``                                                              | 0.5113                       |
+-------------------------------------------------+------------------------------------------------------------------------------------+------------------------------+
| bot with slot filler & intents & embeddings     |  ``gobot_dstc2_all.json``                                                          | 0.5145                       |
+-------------------------------------------------+------------------------------------------------------------------------------------+------------------------------+
| bot with slot filler & embeddings & attention   |  ``gobot_dstc2_best.json``                                                         | **0.5525**                   |
+-------------------------------------------------+------------------------------------------------------------------------------------+------------------------------+

There is another modification of DSTC2 dataset called dialog babi Task6 `[3] <#references>`__. It differs from ours
in train/valid/test split and intent/action labeling.

These are the test scores provided by Williams et al. (2017) `[1] <#references>`__ (can't be directly compared with
above):

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


.. |alt text| image:: ../_static/diagram.png
