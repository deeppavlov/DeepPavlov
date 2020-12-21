IntentCatcher in DeepPavlov
===========================

IntentCatcher is a model used for intent detection for models in conversational systems.
It consists of an embedder, which is a Transformer model, and a number of dense layer, that are fitted upon provided embeddings. The current provided embeddings are: Universal Sentence Encoder [1]_ and it's larger version.
The main feature is that user can construct a dataset using regular expressions as an input to the model, and it will generate limited number of phrases from them to fit.

Dataset construction
--------------------

Dataset can be constructed in 2 ways: listing number of intents and regular expressions in .json, or just a usual .csv format.
The json format is down below:

.. code:: json
  {
    "intent_1": ["regexp1", "regexp2"]
  }

To use data in this format, don't forget to add ``intent_catcher_reader`` as a dataset_reader in the config of model.

Train and evaluate model
------------------------

All the embeddings come pre-trained, and there is no need to install them. Though, for both Command Line Interface (CLI) and Python it is necessary to install dependences first.
To do so, run:

.. code:: bash

    python -m deeppavlov install intent_catcher

To use a pre-trained model from CLI use the following command:

.. code:: bash

    python deeppavlov/deep.py interact intent_catcher

where ``intent_catcher`` is the name of the config.

The provided config example is :config:`intent_catcher <intent_catcher/intent_catcher.json>`

.. [1] Cer, Daniel, et al. "Universal sentence encoder." arXiv preprint arXiv:1803.11175 (2018).
