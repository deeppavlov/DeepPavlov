Intent Catcher
##############

Overview
********
Intent Catcher is an NLP component used for intent detection in the Conversational AI systems.

It consists of an embedder, which is a Transformer model, and a number of dense layers, that are fitted upon provided embeddings. The current provided embeddings are: Universal Sentence Encoder [1]_, and it's larger version.

Intent Catcher has been originally designed for the high-level intent detection as part of the `DREAM Socialbot <https://deeppavlov.ai/challenges/dream_alexa_3>`_ that was built by DeepPavlov team for Alexa Prize 3.

Goals
=====
Typical approach for building ML-based intent classification is based on providing a relatively large number of examples for each of the intents. This might make sense when a number of intents is relatively small and there is enough data (e.g., a small internal organizational chatbot) but is questionable when the number of intents is large and amount of available data is relatively small.

For Alexa Prize 3, typical approach didn't work. Alexa Prize socialbots are expected to react a wide number of user intents in the open domain. The team needed to have a simple and fast way to add more intents, and add a relatively small number of examples for each new intent. Using regular expressions alone wouldn't be useful. But they could be used for up-sampling.

Intent Catcher was designed around idea that by adding an additional cost of requiring basic knowledge of Regular Expressions, users would be able to provide a smaller number of examples in RegEx format to enable up-sampling. In addition to that, it turned out that using RegEx directly, in addition to the up-sampled dataset was useful, too. Finally, there was need to check punctuation as a useful way to distinguish statements from questions and the like.

Features
********
* Up-sampling using RegEx-based format
* Direct RegEx-based pattern matching
* Additional checks for punctuation

How Do I: Train My Intent Classifier
************************************

Dataset construction
====================

Dataset can be constructed in 2 ways: listing number of intents and regular expressions in .json, or just a usual .csv format.
The json format is down below:

.. code:: json

    {
        "intent_1": ["regexp1", "regexp2"]
    }

To use data in this format, don't forget to add ``intent_catcher_reader`` as a dataset_reader in the config of model.

Train and evaluate model
========================

All the embeddings come pre-trained, and there is no need to install them. Though, for both Command Line Interface (CLI) and Python it is necessary to install dependences first.
To do so, run:

.. code:: bash

    python -m deeppavlov install intent_catcher

To use a pre-trained model from CLI use the following command:

.. code:: bash

    python -m deeppavlov interact intent_catcher -d

where ``intent_catcher`` is the name of the config.

The provided config example is :config:`intent_catcher <intent_catcher/intent_catcher.json>`


How Do I: Integrate Intent Catcher into DeepPavlov Deepy
********************************************************

To integrate your Intent Catcher-based intent classifier into your Multiskill AI Assistant built using DeepPavlov Conversational AI Stack, follow the following instructions:

1. Clone `Deepy repository <https://github.com/deepmipt/assistant-base>`_
2. Replace ``docker-compose.yml`` in the root of the repository and ``pipeline_conf.json`` in the ``/agent/`` subdirectory with the corresponding files from the `deepy_adv <https://github.com/deepmipt/assistant-base/tree/main/assistant_dists/deepy_adv>`_ **Deepy Distribution**
3. Clone the `Tutorial Notebook <https://colab.research.google.com/drive/1l6Fhj3rEVup0N-n9Jy5z_iA3b1W53V6m?usp=sharing>`_
4. Change its ``intents`` based on your project needs with your custom **intents**
5. Train the Intent Catcher model in your copy of the Tutorial Notebook
6. Download and put saved data from your copy of the Tutorial Notebook into the `Intent Catcher <https://github.com/deepmipt/assistant-base/tree/main/annotators/intent_catcher>`_
7. [Optional] Unless you need a Chit-Chat skill remove `it <https://github.com/deepmipt/assistant-base/tree/main/skills/program-y>`_ from at both the ``/agent/pipeline_conf.json`` and from ``docker-compose.yml``
8. Use ``docker-compose up --build`` command to build and run your DeepPavlov-based Multiskill AI Assistant

.. note::
   In the future versions of the DeepPavlov Library we will provide a more comprehensive update to the documentation to further simplify the process of adding DeepPavlov NLP components as annotators to the Multiskill AI Assistants built using DeepPavlov Conversational AI Stack. Stay tuned!

References
************
.. [1] Cer, Daniel, et al. "Universal sentence encoder." arXiv preprint arXiv:1803.11175 (2018).
