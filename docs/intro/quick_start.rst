QuickStart
------------

First, follow instructions on :doc:`Installation page </intro/installation>`
to install ``deeppavlov`` package for Python 3.6/3.7.

DeepPavlov contains a bunch of great pre-trained NLP models. Each model is
determined by its config file. List of models is available on
:doc:`the doc page </features/overview>` or in
the ``deeppavlov.configs``:

    .. code:: python
        
        from deeppavlov import configs

When you've decided on the model (+ config file), there are two ways to train,
evaluate and infer it:

* via `Command line interface (CLI)`_ and
* via `Python`_.

Before making choice of an interface, install model's package requirements
(CLI):

    .. code:: bash
        
        python -m deeppavlov install <config_path>

    * where ``<config_path>`` is path to the chosen model's config file (e.g.
      ``deeppavlov/configs/ner/slotfill_dstc2.json``) or just name without
      `.json` extension (e.g. ``slotfill_dstc2``)


Command line interface (CLI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get predictions from a model interactively through CLI, run

    .. code:: bash
        
        python -m deeppavlov interact <config_path> [-d]

    * ``-d`` downloads required data -- pretrained model files and embeddings
      (optional).

You can train it in the same simple way:

    .. code:: bash
        
        python -m deeppavlov train <config_path> [-d]

    Dataset will be downloaded regardless of whether there was ``-d`` flag or
    not.

    To train on your own data, you need to modify dataset reader path in the
    `train section doc <configuration.html#Train-config>`__. The data format is
    specified in the corresponding model doc page. 

There are even more actions you can perform with configs:

    .. code:: bash
        
        python -m deeppavlov <action> <config_path> [-d]

    * ``<action>`` can be
        * ``download`` to download model's data (same as ``-d``),
        * ``train`` to train the model on the data specified in the config file,
        * ``evaluate`` to calculate metrics on the same dataset,
        * ``interact`` to interact via CLI,
        * ``riseapi`` to run a REST API server (see :doc:`docs
          </integrations/rest_api>`),
        * ``risesocket`` to run a socket API server (see :doc:`docs
          </integrations/socket_api>`),
        * ``telegram`` to run as a Telegram bot (see :doc:`docs
          </integrations/telegram>`),
        * ``msbot`` to run a Miscrosoft Bot Framework server (see
          :doc:`docs </integrations/ms_bot>`),
        * ``predict`` to get prediction for samples from `stdin` or from
          `<file_path>` if ``-f <file_path>`` is specified.
    * ``<config_path>`` specifies path (or name) of model's config file
    * ``-d`` downloads required data


Python
~~~~~~

To get predictions from a model interactively through Python, run

    .. code:: python
        
        from deeppavlov import build_model

        model = build_model(<config_path>, download=True)

        # get predictions for 'input_text1', 'input_text2'
        model(['input_text1', 'input_text2'])

    * where ``download=True`` downloads required data from web -- pretrained model
      files and embeddings (optional),
    * ``<config_path>`` is path to the chosen model's config file (e.g.
      ``"deeppavlov/configs/ner/ner_ontonotes_bert_mult.json"``) or
      ``deeppavlov.configs`` attribute (e.g.
      ``deeppavlov.configs.ner.ner_ontonotes_bert_mult`` without quotation marks).

You can train it in the same simple way:

    .. code:: python
        
        from deeppavlov import train_model 

        model = train_model(<config_path>, download=True)

    * ``download=True`` downloads pretrained model, therefore the pretrained
      model will be, first, loaded and then train (optional).

    Dataset will be downloaded regardless of whether there was ``-d`` flag or
    not.

    To train on your own data, you need to modify dataset reader path in the
    `train section doc <configuration.html#Train-config>`__. The data format is
    specified in the corresponding model doc page. 

You can also calculate metrics on the dataset specified in your config file:

    .. code:: python
        
        from deeppavlov import evaluate_model 

        model = evaluate_model(<config_path>, download=True)

There are also available integrations with various messengers, see
:doc:`Telegram Bot doc page </integrations/telegram>` and others in the
Integrations section for more info.


Using GPU
~~~~~~~~~

To run or train **TensorFlow**-based DeepPavlov models on GPU you should have `CUDA <https://developer.nvidia.com/cuda-toolkit>`__ 10.0
installed on your host machine and TensorFlow with GPU support (``tensorflow-gpu``)
installed in your python environment. Current supported TensorFlow version is 1.15.5. Run

    .. code:: bash

        pip install tensorflow-gpu==1.15.5

before installing model's package requirements to install supported ``tensorflow-gpu`` version.

To run or train **PyTorch**-based DeepPavlov models on GPU you should also have `CUDA <https://developer.nvidia.com/cuda-toolkit>`__ 9.0 or 10.0
installed on your host machine, and install model's package requirements.
If you want to run the code on GPU, just make the device visible for the script.
If you want to use a particular device, you may set it in command line:

    .. code:: bash

        export CUDA_VISIBLE_DEVICES=3; python -m deeppavlov train <config_path>

or in Python script:

    .. code:: python

        import os

        os.environ["CUDA_VISIBLE_DEVICES"]="3"

In case one wants to left the GPU device visible but use CPU, one can set directly in the configuration file in dictionary
with model parameters `"device": "cpu"`.


Pretrained models
~~~~~~~~~~~~~~~~~

DeepPavlov provides a wide range of pretrained models and skills.
See :doc:`features overview </features/overview>` for more info. Please
note that most of our models are trained on specific datasets for
specific tasks and may require further training on your data.
You can find a list of our out-of-the-box models `below <#out-of-the-box-pretrained-models>`_.


Docker images
~~~~~~~~~~~~~

You can run DeepPavlov models in :doc:`riseapi </integrations/rest_api>` mode
via Docker without installing DP. Both your CPU and GPU (we support NVIDIA graphic
processors) can be utilised, please refer our `CPU <https://hub.docker.com/r/deeppavlov/base-cpu>`_
and `GPU <https://hub.docker.com/r/deeppavlov/base-gpu>`_ Docker images run instructions.


Out-of-the-box pretrained models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While the best way to solve most of the NLP tasks lies through collecting datasets
and training models according to the domain and an actual task itself, DeepPavlov
offers several pretrained models, which can be strong baselines for a wide range of tasks.

You can run these models `via Docker <#docker-images>`_ or in ``riseapi``/``risesocket`` mode to use in
solutions. See :doc:`riseapi </integrations/rest_api>` and :doc:`risesocket </integrations/socket_api>`
modes documentation for API details.


Text Question Answering
=======================

Text Question Answering component answers a question based on a given context (e.g,
a paragraph of text), where the answer to the question is a segment of the context.

.. table::
    :widths: auto

    +----------+------------------------------------------------------------------------------------------------+-------------------------------------------+
    | Language | DeepPavlov config                                                                              | Demo                                      |
    +==========+================================================================================================+===========================================+
    | Multi    | :config:`squad_bert_multilingual_freezed_emb <squad/squad_bert_multilingual_freezed_emb.json>` | https://demo.deeppavlov.ai/#/mu/textqa    |
    +----------+------------------------------------------------------------------------------------------------+-------------------------------------------+
    | En       | :config:`squad_bert_infer <squad/squad_bert_infer.json>`                                       | https://demo.deeppavlov.ai/#/en/textqa    |
    +----------+------------------------------------------------------------------------------------------------+-------------------------------------------+
    | Ru       | :config:`squad_ru_bert_infer <squad/squad_ru_bert_infer.json>`                                 | https://demo.deeppavlov.ai/#/ru/textqa    |
    +----------+------------------------------------------------------------------------------------------------+-------------------------------------------+


Name Entity Recognition
=======================

Named Entity Recognition (NER) classifies tokens in text into predefined categories
(tags), such as person names, quantity expressions, percentage expressions, names
of locations, organizations, as well as expression of time, currency and others.

.. table::
    :widths: auto

    +----------+------------------------------------------------------------------------------------------------+-------------------------------------------+
    | Language | DeepPavlov config                                                                              | Demo                                      |
    +==========+================================================================================================+===========================================+
    | Multi    | :config:`ner_ontonotes_bert_mult <ner/ner_ontonotes_bert_mult.json>`                           | https://demo.deeppavlov.ai/#/mu/ner       |
    +----------+------------------------------------------------------------------------------------------------+-------------------------------------------+
    | En       | :config:`ner_ontonotes_bert_mult <ner/ner_ontonotes_bert_mult.json>`                           | https://demo.deeppavlov.ai/#/en/ner       |
    +----------+------------------------------------------------------------------------------------------------+-------------------------------------------+
    | Ru       | :config:`ner_rus_bert <ner/ner_rus_bert.json>`                                                 | https://demo.deeppavlov.ai/#/ru/ner       |
    +----------+------------------------------------------------------------------------------------------------+-------------------------------------------+


Insult Detection
================

Insult detection predicts whether a text (e.g, post or speech in some
public discussion) is considered insulting to one of the persons it is
related to.

.. table::
    :widths: auto

    +----------+------------------------------------------------------------------------------------------------+-------------------------------------------+
    | Language | DeepPavlov config                                                                              | Demo                                      |
    +==========+================================================================================================+===========================================+
    | En       | :config:`insults_kaggle_conv_bert <classifiers/insults_kaggle_conv_bert.json>`                 | https://demo.deeppavlov.ai/#/en/insult    |
    +----------+------------------------------------------------------------------------------------------------+-------------------------------------------+


Sentiment Analysis
==================

Classify text according to a prevailing emotion (positive, negative, etc.) in it.

.. table::
    :widths: auto

    +----------+------------------------------------------------------------------------------------------------+-------------------------------------------+
    | Language | DeepPavlov config                                                                              | Demo                                      |
    +==========+================================================================================================+===========================================+
    | Ru       | :config:`rusentiment_elmo_twitter_cnn <classifiers/rusentiment_elmo_twitter_cnn.json>`         | https://demo.deeppavlov.ai/#/ru/sentiment |
    +----------+------------------------------------------------------------------------------------------------+-------------------------------------------+


Paraphrase Detection
====================

Detect if two given texts have the same meaning.

.. table::
    :widths: auto

    +----------+------------------------------------------------------------------------------------------------+-------------------------------------------+
    | Language | DeepPavlov config                                                                              | Demo                                      |
    +==========+================================================================================================+===========================================+
    | En       | :config:`paraphraser_bert <classifiers/paraphraser_bert.json>`                                 | None                                      |
    +----------+------------------------------------------------------------------------------------------------+-------------------------------------------+
    | Ru       | :config:`paraphraser_rubert <classifiers/paraphraser_rubert.json>`                             | None                                      |
    +----------+------------------------------------------------------------------------------------------------+-------------------------------------------+
