Classification models in DeepPavlov
===================================

In DeepPavlov one can find code for training and using classification models
which are implemented as a number of different **neural networks** or **sklearn models**.
Models can be used for binary, multi-class or multi-label classification.
List of available classifiers (more info see below):

* **BERT classifier** (see :doc:`here </apiref/models/bert>`) builds BERT [8]_ architecture for classification problem on **TensorFlow** or on **PyTorch**.

* **Keras classifier** (see :doc:`here </apiref/models/classifiers>`) builds neural network on Keras with tensorflow backend.

* **PyTorch classifier** (see :doc:`here </apiref/models/classifiers>`) builds neural network on PyTorch.

* **Sklearn classifier** (see :doc:`here </apiref/models/sklearn>`) builds most of sklearn classifiers.

Quick start
-----------

Command line
~~~~~~~~~~~~

**INSTALL** First whatever model you have chose you would need to install additional requirements:

.. code:: bash

    python -m deeppavlov install <path_to_config>

where ``<path_to_config>`` is a path to one of the :config:`provided config files <classifiers>`
or its name without an extension, for example :config:`"intents_snips" <classifiers/intents_snips.json>`.

To download pre-trained models, vocabs, embeddings on the dataset of interest one should run the following command
providing corresponding name of the config file (see above)
or provide flag ``-d`` for commands like ``interact``, ``telegram``, ``train``, ``evaluate``.:

.. code:: bash

    python -m deeppavlov download  <path_to_config>

where ``<path_to_config>`` is a path to one of the :config:`provided config files <classifiers>`
or its name without an extension, for example :config:`"intents_snips" <classifiers/intents_snips.json>`.

When using KerasClassificationModel for **Windows** platforms one have to set `KERAS_BACKEND` to `tensorflow`:

.. code:: bash

    set "KERAS_BACKEND=tensorflow"

**INTERACT** One can run the following command to interact in command line interface with provided config:

.. code:: bash

    python -m deeppavlov interact <path_to_config> [-d]

where ``<path_to_config>`` is a path to one of the :config:`provided config files <classifiers>`
or its name without an extension, for example :config:`"intents_snips" <classifiers/intents_snips.json>`.
With the optional ``-d`` parameter all the data required to run selected pipeline will be **downloaded**.

**TRAIN** After preparing the config file (including change of dataset, pipeline elements or parameters)
one can train model from scratch or from pre-trained model optionally.
To train model **from scratch** one should set  ``load_path`` to an **empty or non-existing** directory,
and ``save_path`` to a directory where trained model will be saved.
To train model **from saved** one should set ``load_path`` to **existing** directory containing
model's files (pay attention that model can be loaded from saved only if the clue sizes of network
layers coincide, other parameters of model as well as training parameters,
embedder, tokenizer, preprocessor and postprocessors could be changed
but be attentive in case of changing embedder - different embeddings of tokens will not give
the same results).
Then training can be run in the following way:

.. code:: bash

    python -m deeppavlov train <path_to_config>

where ``<path_to_config>`` is a path to one of the :config:`provided config files <classifiers>`
or its name without an extension, for example :config:`"intents_snips" <classifiers/intents_snips.json>`.
With the optional ``-d`` parameter all the data required to run selected pipeline will be **downloaded**.

Python code
~~~~~~~~~~~

One can also use these configs in python code.
When using ``KerasClassificationModel`` for **Windows** platform
one needs to set ``KERAS_BACKEND`` to ``tensorflow`` in the following way:

.. code:: python

    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"

**INTERACT** To download required data one have to set ``download`` parameter to ``True``.
Then one can build and interact a model from configuration file:

.. code:: python

    from deeppavlov import build_model, configs

    CONFIG_PATH = configs.classifiers.intents_snips  # could also be configuration dictionary or string path or `pathlib.Path` instance

    model = build_model(CONFIG_PATH, download=True)  # in case of necessity to download some data

    model = build_model(CONFIG_PATH, download=False)  # otherwise

    print(model(["What is the weather in Boston today?"]))

    >>> [['GetWeather']]

**TRAIN** Also training can be run in the following way:

.. code:: python

    from deeppavlov import train_model, configs

    CONFIG_PATH = configs.classifiers.intents_snips  # could also be configuration dictionary or string path or `pathlib.Path` instance

    model = train_model(CONFIG_PATH, download=True)  # in case of necessity to download some data

    model = train_model(CONFIG_PATH, download=False)  # otherwise

BERT models
-----------

BERT (Bidirectional Encoder Representations from Transformers) [8]_ is a Transformer pre-trained on masked language model
and next sentence prediction tasks. This approach showed state-of-the-art results on a wide range of NLP tasks in
English.

**deeppavlov.models.bert.BertClassifierModel** (see :doc:`here </apiref/models/bert>`) provides easy to use
solution for classification problem using pre-trained BERT.
Several **pre-trained English, multi-lingual and Russian BERT** models are provided in
:doc:`our BERT documentation </features/models/bert>`.

Two main components of BERT classifier pipeline in DeepPavlov are
``deeppavlov.models.preprocessors.bert_preprocessor.BertPreprocessor`` on TensorFlow (or ``deeppavlov.models.preprocessors.torch_transformers_preprocessor.TorchTransformersPreprocessor`` on PyTorch) (see :doc:`here </apiref/models/bert>`)
and ``deeppavlov.models.bert.bert_classifier.BertClassifierModel`` on TensorFlow (or ``deeppavlov.models.torch_bert.torch_transformers_classifier.TorchTransformersClassifierModel`` on PyTorch) (see :doc:`here </apiref/models/bert>`).
The ``deeppavlov.models.torch_bert.torch_transformers_classifier.TorchTransformersClassifierModel`` class supports any Transformer-based model.

Non-processed texts should be given to ``bert_preprocessor`` (``torch_transformers_preprocessor``) for tokenization on subtokens,
encoding subtokens with their indices and creating tokens and segment masks.
If one processed classes to one-hot labels in pipeline, ``one_hot_labels`` should be set to ``true``.

``bert_classifier`` (``torch_bert_classifier``) has a dense layer of number of classes size upon pooled outputs of Transformer encoder,
it is followed by ``softmax`` activation (``sigmoid`` if ``multilabel`` parameter is set to ``true`` in config).

Neural Networks on Keras
------------------------

**deeppavlov.models.classifiers.KerasClassificationModel** (see :doc:`here </apiref/models/classifiers>`)
contains a number of different neural network configurations for classification task.
Please, pay attention that each model has its own parameters that should be specified in config.
Information about parameters could be found :doc:`here </apiref/models/classifiers>`.
One of the available network configurations can be chosen in ``model_name`` parameter in config.
Below the list of available models is presented:

* ``cnn_model`` -- Shallow-and-wide CNN [1]_ with max pooling after convolution,
* ``dcnn_model`` -- Deep CNN with number of layers determined by the given number of kernel sizes and filters,
* ``cnn_model_max_and_aver_pool`` -- Shallow-and-wide CNN [1]_ with max and average pooling concatenation after convolution,
* ``bilstm_model`` -- Bidirectional LSTM,
* ``bilstm_bilstm_model`` -- 2-layers bidirectional LSTM,
* ``bilstm_cnn_model`` -- Bidirectional LSTM followed by shallow-and-wide CNN,
* ``cnn_bilstm_model`` -- Shallow-and-wide CNN followed by bidirectional LSTM,
* ``bilstm_self_add_attention_model`` -- Bidirectional LSTM followed by self additive attention layer,
* ``bilstm_self_mult_attention_model`` -- Bidirectional LSTM followed by self multiplicative attention layer,
* ``bigru_model`` -- Bidirectional GRU model.


Neural Networks on PyTorch
--------------------------

**deeppavlov.models.classifiers.TorchClassificationModel** (see :doc:`here </apiref/models/classifiers>`)
does not contain a zoo of models while it has an example of shallow-and-wide CNN (``swcnn_model``).
An instruction of how to build your own architecture on PyTorch one may find :doc:`here </intro/choose_framework>`.

Sklearn models
--------------

**deeppavlov.models.sklearn.SklearnComponent** (see :doc:`here </apiref/models/sklearn>`) is
a universal wrapper for all ``sklearn`` model that could be fitted.
One can set ``model_class`` parameter to full name of model (for example,
``sklearn.feature_extraction.text:TfidfVectorizer`` or ``sklearn.linear_model:LogisticRegression``).
Parameter ``infer_method`` should be set to class method for prediction
(``predict``, ``predict_proba``, ``predict_log_proba`` or ``transform``).
As for text classification in DeepPavlov we assign list of labels for each sample,
it is required to ensure that output of a classifier-``sklearn_component`` is a list of labels for each sample.
Therefore, for sklearn component classifier one should set ``ensure_list_output`` to ``true``.


Pre-trained models
------------------

We also provide with **pre-trained models** for classification on DSTC 2 dataset, SNIPS dataset, "AG News" dataset,
"Detecting Insults in Social Commentary", Twitter sentiment in Russian dataset.

`DSTC 2 dataset <http://camdial.org/~mh521/dstc/>`__ does not initially contain information about **intents**,
therefore, ``Dstc2IntentsDatasetIterator`` (``deeppavlov/dataset_iterators/dstc2_intents_interator.py``) instance
extracts artificial intents for each user reply using information from acts and slots.

Below we give several examples of intent construction:

    System: "Hello, welcome to the Cambridge restaurant system. You can
    ask for restaurants by area, price range or food type. How may I
    help you?"

    User: "cheap restaurant"

In the original dataset this user reply has characteristics

.. code:: bash

    "goals": {"pricerange": "cheap"}, 
    "db_result": null, 
    "dialog-acts": [{"slots": [["pricerange", "cheap"]], "act": "inform"}]}

This message contains only one intent: ``inform_pricerange``.

    User: "thank you good bye",

In the original dataset this user reply has characteristics

.. code:: bash

    "goals": {"food": "dontcare", "pricerange": "cheap", "area": "south"}, 
    "db_result": null, 
    "dialog-acts": [{"slots": [], "act": "thankyou"}, {"slots": [], "act": "bye"}]}

This message contains two intents ``(thankyou, bye)``. Train, valid and
test division is the same as on web-site.

`SNIPS dataset <https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines>`__
contains **intent classification** task for 7 intents (approximately 2.4
samples per intent):

-  GetWeather
-  BookRestaurant
-  PlayMusic
-  AddToPlaylist
-  RateBook
-  SearchScreeningEvent
-  SearchCreativeWork

Initially, classification model on SNIPS dataset [7]_ was trained only as an
example of usage that is why we provide pre-trained model for SNIPS with
embeddings trained on DSTC-2 dataset that is not the best choice for
this task. Train set is divided to train and validation sets to
illustrate ``basic_classification_iterator`` work.

`Detecting Insults in Social Commentary dataset <https://www.kaggle.com/c/detecting-insults-in-social-commentary>`__
contains binary classification task for **detecting insults** for
participants of conversation. Train, valid and test division is the same
as for the Kaggle challenge.

`AG News dataset <https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html>`__
contains **topic classification** task for 5 classes (range from 0
to 4 points scale). Test set is initial one from a web-site, valid is a
Stratified division 1/5 from the train set from web-site with 42 seed,
and the train set is the rest.

`Twitter mokoron dataset <http://study.mokoron.com/>`__ contains
**sentiment classification** of Russian tweets for positive and negative
replies [2]_. It was automatically labeled.
Train, valid and test division is made by hands (Stratified
division: 1/5 from all dataset for test set with 42 seed, then 1/5 from
the rest for validation set with 42 seed). Two provided pre-trained
models were trained on the same dataset but with and without preprocessing.
The main difference between scores is caused by the fact that some symbols
(deleted during preprocessing) were used for automatic labelling. Therefore,
it can be considered that model trained on preprocessed data is
based on semantics while model trained on unprocessed data
is based on punctuation and syntax.

`RuSentiment dataset <http://text-machine.cs.uml.edu/projects/rusentiment/>`__ contains
**sentiment classification** of social media posts for Russian language within 5 classes 'positive', 'negative',
'neutral', 'speech', 'skip'.

`SentiRuEval dataset <http://www.dialog-21.ru/evaluation/2016/sentiment/>`__ contains
**sentiment classification** of reviews for Russian language within 4 classes 'positive', 'negative',
'neutral', 'both'. Datasets on four different themes 'Banks', 'Telecom', 'Restaurants', 'Cars' are
combined to one big dataset.

`Questions on Yahoo Answers labeled as either informational or conversational dataset <https://webscope.sandbox.yahoo.com/catalog.php?datatype=l>`__
contains **intent classification** of English questions into two category: informational (`0`) and conversational (`1`) questions.
The dataset includes some additional metadata but for the presented pre-trained model only `Title` of questions and `Label` were used.
Embeddings were obtained from language model (ELMo) fine-tuned on the dataset

`L6 - Yahoo! Answers Comprehensive Questions and Answers <https://webscope.sandbox.yahoo.com/catalog.php?datatype=l>`__.
We do not provide datasets, both are available upon request to Yahoo Research.
Therefore, this model is available only for interaction.

`Stanford Sentiment Treebank <https://nlp.stanford.edu/sentiment/index.html>`__ contains 5-classes fine-grained **sentiment classification**
of sentences. Each sentence were initially labelled with floating point value from 0 to 1. For fine-grained classification
the floating point labels are converted to integer labels according to the intervals `[0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]`
corresponding to `very negative`, `negative`, `neutral`, `positive`, `very positive` classes.

`Yelp Reviews <https://www.yelp.com/dataset>`__ contains 5-classes **sentiment classification** of product reviews.
The labels are `1`, `2`, `3`, `4`, `5` corresponding to `very negative`, `negative`, `neutral`, `positive`, `very positive` classes.
The reviews are long enough (cut up to 200 subtokens).


+------------------+--------------------+------+-------------------------------------------------------------------------------------------------+-------------+--------+--------+-----------+
| Task             | Dataset            | Lang | Model                                                                                           | Metric      | Valid  | Test   | Downloads |
+==================+====================+======+=================================================================================================+=============+========+========+===========+
| 28 intents       | `DSTC 2`_          | En   | :config:`DSTC 2 emb <classifiers/intents_dstc2.json>`                                           | Accuracy    | 0.7613 | 0.7733 |  800 Mb   |
+                  +                    +      +-------------------------------------------------------------------------------------------------+             +--------+--------+-----------+
|                  |                    |      | :config:`Wiki emb <classifiers/intents_dstc2_big.json>`                                         |             | 0.9629 | 0.9617 |  8.5 Gb   |
+                  +                    +      +-------------------------------------------------------------------------------------------------+             +--------+--------+-----------+
|                  |                    |      | :config:`BERT <classifiers/intents_dstc2_bert.json>`                                            |             | 0.9673 | 0.9636 |  800 Mb   |
+------------------+--------------------+      +-------------------------------------------------------------------------------------------------+-------------+--------+--------+-----------+
| 7 intents        | `SNIPS-2017`_ [7]_ |      | :config:`DSTC 2 emb <classifiers/intents_snips.json>`                                           | F1-macro    | 0.8591 |    --  |  800 Mb   |
+                  +                    +      +-------------------------------------------------------------------------------------------------+             +--------+--------+-----------+
|                  |                    |      | :config:`Wiki emb <classifiers/intents_snips_big.json>`                                         |             | 0.9820 |    --  |  8.5 Gb   |
+                  +                    +      +-------------------------------------------------------------------------------------------------+             +--------+--------+-----------+
|                  |                    |      | :config:`Tfidf + SelectKBest + PCA + Wiki emb <classifiers/intents_snips_sklearn.json>`         |             | 0.9673 |    --  |  8.6 Gb   |
+                  +                    +      +-------------------------------------------------------------------------------------------------+             +--------+--------+-----------+
|                  |                    |      | :config:`Wiki emb weighted by Tfidf <classifiers/intents_snips_tfidf_weighted.json>`            |             | 0.9786 |    --  |  8.5 Gb   |
+------------------+--------------------+      +-------------------------------------------------------------------------------------------------+-------------+--------+--------+-----------+
| Insult detection | `Insults`_         |      | :config:`Reddit emb <classifiers/insults_kaggle.json>`                                          | ROC-AUC     | 0.9263 | 0.8556 |  6.2 Gb   |
+                  +                    +      +-------------------------------------------------------------------------------------------------+             +--------+--------+-----------+
|                  |                    |      | :config:`English BERT <classifiers/insults_kaggle_bert.json>`                                   |             | 0.9255 | 0.8612 |  1200 Mb  |
+                  +                    +      +-------------------------------------------------------------------------------------------------+             +--------+--------+-----------+
|                  |                    |      | :config:`English Conversational BERT <classifiers/insults_kaggle_conv_bert.json>`               |             | 0.9389 | 0.8941 |  1200 Mb  |
+                  +                    +      +-------------------------------------------------------------------------------------------------+             +--------+--------+-----------+
|                  |                    |      | :config:`English BERT on PyTorch <classifiers/insults_kaggle_bert_torch.json>`                  |             | 0.9329 | 0.877  |  1.1 Gb   |
+------------------+--------------------+      +-------------------------------------------------------------------------------------------------+-------------+--------+--------+-----------+
| 5 topics         | `AG News`_         |      | :config:`Wiki emb <classifiers/topic_ag_news.json>`                                             | Accuracy    | 0.8922 | 0.9059 |  8.5 Gb   |
+------------------+--------------------+      +-------------------------------------------------------------------------------------------------+-------------+--------+--------+-----------+
| Intent           |`Yahoo-L31`_        |      | :config:`Yahoo-L31 on conversational BERT <classifiers/yahoo_convers_vs_info_bert.json>`        | ROC-AUC     | 0.9436 |   --   |  1200 Mb  |
+------------------+--------------------+      +-------------------------------------------------------------------------------------------------+-------------+--------+--------+-----------+
| Sentiment        |`SST`_              |      | :config:`5-classes SST on conversational BERT <classifiers/sentiment_sst_conv_bert.json>`       | Accuracy    | 0.6456 | 0.6715 |  400 Mb   |
+                  +                    +      +-------------------------------------------------------------------------------------------------+             +--------+--------+-----------+
|                  |                    |      | :config:`5-classes SST on multilingual BERT <classifiers/sentiment_sst_multi_bert.json>`        |             | 0.5738 | 0.6024 |  660 Mb   |
+                  +                    +      +-------------------------------------------------------------------------------------------------+             +--------+--------+-----------+
|                  |                    |      | :config:`3-classes SST SWCNN on PyTorch <classifiers/sst_torch_swcnn.json>`                     |             | 0.7379 | 0.6312 |  4.3 Mb   |
+                  +--------------------+      +-------------------------------------------------------------------------------------------------+             +--------+--------+-----------+
|                  |`Yelp`_             |      | :config:`5-classes Yelp on conversational BERT <classifiers/sentiment_yelp_conv_bert.json>`     |             | 0.6925 | 0.6842 |  400 Mb   |
+                  +                    +      +-------------------------------------------------------------------------------------------------+             +--------+--------+-----------+
|                  |                    |      | :config:`5-classes Yelp on multilingual BERT <classifiers/sentiment_yelp_multi_bert.json>`      |             | 0.5896 | 0.5874 |  660 Mb   |
+------------------+--------------------+------+-------------------------------------------------------------------------------------------------+-------------+--------+--------+-----------+
| Sentiment        |`Twitter mokoron`_  | Ru   | :config:`RuWiki+Lenta emb w/o preprocessing <classifiers/sentiment_twitter.json>`               |             | 0.9965 | 0.9961 |  6.2 Gb   |
+                  +                    +      +-------------------------------------------------------------------------------------------------+             +--------+--------+-----------+
|                  |                    |      | :config:`RuWiki+Lenta emb with preprocessing <classifiers/sentiment_twitter_preproc.json>`      |             | 0.7823 | 0.7759 |  6.2 Gb   |
+                  +--------------------+      +-------------------------------------------------------------------------------------------------+-------------+--------+--------+-----------+
|                  |`RuSentiment`_      |      | :config:`RuWiki+Lenta emb <classifiers/rusentiment_cnn.json>`                                   | F1-weighted | 0.6541 | 0.7016 |  6.2 Gb   |
+                  +                    +      +-------------------------------------------------------------------------------------------------+             +--------+--------+-----------+
|                  |                    |      | :config:`Twitter emb super-convergence <classifiers/rusentiment_bigru_superconv.json>` [6]_     |             | 0.7301 | 0.7576 |  3.4 Gb   |
+                  +                    +      +-------------------------------------------------------------------------------------------------+             +--------+--------+-----------+
|                  |                    |      | :config:`ELMo <classifiers/rusentiment_elmo_twitter_cnn.json>`                                  |             | 0.7519 | 0.7875 |  700 Mb   |
+                  +                    +      +-------------------------------------------------------------------------------------------------+             +--------+--------+-----------+
|                  |                    |      | :config:`Multi-language BERT <classifiers/rusentiment_bert.json>`                               |             | 0.6809 | 0.7193 |  1900 Mb  |
+                  +                    +      +-------------------------------------------------------------------------------------------------+             +--------+--------+-----------+
|                  |                    |      | :config:`Conversational RuBERT <classifiers/rusentiment_convers_bert.json>`                     |             | 0.7548 | 0.7742 |  657 Mb   |
+------------------+--------------------+      +-------------------------------------------------------------------------------------------------+-------------+--------+--------+-----------+
| Intent           |Ru like`Yahoo-L31`_ |      | :config:`Conversational vs Informational on ELMo <classifiers/yahoo_convers_vs_info.json>`      | ROC-AUC     | 0.9412 |   --   |  700 Mb   |
+------------------+--------------------+------+-------------------------------------------------------------------------------------------------+-------------+--------+--------+-----------+

.. _`DSTC 2`: http://camdial.org/~mh521/dstc/
.. _`SNIPS-2017`: https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines
.. _`Insults`: https://www.kaggle.com/c/detecting-insults-in-social-commentary
.. _`AG News`: https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html
.. _`Twitter mokoron`: http://study.mokoron.com/
.. _`RuSentiment`: http://text-machine.cs.uml.edu/projects/rusentiment/
.. _`SentiRuEval`: http://www.dialog-21.ru/evaluation/2016/sentiment/
.. _`Yahoo-L31`: https://webscope.sandbox.yahoo.com/catalog.php?datatype=l
.. _`Yahoo-L6`: https://webscope.sandbox.yahoo.com/catalog.php?datatype=l
.. _`SST`: https://nlp.stanford.edu/sentiment/index.html
.. _`Yelp`: https://www.yelp.com/dataset

GLUE Benchmark
--------------
The General Language Understanding Evaluation (GLUE) benchmark is a collection of resources for training, evaluating,
and analyzing natural language understanding systems. More details are on the official page https://gluebenchmark.com/.

In DeepPavlov there is a set of configuration files to run training and evaluation on GLUE tasks train/dev sets.
DeepPavlov (DP) results on dev sets are averaged over 3 runs. We report the same metrics as on the official leaderboard
https://gluebenchmark.com/leaderboard.
 
+-------------------------------------------------+----------+---------+-------------+---------------+-------------+-------------+--------+---------+
|   Models                                        | CoLA     | SST2    | MRPC        | STS-B         | QQP         | MNLI-m/mm   | QNLI   | RTE     |
+=================================================+==========+=========+=============+===============+=============+=============+========+=========+
| :config:`DP bert-base-cased <classifiers/glue/>`| 61.50    | 92.62   | 89.69/85.37 | 87.62/87.32   | 86.85/90.11 | 83.43/83.64 | 90.95  |  68.23  |
+-------------------------------------------------+----------+---------+-------------+---------------+-------------+-------------+--------+---------+
| DP bert-base-uncased                            | 62.27    | 92.78   | 88.99/84.17 | 88.73/88.35   | 87.29/90.39 | 84.04/84.27 | 91.61  |  71.34  |
+-------------------------------------------------+----------+---------+-------------+---------------+-------------+-------------+--------+---------+
| `HuggingFace bert-base-uncased`_                | 49.23    | 91.97   | 89.47/85.29 | 83.95/83.70   | 84.31/88.40 | 80.61/81.08 | 87.46  |  61.73  |
+-------------------------------------------------+----------+---------+-------------+---------------+-------------+-------------+--------+---------+

.. _`HuggingFace bert-base-uncased`: https://github.com/huggingface/transformers/tree/master/examples/text-classification#run-pytorch-version


How to train on other datasets
------------------------------

We provide dataset reader ``BasicClassificationDatasetReader`` and dataset
``BasicClassificationDatasetIterator`` to work with ``.csv`` and ``.json`` files. These classes are described in
:doc:`readers docs </apiref/dataset_readers>` and :doc:`dataset iterators docs </apiref/dataset_iterators>`.

Data files should be in the following format (classes can be separated by custom symbol
given in the config as ``class_sep``, here ``class_sep=","``):

+-----------+---------------------------------+
| x         | y                               |
+===========+=================================+
| text\_0   | class\_0                        |
+-----------+---------------------------------+
| text\_1   | class\_0                        |
+-----------+---------------------------------+
| text\_2   | class\_1,class\_2               |
+-----------+---------------------------------+
| text\_3   | class\_1,class\_0,class\_2      |
+-----------+---------------------------------+
| ...       | ...                             |
+-----------+---------------------------------+

To train model one should

* set ``data_path`` to the directory to which ``train.csv`` should be downloaded,
* set ``save_path`` to the directory where the trained model should be saved,
* set all other parameters of model as well as embedder, tokenizer and preprocessor to desired ones.

Then training process can be run in the same way:

.. code:: bash

    python -m deeppavlov train <path_to_config>

Comparison
----------

The comparison of the presented model is given on **SNIPS** dataset [7]_. The
evaluation of model scores was conducted in the same way as in [3]_ to
compare with the results from the report of the authors of the dataset.
The results were achieved with tuning of parameters and embeddings
trained on Reddit dataset.

+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+
| Model                  | AddToPlaylist   | BookRestaurant   | GetWheather   | PlayMusic    | RateBook     | SearchCreativeWork   | SearchScreeningEvent   |
+========================+=================+==================+===============+==============+==============+======================+========================+
| api.ai                 | 0.9931          | 0.9949           | 0.9935        | 0.9811       | 0.9992       | 0.9659               | 0.9801                 |
+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+
| ibm.watson             | 0.9931          | 0.9950           | 0.9950        | 0.9822       | 0.9996       | 0.9643               | 0.9750                 |
+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+
| microsoft.luis         | 0.9943          | 0.9935           | 0.9925        | 0.9815       | 0.9988       | 0.9620               | 0.9749                 |
+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+
| wit.ai                 | 0.9877          | 0.9913           | 0.9921        | 0.9766       | 0.9977       | 0.9458               | 0.9673                 |
+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+
| snips.ai               | 0.9873          |       0.9921     | 0.9939        | 0.9729       | 0.9985       | 0.9455               | 0.9613                 |
+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+
| recast.ai              | 0.9894          | 0.9943           | 0.9910        | 0.9660       | 0.9981       | 0.9424               | 0.9539                 |
+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+
| amazon.lex             | 0.9930          | 0.9862           | 0.9825        | 0.9709       | 0.9981       | 0.9427               | 0.9581                 |
+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+
+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+
| Shallow-and-wide CNN   | **0.9956**      | **0.9973**       | **0.9968**    | **0.9871**   | **0.9998**   | **0.9752**           | **0.9854**             |
+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+

How to improve the performance
------------------------------

-  One can use FastText [4]_ to train embeddings that are better suited
   for considered datasets.
-  One can use some custom preprocessing to clean texts.
-  One can use ELMo [5]_ or BERT [8]_.
-  All the parameters should be tuned on the validation set.

References
----------

.. [1] Kim Y. Convolutional neural networks for sentence classification //arXiv preprint arXiv:1408.5882. – 2014.

.. [2] Ю. В. Рубцова. Построение корпуса текстов для настройки тонового классификатора // Программные продукты и системы, 2015, №1(109), –С.72-78

.. [3] https://www.slideshare.net/KonstantinSavenkov/nlu-intent-detection-benchmark-by-intento-august-2017

.. [4] P. Bojanowski\ *, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information.

.. [5] Peters, Matthew E., et al. "Deep contextualized word representations." arXiv preprint arXiv:1802.05365 (2018).

.. [6] Smith L. N., Topin N. Super-convergence: Very fast training of residual networks using large learning rates. – 2018.

.. [7] Coucke A. et al. Snips voice platform: an embedded spoken language understanding system for private-by-design voice interfaces //arXiv preprint arXiv:1805.10190. – 2018.

.. [8] Devlin J. et al. Bert: Pre-training of deep bidirectional transformers for language understanding //arXiv preprint arXiv:1810.04805. – 2018.
