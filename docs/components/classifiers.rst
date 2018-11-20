Classification models in DeepPavlov
===================================

In this repository one can find code for training and using classification models
which are implemented as a number of different **neural networks** (for example, shallow-and-wide Convolutional
Neural Network [1]_) or **sklearn models**.
Models can be used for binary, multi-class or multi-label classification.

Available classifiers are:

* **deeppavlov.models.classifiers.KerasClassificationModel** (registered as ``keras_classification_model``) builds neural network on Keras with tensorflow backend. One of the available network configurations can be chosen in ``model_name`` parameter in config. List of implemented networks can be found bellow.

* **deeppavlov.models.sklearn.SklearnComponent** (registered as ``sklearn_component``) builds most of sklearn classifiers. Chosen model should be passed to ``model_class``, e.g. ``"model_class": "sklearn.neighbors:KNeighborsClassifier"``, as well as ``infer_method`` can be assigned to any sklearn model's prediction methods (e.g. ``predict`` or ``predict_proba``). As for text classification in DeepPavlov we assign list of labels for each sample, it is required to ensure that output of a classifier-``sklearn_component`` is a list of labels for each sample. Therefore, for sklearn component classifier one should set ``ensure_list_output`` to ``true``.

Quick start
-----------

One can run the following command to try provided pipelines out:

::

    python -m deeppavlov interact <path_to_config> [-d]

where ``<path_to_config>`` is one of the :config:`provided config files <classifiers>`.
With the optional ``-d`` parameter all the data required to run
selected pipeline will be downloaded.

One can also use these configs in your python code. To download required data one have to set ``download`` parameter to ``True``.

.. code:: python

    from deeppavlov import build_model, configs

    CONFIG_PATH = configs.classifiers.intents_snips

    model = build_model(CONFIG_PATH, download=True)

    print(model(["What is the weather in Boston today?"]))

    >>> [['GetWeather']]


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

::

    "goals": {"pricerange": "cheap"}, 
    "db_result": null, 
    "dialog-acts": [{"slots": [["pricerange", "cheap"]], "act": "inform"}]}

This message contains only one intent: ``inform_pricerange``.

    User: "thank you good bye",

In the original dataset this user reply has characteristics

::

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

Initially, classification model on SNIPS dataset was trained only as an
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

`Questions on Yahoo Answers labeled as either informational or conversational dataset <https://webscope.sandbox.yahoo.com/catalog.php?datatype=l>`__
contains **intent classification** of English questions into two category: informational (`0`) and conversational (`1`) questions.
The dataset includes some additional metadata but for the presented pre-trained model only `Title` of questions and `Label` were used.
Embeddings were obtained from language model (ELMo) fine-tuned on the dataset
`L6 - Yahoo! Answers Comprehensive Questions and Answers <https://webscope.sandbox.yahoo.com/catalog.php?datatype=l>`__.
We do not provide datasets, both are available upon request to Yahoo Research.
Therefore, this model is available only for interaction.

+-------------------+--------------------------------------------------------------------------------------------------------------+------------------+------+----------+--------+--------+
| Dataset           | Model                                                                                                        | Task             | Lang | Metric   | Valid  | Test   |
+-------------------+--------------------------------------------------------------------------------------------------------------+------------------+------+----------+--------+--------+
| `DSTC 2`_         | :config:`DSTC 2 on DSTC 2 embeddings <classifiers/intents_dstc2.json>`                                       | 28 intents       | En   | Accuracy | 0.7732 | 0.7868 |
+-------------------+--------------------------------------------------------------------------------------------------------------+------------------+------+----------+--------+--------+
| `DSTC 2`_         | :config:`DSTC 2 on Wiki embeddings <classifiers/intents_dstc2_big.json>`                                     | 28 intents       | En   | Accuracy | 0.9602 | 0.9593 |
+-------------------+--------------------------------------------------------------------------------------------------------------+------------------+------+----------+--------+--------+
| `SNIPS-2017`_     | :config:`SNIPS on DSTC 2 embeddings <classifiers/intents_snips.json>`                                        | 7 intents        | En   | F1       | 0.8664 |    --  |
+-------------------+--------------------------------------------------------------------------------------------------------------+------------------+------+----------+--------+--------+
| `SNIPS-2017`_     | :config:`SNIPS on Wiki embeddings <classifiers/intents_snips_big.json>`                                      | 7 intents        | En   | F1       | 0.9808 |    --  |
+-------------------+--------------------------------------------------------------------------------------------------------------+------------------+------+----------+--------+--------+
| `Insults`_        | :config:`InsultsKaggle on Reddit embeddings <classifiers/insults_kaggle.json>`                               | Insult detection | En   | ROC-AUC  | 0.9271 | 0.8618 |
+-------------------+--------------------------------------------------------------------------------------------------------------+------------------+------+----------+--------+--------+
| `AG News`_        | :config:`AG News on Wiki embeddings <classifiers/topic_ag_news.json>`                                        | 5 topics         | En   | Accuracy | 0.8876 | 0.9011 |
+-------------------+--------------------------------------------------------------------------------------------------------------+------------------+------+----------+--------+--------+
|`Twitter mokoron`_ | :config:`Twitter on RuWiki+Lenta embeddings without any preprocessing <classifiers/sentiment_twitter.json>`  | Sentiment        | Ru   | Accuracy | 0.9972 | 0.9971 |
+-------------------+--------------------------------------------------------------------------------------------------------------+------------------+------+----------+--------+--------+
|`Twitter mokoron`_ | :config:`Twitter on RuWiki+Lenta embeddings with preprocessing <classifiers/sentiment_twitter_preproc.json>` | Sentiment        | Ru   | Accuracy | 0.7811 | 0.7749 |
+-------------------+--------------------------------------------------------------------------------------------------------------+------------------+------+----------+--------+--------+
|`RuSentiment`_     | :config:`RuSentiment on RuWiki+Lenta embeddings <classifiers/rusentiment_cnn.json>`                          | Sentiment        | Ru   | F1       | 0.6393 | 0.6539 |
+-------------------+--------------------------------------------------------------------------------------------------------------+------------------+------+----------+--------+--------+
|`RuSentiment`_     | :config:`RuSentiment on ELMo <classifiers/rusentiment_elmo.json>`                                            | Sentiment        | Ru   | F1       | 0.7066 | 0.7301 |
+-------------------+--------------------------------------------------------------------------------------------------------------+------------------+------+----------+--------+--------+

.. _`DSTC 2`: http://camdial.org/~mh521/dstc/
.. _`SNIPS-2017`: https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines
.. _`Insults`: https://www.kaggle.com/c/detecting-insults-in-social-commentary
.. _`AG News`: https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html
.. _`Twitter mokoron`: http://study.mokoron.com/
.. _`RuSentiment`: http://text-machine.cs.uml.edu/projects/rusentiment/
.. _`Yahoo-L31`: https://webscope.sandbox.yahoo.com/catalog.php?datatype=l
.. _`Yahoo-L6`: https://webscope.sandbox.yahoo.com/catalog.php?datatype=l

Download pre-trained model
--------------------------

DeepPavlov provides the following **pre-trained models**:

-  :config:`intents_dstc2.json <classifiers/intents_dstc2.json>` -- DSTC 2 - intent model for English language with embeddings trained
   via fastText on DSTC 2 (800 Mb).
-  :config:`intents_dstc2_big.json <classifiers/intents_dstc2_big.json>` -- DSTC 2 - intent model for English language with `embeddings trained
   on Wiki <https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md>`__.
   This model achieves higher accuracy than the first one.
-  :config:`intents_snips.json <classifiers/intents_snips.json>` -- SNIPS - intent model for English language.
-  :config:`insults_kaggle.json <classifiers/insults_kaggle.json>` -- Insults analysis for English language.
-  :config:`topic_ag_news.json <classifiers/topic_ag_news.json>` -- AG News topic analysis for English language.
-  :config:`sentiment_twitter.json <classifiers/sentiment_twitter.json>` -- Twitter Mokoron sentiment analysis for **Russian** language.

To download pre-trained models, vocabs, embeddings on the dataset of interest one should run the following command
providing corresponding name of the config file (see above):

::

    python deep.py download configs/classifiers/intents_dstc2.json

or provide flag ``-d`` for commands like ``interact``, ``interactbot``,
etc. The flag ``-d`` provides downloading all the required components.


Infer from pre-trained model
----------------------------

To use a pre-trained model for inference one should run the following
command providing corresponding name of the config file (see above):

::

    python deep.py interact configs/classifiers/intents_dstc2.json

or

::

    python deep.py interactbot configs/classifiers/intents_dstc2.json -t <TELEGRAM_TOKEN>

For 'interactbot' mode one should specify a Telegram bot token in ``-t`` parameter or in the ``TELEGRAM_TOKEN``
environment variable.

Now user can enter a text string and get output of two elements: the first one is an array of classes names
which the string belongs to, and the second one is a dictionary with probability distribution among all
the considered classes (take into account that for multi-class classification then sum of probabilities
is not equal to 1).

An example of interacting the model from :config:`intents_dstc2.json <classifiers/intents_dstc2.json>`

::

    :: hey! I want cheap restaurant
    >> (array(['inform_pricerange'], dtype='<U17'), {'ack': 0.0040760376, 'affirm': 0.017633557, 'bye': 0.023906048, 'confirm_area': 0.0040424005, 'confirm_food': 0.012261569, 'confirm_pricerange': 0.007227284, 'deny_food': 0.003502861, 'deny_name': 0.003412795, 'hello': 0.0061915903, 'inform_area': 0.15999688, 'inform_food': 0.18303667, 'inform_name': 0.0042709936, 'inform_pricerange': 0.30197725, 'inform_this': 0.03864918, 'negate': 0.016452404, 'repeat': 0.003964727, 'reqalts': 0.026930325, 'reqmore': 0.0030793257, 'request_addr': 0.08075432, 'request_area': 0.018258458, 'request_food': 0.018060096, 'request_phone': 0.07433994, 'request_postcode': 0.012727374, 'request_pricerange': 0.024933394, 'request_signature': 0.0034591882, 'restart': 0.0038622846, 'thankyou': 0.036836267, 'unknown': 0.045310754})

and an example of interacting the model from
:config:`intents_dstc2_big.json <classifiers/intents_dstc2_big.json>`

::

    ::I want cheap chinese restaurant
    >> (array(['inform_food', 'inform_pricerange'], dtype='<U18'), {'ack': 0.008203662, 'affirm': 0.010941843, 'bye': 0.0058273915, 'confirm_area': 0.011861361, 'confirm_food': 0.017537124, 'confirm_pricerange': 0.012897875, 'deny_food': 0.009804511, 'deny_name': 0.008331243, 'hello': 0.009887574, 'inform_area': 0.009167877, 'inform_food': 0.9627541, 'inform_name': 0.008696462, 'inform_pricerange': 0.98613375, 'inform_this': 0.009358878, 'negate': 0.011380567, 'repeat': 0.00850759, 'reqalts': 0.012249454, 'reqmore': 0.008230184, 'request_addr': 0.006192594, 'request_area': 0.009336099, 'request_food': 0.008417402, 'request_phone': 0.004564096, 'request_postcode': 0.006752021, 'request_pricerange': 0.010917218, 'request_signature': 0.008601435, 'restart': 0.00838949, 'thankyou': 0.0060319724, 'unknown': 0.010502234})

Train model
-----------

Available Neural models
~~~~~~~~~~~~~~~~~~~~~~~

DeepPavlov contains a number of different model configurations for
classification task. Below the list of available models is presented:

* ``cnn_model`` -- Shallow-and-wide CNN with max pooling after convolution,
* ``dcnn_model`` -- Deep CNN with number of layers determined by the given number of kernel sizes and filters,
* ``cnn_model_max_and_aver_pool`` -- Shallow-and-wide CNN with max and average pooling concatenation after convolution,
* ``bilstm_model`` -- Bidirectional LSTM,
* ``bilstm_bilstm_model`` -- 2-layers bidirectional LSTM,
* ``bilstm_cnn_model`` -- Bidirectional LSTM followed by shallow-and-wide CNN,
* ``cnn_bilstm_model`` -- Shallow-and-wide CNN followed by bidirectional LSTM,
* ``bilstm_self_add_attention_model`` -- Bidirectional LSTM followed by self additive attention layer,
* ``bilstm_self_mult_attention_model`` -- Bidirectional LSTM followed by self multiplicative attention layer,
* ``bigru_model`` -- Bidirectional GRU model.

**Please, pay attention that each model has its own parameters that should be specified in config.**

Train again on provided datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To train from pre-trained model, re-train a model or train it
with other parameters on one of the provided datasets,
one should set ``save_path`` to a directory where the trained
model will be saved (pre-trained model will be loaded if ``load_path``
is provided and files exist, otherwise it will be created from scratch).
All other parameters of the model as well as embedder, tokenizer and preprocessor
could be changed. Then training can be run in the following way:

::

    python deep.py train "path_to_config"

Train on other datasets
~~~~~~~~~~~~~~~~~~~~~~~

Constructing intents from DSTC 2 makes ``Dstc2IntentsDatasetIterator`` difficult to use.
Therefore, we also provide another dataset reader ``BasicClassificationDatasetReader`` and dataset
``BasicClassificationDatasetIterator`` to work with ``.csv`` and ``.json`` files. These classes are described in
``deeppavlov/dataset_readers/basic_classification_reader.py`` and
``deeppavlov/dataset_iterators/basic_classification_dataset_iterator.py``.

Data files should be in the following format:

+-----------+---------------------------------+
| x         | y                               |
+===========+=================================+
| text\_0   | intent\_0                       |
+-----------+---------------------------------+
| text\_1   | intent\_0                       |
+-----------+---------------------------------+
| text\_2   | intent\_1,intent\_2             |
+-----------+---------------------------------+
| text\_3   | intent\_1,intent\_0,intent\_2   |
+-----------+---------------------------------+
| ...       | ...                             |
+-----------+---------------------------------+

To train model one should

* set ``data_path`` to the directory to which ``train.csv`` should be downloaded,
* set ``save_path`` to the directory where the trained model should be saved,
* set all other parameters of model as well as embedder, tokenizer and preprocessor to desired ones.

Then training process can be run in the same way:

::

    python deep.py train "path_to_config"

The current version of :config:`intents_snips.json <classifiers/intents_snips.json>`` contains parameters for
intent recognition for SNIPS benchmark dataset that was restored in
``.csv`` format and will be downloaded automatically.

**Important: we do not provide any special embedding binary file for
SNIPS dataset. In order to train the model one should provide own
embedding binary file, because embedding file trained on DSTC-2 dataset
is not the best choice for this task.**

Comparison
----------

As no one had published intent recognition for DSTC-2 data, the
comparison of the presented model is given on **SNIPS** dataset. The
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
-  One can use ELMo [5]_ embeddings.
-  All the parameters should be tuned on the validation set.

References
----------

.. [1] Kim Y. Convolutional neural networks for sentence classification //arXiv preprint arXiv:1408.5882. – 2014.

.. [2] Ю. В. Рубцова. Построение корпуса текстов для настройки тонового классификатора // Программные продукты и системы, 2015, №1(109), –С.72-78

.. [3] https://www.slideshare.net/KonstantinSavenkov/nlu-intent-detection-benchmark-by-intento-august-2017

.. [4] P. Bojanowski\ *, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information.

.. [5] Peters, Matthew E., et al. "Deep contextualized word representations." arXiv preprint arXiv:1802.05365 (2018).