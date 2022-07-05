Classification models in DeepPavlov
===================================

In DeepPavlov one can find code for training and using classification models
which are implemented as a number of different **neural networks** or **sklearn models**.
Models can be used for binary, multi-class or multi-label classification.
List of available classifiers (more info see below):

* **BERT classifier** (see :doc:`here </apiref/models/torch_bert>`) builds BERT [4]_ architecture for classification problem on **PyTorch**.

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
or its name without an extension, for example :config:`"insults_kaggle_bert" <classifiers/insults_kaggle_bert.json>`.

To download pre-trained models, vocabs, embeddings on the dataset of interest one should run the following command
providing corresponding name of the config file (see above)
or provide flag ``-d`` for commands like ``interact``,  ``train``, ``evaluate``:

.. code:: bash

    python -m deeppavlov download  <path_to_config>

where ``<path_to_config>`` is a path to one of the :config:`provided config files <classifiers>`
or its name without an extension, for example :config:`"insults_kaggle_bert" <classifiers/insults_kaggle_bert.json>`.

**INTERACT** One can run the following command to interact in command line interface with provided config:

.. code:: bash

    python -m deeppavlov interact <path_to_config> [-d]

where ``<path_to_config>`` is a path to one of the :config:`provided config files <classifiers>`
or its name without an extension, for example :config:`"insults_kaggle_bert" <classifiers/insults_kaggle_bert.json>`.
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
or its name without an extension, for example :config:`"insults_kaggle_bert" <classifiers/insults_kaggle_bert.json>`.
With the optional ``-d`` parameter all the data required to run selected pipeline will be **downloaded**.

Python code
~~~~~~~~~~~

One can also use these configs in python code.

**INTERACT** To download required data one have to set ``download`` parameter to ``True``.
Then one can build and interact a model from configuration file:

.. code:: python

    from deeppavlov import build_model

    model = build_model('insults_kaggle_bert', download=True)  # in case of necessity to download some data

    model = build_model('insults_kaggle_bert', download=False)  # otherwise

    print(model(["You are dumb", "He lay flat on the brown, pine-needled floor of the forest"]))

    >>> ['Insult', 'Not Insult']

**TRAIN** Also training can be run in the following way:

.. code:: python

    from deeppavlov import train_model

    model = train_model('insults_kaggle_bert', download=True)  # in case of necessity to download some data

    model = train_model('insults_kaggle_bert', download=False)  # otherwise

BERT models
-----------

BERT (Bidirectional Encoder Representations from Transformers) [4]_ is a Transformer pre-trained on masked language model
and next sentence prediction tasks. This approach showed state-of-the-art results on a wide range of NLP tasks in
English.

**deeppavlov.models.torch_bert.torch_transformers_classifier.TorchTransformersClassifierModel** (see :doc:`here </apiref/models/torch_bert>`) provides easy to use
solution for classification problem using pre-trained BERT.
Several **pre-trained English, multi-lingual and Russian BERT** models are provided in
:doc:`our BERT documentation </features/models/bert>`.

Two main components of BERT classifier pipeline in DeepPavlov are
``deeppavlov.models.preprocessors.torch_transformers_preprocessor.TorchTransformersPreprocessor``
and ``deeppavlov.models.torch_bert.torch_transformers_classifier.TorchTransformersClassifierModel`` (see :doc:`here </apiref/models/torch_bert>`).
The ``deeppavlov.models.torch_bert.torch_transformers_classifier.TorchTransformersClassifierModel`` class supports any Transformer-based model.

Non-processed texts should be given to ``torch_transformers_preprocessor`` for tokenization on subtokens,
encoding subtokens with their indices and creating tokens and segment masks.
If one processed classes to one-hot labels in pipeline, ``one_hot_labels`` should be set to ``true``.

``torch_transformers_classifier`` has a dense layer of number of classes size upon pooled outputs of Transformer encoder,
it is followed by ``softmax`` activation (``sigmoid`` if ``multilabel`` parameter is set to ``true`` in config).

Neural Networks on PyTorch
--------------------------

**deeppavlov.models.classifiers.TorchClassificationModel** (see :doc:`here </apiref/models/classifiers>`)
could be used for implementation of different neural network configurations for classification task.

If you want to build your own architecture for **text classification** tasks, do the following:

    .. code:: python

        from deeppavlov.models.classifiers.torch_classification_model import TorchTextClassificationModel

        class MyModel(TorchTextClassificationModel):

            def my_network_architecture(self, **kwargs):
                model = <create Torch model using parameters from kwargs>
                return model

In the config file, assign ``"class_name": "module.path.to.my.model.file:MyModel"``
and ``"model_name": "my_network_architecture"`` in the dictionary with the main model.

If you want to build your own **PyTorch**-based model for **some other NLP** task, do the following:

    .. code:: python

        from deeppavlov.core.models.torch_model import TorchModel

        class MyModel(TorchModel):

            def train_on_batch(x, y, *args, **kwargs):
                <your code here>
                return loss

            def __call__(data, *args, **kwargs):
                <your code here>
                return predictions

            def my_network_architecture(self, **kwargs):
                model = <create Torch model using parameters from kwargs>
                return model

In the config file, assign ``"class_name": "module.path.to.my.model.file:MyModel"``
and ``"model_name": "my_network_architecture"`` in the dictionary with the main model.

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

We also provide with **pre-trained models** for classification on "AG News" dataset,
"Detecting Insults in Social Commentary", Twitter sentiment in Russian dataset.

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
replies [1]_. It was automatically labeled.
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


+------------------+--------------------+------+-------------------------------------------------------------------------------------------------+-------------+--------+--------+-----------+
| Task             | Dataset            | Lang | Model                                                                                           | Metric      | Valid  | Test   | Downloads |
+==================+====================+======+=================================================================================================+=============+========+========+===========+
| Insult detection | `Insults`_         | En   | :config:`English BERT <classifiers/insults_kaggle_bert.json>`                                   | ROC-AUC     | 0.9327 | 0.8602 |  1.1 Gb   |
+------------------+--------------------+      +-------------------------------------------------------------------------------------------------+-------------+--------+--------+-----------+
| Sentiment        |`SST`_              |      | :config:`5-classes SST on conversational BERT <classifiers/sentiment_sst_conv_bert.json>`       | Accuracy    | 0.6293 | 0.6626 |  1.1 Gb   |
+------------------+--------------------+------+-------------------------------------------------------------------------------------------------+-------------+--------+--------+-----------+
| Sentiment        |`Twitter mokoron`_  | Ru   | :config:`RuWiki+Lenta emb w/o preprocessing <classifiers/sentiment_twitter.json>`               | F1-macro    | 0.9965 | 0.9961 |  6.2 Gb   |
+                  +--------------------+      +-------------------------------------------------------------------------------------------------+-------------+--------+--------+-----------+
|                  |`RuSentiment`_      |      | :config:`Multi-language BERT <classifiers/rusentiment_bert.json>`                               | F1-weighted | 0.6787 | 0.7005 |  1.3 Gb   |
+                  +                    +      +-------------------------------------------------------------------------------------------------+             +--------+--------+-----------+
|                  |                    |      | :config:`Conversational RuBERT <classifiers/rusentiment_convers_bert.json>`                     |             | 0.739  | 0.7724 |  1.5 Gb   |
+------------------+--------------------+------+-------------------------------------------------------------------------------------------------+-------------+--------+--------+-----------+

.. _`DSTC 2`: http://camdial.org/~mh521/dstc/
.. _`Insults`: https://www.kaggle.com/c/detecting-insults-in-social-commentary
.. _`AG News`: https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html
.. _`Twitter mokoron`: http://study.mokoron.com/
.. _`RuSentiment`: http://text-machine.cs.uml.edu/projects/rusentiment/
.. _`SentiRuEval`: http://www.dialog-21.ru/evaluation/2016/sentiment/
.. _`Yahoo-L31`: https://webscope.sandbox.yahoo.com/catalog.php?datatype=l
.. _`Yahoo-L6`: https://webscope.sandbox.yahoo.com/catalog.php?datatype=l
.. _`SST`: https://nlp.stanford.edu/sentiment/index.html

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

How to improve the performance
------------------------------

-  One can use FastText [2]_ to train embeddings that are better suited
   for considered datasets.
-  One can use some custom preprocessing to clean texts.
-  One can use ELMo [3]_ or BERT [4]_.
-  All the parameters should be tuned on the validation set.

References
----------

.. [1] Ю. В. Рубцова. Построение корпуса текстов для настройки тонового классификатора // Программные продукты и системы, 2015, №1(109), –С.72-78

.. [2] P. Bojanowski\ *, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information.

.. [3] Peters, Matthew E., et al. "Deep contextualized word representations." arXiv preprint arXiv:1802.05365 (2018).

.. [4] Devlin J. et al. Bert: Pre-training of deep bidirectional transformers for language understanding //arXiv preprint arXiv:1810.04805. – 2018.
