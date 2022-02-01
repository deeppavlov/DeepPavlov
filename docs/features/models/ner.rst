Named Entity Recognition (NER)
==============================

Train and use the model
-----------------------

There are three main types of models available: Standard RNN-based model, BERT-based model (on TensorFlow and PyTorch), and the hybrid model.
To see details about BERT based models see :doc:`here </features/models/bert>`. The last one, the hybrid model, reproduces the architecture proposed
in the paper `A Deep Neural Network Model for the Task of Named Entity Recognition <http://www.ijmlc.org/show-83-881-1.html>`__.
Any pre-trained model can be used for inference from both Command Line Interface (CLI) and Python. Before using the
model make sure that all required packages are installed using the command:

.. code:: bash

    python -m deeppavlov install ner_ontonotes_bert_torch

To use a pre-trained model from CLI use the following command:

.. code:: bash

    python deeppavlov/deep.py interact ner_ontonotes_bert_torch [-d]

where ``ner_ontonotes_bert_torch`` is the name of the config and ``-d`` is an optional download key. The key ``-d`` is used
to download the pre-trained model along with embeddings and all other files needed to run the model. Other possible
commands are ``train``, ``evaluate``, and ``download``,


Here is the list of all available configs:


.. table::
    :widths: auto

    +------------------------------------------------------------------------+--------------------+----------+-----------------+------------+------------+
    | Model                                                                  | Dataset            | Language | Embeddings Size | Model Size |  F1 score  |
    +========================================================================+====================+==========+=================+============+============+
    | :config:`ner_rus_bert_torch <ner/ner_rus_bert_torch.json>`             | Collection3 [1]_   | Ru       | 700 MB          |   2.0 GB   | **97.7**   |
    +------------------------------------------------------------------------+                    +          +-----------------+------------+------------+
    | :config:`ner_collection3_m1 <ner/ner_collection3_m1.json>`             |                    |          | 1.1 GB          |    1 GB    |   97.8     |
    +------------------------------------------------------------------------+                    +          +-----------------+------------+------------+
    | :config:`ner_rus <ner/ner_rus.json>`                                   |                    |          | 1.0 GB          |   5.6 MB   |   95.1     |
    +------------------------------------------------------------------------+--------------------+----------+-----------------+------------+------------+
    | :config:`<ner/ner_ontonotes_bert_mult_torch.json>`                     | Ontonotes          | Multi    | 700 MB          |   2.0 GB   | **87.2**   |
    +------------------------------------------------------------------------+                    +----------+-----------------+------------+------------+
    | :config:`ner_ontonotes_bert_torch <ner/ner_ontonotes_bert_torch.json>` |                    | En       | 400 MB          |   1.3 GB   |   87.9     |
    +------------------------------------------------------------------------+                    +          +-----------------+------------+------------+
    | :config:`ner_ontonotes <ner/ner_ontonotes.json>`                       |                    |          | 331 MB          |   7.8 MB   |   86.7     |
    +------------------------------------------------------------------------+--------------------+          +-----------------+------------+------------+
    | :config:`ner_conll2003_bert <ner/ner_conll2003_bert.json>`             | CoNLL-2003         |          | 400 MB          |   850 MB   |   91.7     |
    +------------------------------------------------------------------------+                    +          +-----------------+------------+------------+
    | :config:`ner_conll2003_torch_bert <ner/ner_conll2003_torch_bert.json>` |                    |          | ---             |   1.3 GB   |   90.7     |
    +------------------------------------------------------------------------+                    +          +-----------------+------------+------------+
    | :config:`ner_conll2003 <ner/ner_conll2003.json>`                       |                    |          | 331 MB          |   3.1 MB   |   89.9     |
    +------------------------------------------------------------------------+                    +          +-----------------+------------+------------+
    | :config:`conll2003_m1 <ner/conll2003_m1.json>`                         |                    |          | 339 MB          |  359.7 MB  | **91.9**   |
    +------------------------------------------------------------------------+--------------------+----------+-----------------+------------+------------+

Models can be used from Python using the following code:

.. code:: python

    from deeppavlov import configs, build_model

    ner_model = build_model(configs.ner.ner_ontonotes_bert_torch, download=True)

    ner_model(['Bob Ross lived in Florida'])
    >>> [[['Bob', 'Ross', 'lived', 'in', 'Florida']], [['B-PERSON', 'I-PERSON', 'O', 'O', 'B-GPE']]]

The model also can be trained from the Python:

.. code:: python

    from deeppavlov import configs, train_model

    ner_model = train_model(configs.ner.ner_ontonotes_bert_torch)

The data for training should be placed in the folder provided in the config:

.. code:: python

    from deeppavlov import configs, train_model
    from deeppavlov.core.commands.utils import parse_config
    
    config_dict = parse_config(configs.ner.ner_ontonotes_bert_torch)

    print(config_dict['dataset_reader']['data_path'])
    >>> '~/.deeppavlov/downloads/ontonotes'

There must be three txt files: train.txt, valid.txt, and test.txt. Furthermore the `data_path` can be changed from code.
The format of the data is described in the `Training data`_ section.


.. _ner_multi_bert:

Multilingual BERT Zero-Shot Transfer
------------------------------------

Multilingual BERT models allow to perform zero-shot transfer from one language to another. The model
:config:`ner_ontonotes_bert_mult_torch <ner/ner_ontonotes_bert_mult_torch.json>` was trained on OntoNotes corpus which has 19 types
in the markup schema. The model performance was evaluated on Russian corpus Collection 3 [1]_. Results of the
transfer are presented in the table below.

+---------+-------+
|TOTAL    | 79.39 |
+---------+-------+
|PER      | 95.74 |
+---------+-------+
|LOC      | 82.62 |
+---------+-------+
|ORG      | 55.68 |
+---------+-------+


The following Python code can be used to infer the model:

.. code:: python

    from deeppavlov import configs, build_model

    ner_model = build_model(configs.ner.ner_ontonotes_bert_mult_torch, download=True)

    ner_model(['Curling World Championship will be held in Antananarivo'])
    >>> (['Curling', 'World', 'Championship', 'will', 'be', 'held', 'in', 'Antananarivo']],
    [['B-EVENT', 'I-EVENT', 'I-EVENT', 'O', 'O', 'O', 'O', 'B-GPE'])

    ner_model(['Mistrzostwa Świata w Curlingu odbędą się w Antananarivo'])
    >>> (['Mistrzostwa', 'Świata', 'w', 'Curlingu', 'odbędą', 'się', 'w', 'Antananarivo']],
    [['B-EVENT', 'I-EVENT', 'I-EVENT', 'I-EVENT', 'O', 'O', 'O', 'B-GPE'])

    ner_model(['Чемпионат мира по кёрлингу пройдёт в Антананариву'])
    >>> (['Чемпионат', 'мира', 'по', 'кёрлингу', 'пройдёт', 'в', 'Антананариву'], 
    ['B-EVENT', 'I-EVENT', 'I-EVENT', 'I-EVENT', 'O', 'O', 'B-GPE'])

The list of available tags and their descriptions are presented below.

+--------------+--------------------------------------------------------+
| PERSON       | People including fictional                             |
+--------------+--------------------------------------------------------+
| NORP         | Nationalities or religious or political groups         |
+--------------+--------------------------------------------------------+
| FACILITY     | Buildings, airports, highways, bridges, etc.           |
+--------------+--------------------------------------------------------+
| ORGANIZATION | Companies, agencies, institutions, etc.                |
+--------------+--------------------------------------------------------+
| GPE          | Countries, cities, states                              |
+--------------+--------------------------------------------------------+
| LOCATION     | Non-GPE locations, mountain ranges, bodies of water    |
+--------------+--------------------------------------------------------+
| PRODUCT      | Vehicles, weapons, foods, etc. (Not services)          |
+--------------+--------------------------------------------------------+
| EVENT        | Named hurricanes, battles, wars, sports events, etc.   |
+--------------+--------------------------------------------------------+
| WORK OF ART  | Titles of books, songs, etc.                           |
+--------------+--------------------------------------------------------+
| LAW          | Named documents made into laws                         |
+--------------+--------------------------------------------------------+
| LANGUAGE     | Any named language                                     |
+--------------+--------------------------------------------------------+
| DATE         | Absolute or relative dates or periods                  |
+--------------+--------------------------------------------------------+
| TIME         | Times smaller than a day                               |
+--------------+--------------------------------------------------------+
| PERCENT      | Percentage (including “%”)                             |
+--------------+--------------------------------------------------------+
| MONEY        | Monetary values, including unit                        |
+--------------+--------------------------------------------------------+
| QUANTITY     | Measurements, as of weight or distance                 |
+--------------+--------------------------------------------------------+
| ORDINAL      | “first”, “second”                                      |
+--------------+--------------------------------------------------------+
| CARDINAL     | Numerals that do not fall under another type           |
+--------------+--------------------------------------------------------+

NER task
--------

Named Entity Recognition (NER) is one of the most common tasks in
natural language processing. In most of the cases, NER task can be
formulated as:

*Given a sequence of tokens (words, and maybe punctuation symbols)
provide a tag from a predefined set of tags for each token in the
sequence.*

For NER task there are some common types of entities used as tags:

-  persons
-  locations
-  organizations
-  expressions of time
-  quantities
-  monetary values

Furthermore, to distinguish adjacent entities with the same tag many
applications use BIO tagging scheme. Here "B" denotes beginning of an
entity, "I" stands for "inside" and is used for all words comprising the
entity except the first one, and "O" means the absence of entity.
Example with dropped punctuation:

::

    Bernhard        B-PER
    Riemann         I-PER
    Carl            B-PER
    Friedrich       I-PER
    Gauss           I-PER
    and             O
    Leonhard        B-PER
    Euler           I-PER

In the example above PER means person tag, and "B-" and "I-" are
prefixes identifying beginnings and continuations of the entities.
Without such prefixes, it is impossible to separate Bernhard Riemann
from Carl Friedrich Gauss.

Training data
-------------

To train the neural network, you need to have a dataset in the following
format:

::

    EU B-ORG
    rejects O
    the O
    call O
    of O
    Germany B-LOC
    to O
    boycott O
    lamb O
    from O
    Great B-LOC
    Britain I-LOC
    . O

    China B-LOC
    says O
    time O
    right O
    for O
    Taiwan B-LOC
    talks O
    . O

    ...

The source text is tokenized and tagged. For each token, there is a tag
with BIO markup. Tags are separated from tokens with whitespaces.
Sentences are separated with empty lines.

Dataset is a text file or a set of text files. The dataset must be split
into three parts: train, test, and validation. The train set is used for
training the network, namely adjusting the weights with gradient
descent. The validation set is used for monitoring learning progress and
early stopping. The test set is used for final evaluation of model
quality. Typical partition of a dataset into train, validation, and test
are 80%, 10%, 10%, respectively.


NER-based Model for Sentence Boundary Detection Task
----------------------------------------------------

The task of Sentence Boundary Detection (SBD) is one of the preprocessing tasks in NLP, aiming at splitting
an unpunctuated text into a list of sentences. In a chatbot's architecture, An SBD module can be used as a
preprocessing step to enhance the ability to handle long and complex user's utterances and hence encourage
users to communicate with the chatbot more naturally.

The SBD task can be addressed by firstly reformulating as a Sequence Labeling task, and then applying the
hybrid model mentioned at the beginning of this document. Details of how to use a Sequence Labeling model
to address the SBD task are represented in the paper `Sequence Labeling Approach to the Task of Sentence
Boundary Detection <https://dl.acm.org/doi/abs/10.1145/3380688.3380703>`__. Below is the statistic of the
dataset generated from the DailyDialog dataset [2]_:

+----------------------+---------+
| Number of samples    |   99299 |
+----------------------+---------+
| Number of statements |  111838 |
+----------------------+---------+
| Number of questions  |   37447 |
+----------------------+---------+
| Number of words      | 1139540 |
+----------------------+---------+

Here is the achieved result of training the hybrid model on the above dataset using
the config file :config:`sentseg_dailydialog <sentence_segmentation/sentseg_dailydialog.json>`:

+-----------+-----------+--------+-------+
| Tag       | Precision | Recall |  F1   |
+-----------+-----------+--------+-------+
| Question  |   96.48   | 93.49  | 94.96 |
+-----------+-----------+--------+-------+
| Statement |   96.24   | 96.69  | 96.47 |
+-----------+-----------+--------+-------+
| Overall   |   96.30   | 95.89  | 96.10 |
+-----------+-----------+--------+-------+

The command below is used to download and use the pre-trained model in the CLI:

.. code:: bash

    python -m deeppavlov interact sentseg_dailydialog -d

The model also can be trained from scratch by using the command:

.. code:: bash

    python -m deeppavlov train sentseg_dailydialog





Literature
----------

.. [1] Mozharova V., Loukachevitch N., Two-stage approach in Russian named
    entity recognition // International FRUCT Conference on Intelligence,
    Social Media and Web, ISMW FRUCT 2016. Saint-Petersburg; Russian Federation,
    DOI 10.1109/FRUCT.2016.7584769
.. [2] Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. 2017. DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset. In Proceedings of the 8th International Joint Conference on Natural Language Processing.
