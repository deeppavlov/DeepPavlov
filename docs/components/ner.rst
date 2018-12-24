Named Entity Recognition (NER)
==============================

Train and use the model
-----------------------

Any pre-trained model can be used for inference from both Command Line Interface (CLI) and Python. Before using the
model make sure that all required packages are installed using the command:

.. code:: bash

    python -m deeppavlov install ner_ontonotes

To use a pre-trained model from CLI use the following command:

.. code:: bash

    python deeppavlov/deep.py interact ner_ontonotes [-d]

where ``ner_ontonotes`` is the name of the config and ``-d`` is an optional download key. The key ``-d`` is used
to download the pre-trained model along with embeddings and all other files needed to run the model. Other possible
commands are ``train``, ``evaluate``, and ``download``,


Here is the list of all available configs:

.. table::
    :widths: auto

    +--------------------------------------------------+-------------------+-----------------+------------+----------+
    | Model                                            | Dataset           | Embeddings Size | Model Size | F1 score |
    +==================================================+===================+=================+============+==========+
    | :config:`ner_ontonotes <ner/ner_ontonotes.json>` | OnotoNotes        | 331 MB          | 7.8 MB     | 87.07    |
    +--------------------------------------------------+-------------------+-----------------+------------+----------+
    | :config:`ner_rus <ner/ner_rus.json>`             | Collection3 [13]_ | 1.0 GB          | 5.6 MB     | 95.25    |
    +--------------------------------------------------+-------------------+-----------------+------------+----------+
    | :config:`ner_dstc <ner/ner_dstc.json>`           | DSTC2             | ---             | 626 KB     | 97.17    |
    +--------------------------------------------------+-------------------+-----------------+------------+----------+
    | :config:`ner_conll2003 <ner/ner_conll2003.json>` | CoNLL-2003        | 331 MB          | 3.1 MB     | 89.94    |
    +--------------------------------------------------+-------------------+-----------------+------------+----------+

Models can be used from Python using the following code:

.. code:: python

    from deeppavlov import configs, build_model

    ner_model = build_model(configs.ner.ner_ontonotes, download=True)
    ner_model(['Bob Ross lived in Florida'])
    >>> [[['Bob', 'Ross', 'lived', 'in', 'Florida']], [['B-PERSON', 'I-PERSON', 'O', 'O', 'B-GPE']]]

The model also can be trained from the Python:

.. code:: python

    from deeppavlov import configs, train_model
    ner_model = train_model(configs.ner.ner_ontonotes)

The data for training should be placed in the folder provided in the config:

.. code:: python

    from deeppavlov.core.commands.utils import parse_config
    config_dict = parse_config(configs.ner.ner_ontonotes)
    print(config_dict['dataset_reader']['data_path'])
    >>> '~/.deeppavlov/downloads/ontonotes_ner_data'

There must be three txt files: train.txt, valid.txt, and test.txt.


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

Configuration of the model
--------------------------

Configuration of the model can be performed in code or in JSON
configuration file. To train the model you need to specify four groups
of parameters:

-  ``dataset_reader``
-  ``dataset_iterator``
-  ``chainer``
-  ``train``

In the subsequent text we show the parameter specification in config
file. However, the same notation can be used to specify parameters in
code by replacing the JSON with python dictionary.

Dataset Reader
~~~~~~~~~~~~~~

The dataset reader is a class which reads and parses the data. It
returns a dictionary with three fields: "train", "test", and "valid".
The basic dataset reader is "conll2003\_reader". The dataset reader
config part with "conll2003\_reader" should look like:

::

    "dataset_reader": {
        "class_name": "conll2003_reader",
        "data_path": "/home/user/Data/conll2003/"
    }

where "class_name" refers to the basic ner dataset reader class and data\_path
is the path to the folder with three files, namely: "train.txt",
"valid.txt", and "test.txt". Each file should contain data in the format
presented in *Training data* section. Each line in the file may contain
additional information such as POS tags. However, the token must be the
first in line and NER tag must be the last.

Dataset Iterator
~~~~~~~~~~~~~~~~

For simple batching and shuffling you can use
"data\_learning\_iterator". The part of the configuration file for the
dataset looks like:
``"dataset_iterator": {     "class_name": "data_learning_iterator" }``

There is no additional parameters in this part.

Chainer
~~~~~~~

The chainer part of the configuration file contains the specification of
the neural network model and supplementary things such as vocabularies.
Chainer should be defined as follows:

::

    "chainer": {
        "in": ["x"],
        "in_y": ["y"],
        "pipe": [
          ...
        ],
        "out": ["y_predicted"]
    }

The inputs and outputs must be specified in the pipe. "in" means regular
input that is used for inference and train mode. "in\_y" is used for
training and usually contains ground truth answers. "out" field stands
for model prediction. The model inside the pipe must have output
variable with name "y\_predicted" so that "out" knows where to get
predictions. The major part of "chainer" is "pipe". The "pipe" contains
the pre-processing modules, vocabularies and model. Firstly we define
pre-processing:

::

    "pipe": [
          {
            "in": ["x"],
            "class_name": "lazy_tokenizer",
            "out": ["x"]
          },
          {
            "in": ["x"],
            "class_name": "str_lower",
            "out": ["x_lower"]
          },
          {
            "in": ["x"],
            "class_name": "mask",
            "out": ["mask"]
          },
          {
            "in": ["x_lower"],
            "class_name": "sanitizer",
            "nums": true,
            "out": ["x_san"]
          },
          {
            "in": ["x"],
            "class_name": "char_splitter",
            "out": ["x_char"]
          },
    ]

Module str\_lower performs lowercasing. Module lazy\_tokenizer performs
tokenization if the elements of the batch are strings but not tokens.
The mask module prepares masks for the network. It serves to cope with
different lengths inputs inside the batch. The mask is a matrix filled
with ones and zeros. For instance, for two sentences batch with lengths
2 and 3 the mask will be [[1, 1, 0],[1, 1, 1]]. The ``sanitizer`` is
used for removing diacritical signs and replacing all digits with ones.
The ``char_splitter`` splits tokens into characters.

Then vocabularies must be defined:

::

    "pipe": [
          ...
          {
            "in": ["x_lower"],
            "id": "word_vocab",
            "class_name": "simple_vocab",
            "pad_with_zeros": true,
            "fit_on": ["x_lower"],
            "save_path": "slotfill_dstc2/word.dict",
            "load_path": "slotfill_dstc2/word.dict",
            "out": ["x_tok_ind"]
          },
          {
            "in": ["y"],
            "id": "tag_vocab",
            "class_name": "simple_vocab",
            "pad_with_zeros": true,
            "fit_on": ["y"],
            "save_path": "slotfill_dstc2/tag.dict",
            "load_path": "slotfill_dstc2/tag.dict",
            "out": ["y_ind"]
          },
          {
            "in": ["x_char"],
            "id": "char_vocab",
            "class_name": "simple_vocab",
            "pad_with_zeros": true,
            "fit_on": ["x_char"],
            "save_path": "ner_conll2003/char.dict",
            "load_path": "ner_conll2003/char.dict",
            "out": ["x_char_ind"]
          },
          ...
    ]

Parameters for vocabulary are:

-  ``id`` - the name of the vocabulary which will be used in other
   models
-  ``class_name`` - equal to ``"simple_vocab"``
-  ``fit_on`` - on which data part of the data the vocabulary should
   be fitted (built), possible options are ["x"] or ["y"]
-  ``save_path`` - path to a new file to save the vocabulary
-  ``load_path`` - path to an existing vocabulary (ignored if there
   is no files)
-  ``pad_with_zeros``: whether to pad the resulting index array with
   zeros or not

Vocabularies are used for holding sets of tokens, tags, or characters.
They assign indices to elements of given sets an allow conversion from
tokens to indices and vice versa. Conversion of such kind is needed to
perform lookup in embeddings matrices and compute cross-entropy between
predicted probabilities and target values. For each vocabulary
"simple\_vocab" model is used. "fit\_on" parameter defines on which part
of the data the vocabulary is built. ["x"] stands for the x part of the
data (tokens) and ["y"] stands for the y part (tags). We can also
assemble character-level vocabularies by changing the value of "level"
parameter: "char" instead of "token".

Then the embeddings must be initialized along with embedding matrices:

::

    "pipe": [
        ...
        {
            "in": ["x_san"],
            "id": "glove_emb",
            "class_name": "glove",
            "pad_zero": true,
            "load_path": "embeddings/glove.6B.100d.txt",
            "out": ["x_emb"]
        },
        {
            "id": "embeddings",
            "class_name": "emb_mat_assembler",
            "embedder": "#glove_emb",
            "vocab": "#word_vocab"
          },
          {
            "id": "embeddings_char",
            "class_name": "emb_mat_assembler",
            "character_level": true,
            "emb_dim": 32,
            "embedder": "#glove_emb",
            "vocab": "#char_vocab"
        },
        ...
    ]

The component ``glove_emb`` creates an embedder from GloVe embeddings.
It can be used as a source for the network or the embedding matrix can
be assembled with ``emb_mat_assembler``. The character level embeddings
can be assembled with the source embedder two.

Then the network is defined by the following part of JSON config:

::

    "pipe": [
        ...
        {
            "in": ["x_emb", "mask", "x_char_ind", "cap"],
            "in_y": ["y_ind"],
            "out": ["y_predicted"],
            "class_name": "ner",
            "main": true,
            "token_emb_dim": "#glove_emb.dim",
            "n_hidden_list": [128],
            "net_type": "rnn",
            "cell_type": "lstm",
            "use_cudnn_rnn": true,
            "n_tags": "#tag_vocab.len",
            "capitalization_dim": "#capitalization.dim",
            "char_emb_dim": "#embeddings_char.dim",
            "save_path": "ner_conll2003/model_no_pos",
            "load_path": "ner_conll2003/model_no_pos",
            "char_emb_mat": "#embeddings_char.emb_mat",
            "use_crf": true,
            "use_batch_norm": true,
            "embeddings_dropout": true,
            "top_dropout": true,
            "intra_layer_dropout": true,
            "l2_reg": 0,
            "learning_rate": 1e-2,
            "dropout_keep_prob": 0.7
        },
        ...
    ]

All network parameters are:

-  ``in`` - inputs to be taken from the shared memory. Treated as x. They are used both during the training and inference.
-  ``in_y`` - the target or y input to be taken from shared memory. This input is used during the training.
-  ``class_name`` - the name of the model to be used. In this case we use 'ner' model originally imported from
   ``deeppavlov.models.ner``. We use only 'ner' name relying on the @registry decorator.
-  ``main`` - (reserved for future use) a boolean parameter defining whether this is the main model.
-  ``save_path`` - path to the new file where the model will be saved
-  ``load_path`` - path to a pretrained model from where it will be loaded.

Other parameters are described in :doc:`NerNetwork </apiref/models/ner>` class.

The output of the network are indices of tags predicted by the network.
They must be converted back to the tag strings. This operation is
performed by already created vocabulary:

::

    "pipe": [
        ...
          {
            "ref": "tag_vocab",
            "in": ["y_predicted"],
            "out": ["tags"]
          }
        ...

In this part of config reusing pattern is used. The ``ref`` parameter
serves to refer to already existing component via ``id``. This part also
illustrate omidirectionality of the vocabulary. When strings are passed
to the vocab, it convert them into indices. When the indices are passed
to the vocab, they are converted to the tag strings.

You can see all parts together in ``deeeppavlov/configs/ner/ner_conll2003.json``.

OntoNotes NER
-------------

A pre-trained model for solving OntoNotes task can be used as following:

.. code:: python

    from deeppavlov import build_model, configs

    ner_model = build_model(configs.ner.ner_ontonotes , download=True)
    ner_model(['Computer Sciences Corp. is close to making final an agreement to buy Cleveland Consulting Associates'])

Or from command line:

.. code:: bash

    python -m deeppavlov interact ner_ontonotes [-d]

The F1 scores of this model on test part of OntoNotes is presented in table below.

+--------------------------------+--------------------+
| Model                          | F1 score           |
+================================+====================+
| DeepPavlov                     | **87.07** ± 0.21   |
+--------------------------------+--------------------+
| Strubell at al. (2017) [1]_    | 86.84 ± 0.19       |
+--------------------------------+--------------------+
| Chiu and Nichols (2016) [2]_   | 86.19 ± 0.25       |
+--------------------------------+--------------------+
| Spacy                          | 85.85              |
+--------------------------------+--------------------+
| Durrett and Klein (2014) [3]_  | 84.04              |
+--------------------------------+--------------------+
| Ratinov and Roth (2009) [4]_   | 83.45              |
+--------------------------------+--------------------+

Scores by entity type are presented in the table below:

+-----------------+------------+
| Tag             | F1 score   |
+=================+============+
| TOTAL           | 87.07      |
+-----------------+------------+
| CARDINAL        | 82.80      |
+-----------------+------------+
| DATE            | 84.87      |
+-----------------+------------+
| EVENT           | 68.39      |
+-----------------+------------+
| FAC             | 68.07      |
+-----------------+------------+
| GPE             | 94.61      |
+-----------------+------------+
| LANGUAGE        | 62.91      |
+-----------------+------------+
| LAW             | 48.27      |
+-----------------+------------+
| LOC             | 72.39      |
+-----------------+------------+
| MONEY           | 87.79      |
+-----------------+------------+
| NORP            | 94.27      |
+-----------------+------------+
| ORDINAL         | 79.53      |
+-----------------+------------+
| ORG             | 85.59      |
+-----------------+------------+
| PERCENT         | 89.41      |
+-----------------+------------+
| PERSON          | 91.67      |
+-----------------+------------+
| PRODUCT         | 58.90      |
+-----------------+------------+
| QUANTITY        | 77.93      |
+-----------------+------------+
| TIME            | 62.50      |
+-----------------+------------+
| WORK\_OF\_ART   | 53.17      |
+-----------------+------------+

Results
-------

The NER network component reproduces the architecture from the paper
"*Application of a Hybrid Bi-LSTM-CRF model to the task of Russian Named
Entity Recognition*\ " https://arxiv.org/pdf/1709.09686.pdf, which is
inspired by LSTM+CRF architecture from
https://arxiv.org/pdf/1603.01360.pdf.

Bi-LSTM architecture of NER network was tested on three datasets:

-  Gareev corpus [5]_ (obtainable by request to authors)
-  FactRuEval 2016 [6]_
-  Persons-1000 [7]_

The F1 measure for our model along with the results of other published
solutions are provided in the table below:

+-------------------------------------------------------+--------------------+----------------+-------------------+
| Models                                                | Gareev’s dataset   | Persons-1000   | FactRuEval 2016   |
+=======================================================+====================+================+===================+
| Gareev et al.  [5]_  (Linguistic features + CRF)      | 75.05              |                |                   |
+-------------------------------------------------------+--------------------+----------------+-------------------+
| Malykh et al. [8]_ (Character level CNN)              | 62.49              |                |                   |
+-------------------------------------------------------+--------------------+----------------+-------------------+
| Trofimov [12]_ (regex and dictionaries)               |                    | 95.57          |                   |
+-------------------------------------------------------+--------------------+----------------+-------------------+
| Sysoev et al. [9]_ (dictionaries and embeddings + SVM)|                    |                | 74.67             |
+-------------------------------------------------------+--------------------+----------------+-------------------+
| Ivanitsky et al. [10]_ (SVM + embeddings)             |                    |                | **87.88**         |
+-------------------------------------------------------+--------------------+----------------+-------------------+
| Mozharova et al.  [11]_ (two stage CRF)               |                    | 97.21          |                   |
+-------------------------------------------------------+--------------------+----------------+-------------------+
| Our (Bi-LSTM+CRF)                                     | **87.17**          | **99.26**      | 82.10             |
+-------------------------------------------------------+--------------------+----------------+-------------------+

To run Russian NER model use the following code:

.. code:: python

    from deeppavlov import build_model, configs

    ner_model = build_model(configs.ner.ner_rus , download=True)
    ner_model(['Компания « Андэк » , специализирующаяся на решениях для обеспечения безопасности бизнеса , сообщила о том , что Вячеслав Максимов , заместитель генерального директора компании , возглавил направление по оптимизации процессов управления информационной безопасностью '])


Few-shot Language-Model based
-----------------------------

It is possible to get a clod-start baseline from just a few samples of labeled data in a couple of seconds. The solution
is based on a Language Model trained on open domain corpus. On top of the LM a SVM classification layer is placed. It is
possible to start from as few as 10 sentences containing entities of interest.

The data for training this model should be collected the following way. Given a collection of `N` sentences without
markup, sequentially markup sentences until the total number of sentences with entity of interest become equal
`K`. During the training both sentences with and without markup are used.


Mean chunk-wise F1 scores for Russian language on 10 sentences with entities :

+---------+-------+
|PER      | 84.85 |
+---------+-------+
|LOC      | 68.41 |
+---------+-------+
|ORG      | 32.63 |
+---------+-------+

(the total number of training sentences is bigger and defined by the distribution of sentences with / without entities).

The model can be trained using CLI:

.. code:: bash

    python -m deeppavlov train ner_few_shot_ru

you have to provide the `train.txt`, `valid.txt`, and `test.txt` files in the format described in the `Training data`_
section. The files must be in the `ner_few_shot_data` folder as described in the `dataset_reader` part of the config
:config:`ner/ner_few_shot_ru_train.json <ner/ner_few_shot_ru.json>` .

To train and use the model from python code the following snippet can be used:

.. code:: python

    from deeppavlov import configs, train_model

    ner_model = train_model(configs.ner.ner_few_shot_ru, download=True)
    ner_model(['Example sentence'])

Warning! This model can take a lot of time and memory if the number of sentences is greater than 1000!

If a lot of data is available the few-shot setting can be simulated with special `dataset_iterator`. For this purpose
the config
:config:`ner/ner_few_shot_ru_train.json <ner/ner_few_shot_ru_simulate.json>` . The following code can be used for this
simulation:

.. code:: python

    from deeppavlov import configs, train_model

    ner_model = train_model(configs.ner.ner_few_shot_ru_simulate, download=True)

In this config the `Collection dataset <http://labinform.ru/pub/named_entities/descr_ne.htm>`__ is used. However, if
there are files `train.txt`, `valid.txt`, and `test.txt` in the `ner_few_shot_data` folder they will be used instead.


To use existing few-shot model use the following python interface can be used:

.. code:: python

    from deeppavlov import configs, build_model
    ner_model = build_model(configs.ner.ner_few_shot_ru)
    ner_model([['Example', 'sentence']])
    ner_model(['Example sentence'])



Literature
----------

.. [1] Strubell at al. (2017) Strubell, Emma, et al. "Fast and accurate
    entity recognition with iterated dilated convolutions." Proceedings of
    the 2017 Conference on Empirical Methods in Natural Language Processing.
    2017.

.. [2] Jason PC Chiu and Eric Nichols. 2016. Named entity recognition
    with bidirectional lstm-cnns. Transactions of the Association for
    Computational Linguistics, 4:357–370.

.. [3] Greg Durrett and Dan Klein. 2014. A joint model for entity
    analysis: Coreference, typing and linking. Transactions of the
    Association for Computational Linguistics, 2:477–490.

.. [4] Lev Ratinov and Dan Roth. 2009. Design challenges and
    misconceptions in named entity recognition. In Proceedings of the
    Thirteenth Conference on Computational Natural Language Learning, pages
    147–155. Association for Computational Linguistics.

.. [5] Rinat Gareev, Maksim Tkachenko, Valery Solovyev, Andrey
    Simanovsky, Vladimir Ivanov: Introducing Baselines for Russian Named
    Entity Recognition. Computational Linguistics and Intelligent Text
    Processing, 329 -- 342 (2013).

.. [6] https://github.com/dialogue-evaluation/factRuEval-2016

.. [7] http://ai-center.botik.ru/Airec/index.php/ru/collections/28-persons-1000

.. [8] Malykh, Valentin, and Alexey Ozerin. "Reproducing Russian NER
    Baseline Quality without Additional Data." CDUD@ CLA. 2016.

.. [9] Sysoev A. A., Andrianov I. A.: Named Entity Recognition in
    Russian: the Power of Wiki-Based Approach. dialog-21.ru

.. [10] Ivanitskiy Roman, Alexander Shipilo, Liubov Kovriguina: Russian
    Named Entities Recognition and Classification Using Distributed Word and
    Phrase Representations. In SIMBig, 150 – 156. (2016).

.. [11] Mozharova V., Loukachevitch N.: Two-stage approach in Russian
    named entity recognition. In Intelligence, Social Media and Web (ISMW
    FRUCT), 2016 International FRUCT Conference, 1 – 6 (2016)

.. [12] Trofimov, I.V.: Person name recognition in news articles based on
    the persons- 1000/1111-F collections. In: 16th All-Russian Scientific C
    onference Digital Libraries: Advanced Methods and Technologies, Digital
    Collections, RCDL 2014,pp. 217 – 221 (2014).

.. [13] Mozharova V., Loukachevitch N., Two-stage approach in Russian named
    entity recognition // International FRUCT Conference on Intelligence,
    Social Media and Web, ISMW FRUCT 2016. Saint-Petersburg; Russian Federation,
    DOI 10.1109/FRUCT.2016.7584769
