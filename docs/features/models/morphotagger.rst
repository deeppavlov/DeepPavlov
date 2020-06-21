Neural Morphological Tagging
============================

It is an implementation of neural morphological tagger.
As for now (November, 2019) we have two types of models:
the BERT-based ones (available only for Russian) and
the character-based bidirectional LSTM. The BERT model
includes only a dense layer on the top of BERT embedder.
See the `BERT paper <http://arxiv.org/abs/1810.04805>`__
for a more complete description, as well as the
`BERT section <features/models/bert>`__ of the documentation.
We plan to release more BERT-based models in near future.

Most of our models follow
`Heigold et al., 2017. An extensive empirical evaluation of
character-based morphological tagging for 14
languages <http://www.aclweb.org/anthology/E17-1048>`__.
They also achieve the state-of-the-art performance among open source
systems.

The BERT-based model is trained on `Universal
Dependencies corpora <https://universaldependencies.org/>`__
(version 2.3), while all the other models were trained
on Universal Dependencies 2.0 corpora.

+----------------+--------------+-----------------+-------------------------------+------------------+----------------+
|    Language    | Code         | UDPipe accuracy | UDPipe Future accuracy [#f1]_ | Our top accuracy | Model size (MB)|
+================+==============+=================+===============================+==================+================+
| Arabic         | ar           | 88.31           |                               | 90.85            |  23.7          |
+----------------+--------------+-----------------+-------------------------------+------------------+----------------+
| Czech          | cs           | 91.86           |                               | 94.35            |  41.8          |
+----------------+--------------+-----------------+-------------------------------+------------------+----------------+
| English        | en           | 92.53           |                               | 93.00            |  16.9          |
+----------------+--------------+-----------------+-------------------------------+------------------+----------------+
| French         | fr           | 95.25           |                               | 95.45            |  19.0          |
+----------------+--------------+-----------------+-------------------------------+------------------+----------------+
| German         | de           | 76.65           |                               | 83.83            |  18.6          |
+----------------+--------------+-----------------+-------------------------------+------------------+----------------+
| Hindi          | hi           | 87.74           |                               | 90.01            |  21.9          |
+----------------+--------------+-----------------+-------------------------------+------------------+----------------+
| Hungarian      | hu           | 69.52           |                               | 75.34            |  15.4          |
+----------------+--------------+-----------------+-------------------------------+------------------+----------------+
| Italian        | it           | 96.33           |                               | 96.47            |  32.0          |
+----------------+--------------+-----------------+-------------------------------+------------------+----------------+
| Russian        | ru_syntagrus | 93.57           |                               | 96.23            |  48.7          |
+----------------+--------------+-----------------+-------------------------------+------------------+----------------+
| Russian (UD2.3)| ru_syntagrus | 93.5            | 96.90                         | 97.83            |  661           |
+----------------+--------------+-----------------+-------------------------------+------------------+----------------+
| Spanish        | es_ancora    | 96.88           |                               | 97.00            |  20.8          |
+----------------+--------------+-----------------+-------------------------------+------------------+----------------+
| Turkish        | tr           | 86.98           |                               | 88.03            |  16.1          |
+----------------+--------------+-----------------+-------------------------------+------------------+----------------+

.. rubric:: Footnotes

.. [#f1] No models available, only the source code. The scores are taken from
   `Straka. UDPipe 2.0 Prototype at CoNLL 2018 UD Shared Task. <https://www.aclweb.org/anthology/K18-2020.pdf>`__.


===========================
Usage examples.
===========================

Before using the model make sure that all required packages are installed using the command:

.. code:: bash

    python -m deeppavlov install morpho_ru_syntagrus_pymorphy

For Windows platform one has to set `KERAS_BACKEND` to `tensorflow` (it could be done only once):

.. code:: bash

    set "KERAS_BACKEND=tensorflow"

Python:
---------------------------

For Windows platform if one did not set `KERAS_BACKEND` to `tensorflow` from command line it could be done in python code in the following way:

.. code:: python

    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"


.. code:: python

    from deeppavlov import build_model, configs
    model = build_model(configs.morpho_tagger.UD2_0.morpho_ru_syntagrus_pymorphy, download=True)
    sentences = ["Я шёл домой по незнакомой улице.", "Девушка пела в церковном хоре о всех уставших в чужом краю."]
    for parse in model(sentences):
        print(parse)

If you want to use the obtained tags further in Python, just split the output using tabs and newlines.

You may also pass the tokenized sentences instead of raw ones:

.. code:: python

    sentences = [["Я", "шёл", "домой", "по", "незнакомой", "улице", "."]]
    for parse in model(sentences):
        print(parse)

If your data is large, you can call
:meth:`~deeppavlov.core.common.chainer.Chainer.batched_call` method of the model, which will additionally
separate you list of sentences into small batches.

.. code:: python

    from deeppavlov import build_model, configs
    model = build_model(configs.morpho_tagger.UD2_0.morpho_ru_syntagrus_pymorphy, download=True)
    sentences = ["Я шёл домой по незнакомой улице.", "Девушка пела в церковном хоре о всех уставших в чужом краю."]
    for parse in model.batched_call(sentences, batch_size=16):
        print(parse)

::

    1	Я	PRON,Case=Nom|Number=Sing|Person=1	_
    2	шёл	VERB,Aspect=Imp|Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act	_
    3	домой	ADV,Degree=Pos	_
    4	по	ADP	_
    5	незнакомой	ADJ,Case=Dat|Degree=Pos|Gender=Fem|Number=Sing	_
    6	улице	NOUN,Animacy=Inan|Case=Dat|Gender=Fem|Number=Sing	_
    7	.	PUNCT	_

    1	Девушка	NOUN,Animacy=Anim|Case=Nom|Gender=Fem|Number=Sing	_
    2	пела	VERB,Aspect=Imp|Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act	_
    3	в	ADP	_
    4	церковном	ADJ,Case=Loc|Degree=Pos|Gender=Masc|Number=Sing	_
    5	хоре	NOUN,Animacy=Inan|Case=Loc|Gender=Masc|Number=Sing	_
    6	о	ADP	_
    7	всех	PRON,Animacy=Anim|Case=Loc|Number=Plur	_
    8	уставших	VERB,Aspect=Perf|Case=Loc|Number=Plur|Tense=Past|VerbForm=Part|Voice=Act	_
    9	в	ADP	_
    10	чужом	ADJ,Case=Loc|Degree=Pos|Gender=Masc|Number=Sing	_
    11	краю	NOUN,Animacy=Inan|Case=Loc|Gender=Masc|Number=Sing	_
    12	.	PUNCT	_

If you want the output in UD format, try setting ``"data_format": ud`` in the ``tag_output_prettifier`` section
of :config:`configuration file <morpho_tagger/UD2.0/morpho_ru_syntagrus_pymorphy.json>`
you import.

Advanced models (BERT and lemmatized models).
---------------------------------------------

#. For Russian you can use the BERT-based model. It has much higher performance (97.8% instead of 96.2),
   however, you need a more powerful GPU (ideally, 16 GB) to train it. However, the speed
   of inference and training on such GPU is comparable with character-based model.

#. Exclusively for Russian language you can obtain lemmatized UD output by using either the
   :config:`BERT model <morpho_tagger/BERT/morpho_ru_syntagrus_bert.json>`
   :config:`augmented version <morpho_tagger/UD2.0/morpho_ru_syntagrus_pymorphy_lemmatize.json>`
   of Pymorphy model. Both models select the Pymorphy lemma whose tag correspond to the tag
   predicted by the tagger.

   .. code:: python

       from deeppavlov import build_model, configs
       model = build_model(configs.morpho_tagger.BERT.morpho_ru_syntagrus_bert, download=True)
       # model = build_model(configs.morpho_tagger.UD2_0.morpho_ru_syntagrus_pymorphy_lemmatize, download=True)
       sentences = ["Я шёл домой по незнакомой улице.", "Девушка пела в церковном хоре о всех уставших в чужом краю."]
       for parse in model(sentences):
           print(parse)

   ::

       1	Я	я	PRON	_	Case=Nom|Number=Sing|Person=1	_	_	_	_
       2	шёл	идти	VERB	_	Aspect=Imp|Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act	_	_	_	_
       3	домой	домой	ADV	_	Degree=Pos	_	_	_	_
       4	по	по	ADP	_	_	_	_	_	_
       5	незнакомой	незнакомый	ADJ	_	Case=Dat|Degree=Pos|Gender=Fem|Number=Sing	_	_	_	_
       6	улице	улица	NOUN	_	Animacy=Inan|Case=Dat|Gender=Fem|Number=Sing	_	_	_	_
       7	.	.	PUNCT	_	_	_	_	_	_

       1	Девушка	девушка	NOUN	_	Animacy=Anim|Case=Nom|Gender=Fem|Number=Sing	_	_	_	_
       2	пела	петь	VERB	_	Aspect=Imp|Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act	_	_	_	_
       3	в	в	ADP	_	_	_	_	_	_
       4	церковном	церковный	ADJ	_	Case=Loc|Degree=Pos|Gender=Masc|Number=Sing	_	_	_	_
       5	хоре	хор	NOUN	_	Animacy=Inan|Case=Loc|Gender=Masc|Number=Sing	_	_	_	_
       6	о	о	ADP	_	_	_	_	_	_
       7	всех	весь	PRON	_	Animacy=Anim|Case=Loc|Number=Plur	_	_	_	_
       8	уставших	устать	VERB	_	Aspect=Perf|Case=Loc|Number=Plur|Tense=Past|VerbForm=Part|Voice=Act	_	_	_	_
       9	в	в	ADP	_	_	_	_	_	_
       10	чужом	чужой	ADJ	_	Case=Loc|Degree=Pos|Gender=Masc|Number=Sing	_	_	_	_
       11	краю	край	NOUN	_	Animacy=Inan|Case=Loc|Gender=Masc|Number=Sing	_	_	_	_
       12	.	.	PUNCT	_	_	_	_	_	_

Command line:
----------------

If you want to use our models from scratch, do the following
(all the examples are for ru\_syntagrus\_pymorphy model,
change the filenames accordingly to invoke models for other languages):

#. Download data

    .. code:: bash

       python -m deeppavlov download morpho_ru_syntagrus_pymorphy

   To perform all downloads in runtime you can also run all subsequent
   commands with ``-d`` key,

#. To apply a pre-trained ru\_syntagrus\_pymorphy model to ru\_syntagrus test
   data provided it was downloaded using the previous command, run

   .. code:: bash

     python -m deeppavlov.models.morpho_tagger morpho_ru_syntagrus_pymorphy \
     > -f ~/.deeppavlov/downloads/UD2.0_source/ru_syntagrus/ru_syntagrus-ud-test.conllu

   ``-f`` argument points to the path to the test data. If you do not pass it the model expects data from ``stdin``.
   This command writes the output to stdout, you can redirect it using standard ``>`` notation.

   -  By default the ``deeppavlov.models.morpho_tagger`` script expects the data to be in CoNLL-U format,
      however, you can specify input format by using the `-i` key. For example, your input can be in one word per line
      format, in this case you set this key to ``"vertical"``. Note also that you can pass the data from

    .. code:: bash

        echo -e "Мама\nмыла\nраму\n.\n\nВаркалось\n,\nхливкие\nшорьки\nпырялись\nпо\nнаве\n." \
        > | python -m deeppavlov.models.morpho_tagger morpho_ru_syntagrus_pymorphy -i "vertical"

    ::

        1       Мама    NOUN    Animacy=Anim|Case=Nom|Gender=Fem|Number=Sing
        2       мыла    VERB    Aspect=Imp|Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act
        3       раму    NOUN    Animacy=Inan|Case=Acc|Gender=Fem|Number=Sing
        4       .       PUNCT   _

        1       Варкалось       NOUN    Animacy=Anim|Case=Nom|Gender=Masc|Number=Sing
        2       ,       PUNCT   _
        3       хливкие ADJ     Case=Nom|Degree=Pos|Number=Plur
        4       шорьки  NOUN    Animacy=Inan|Case=Nom|Gender=Masc|Number=Plur
        5       пырялись        VERB    Aspect=Imp|Mood=Ind|Number=Plur|Tense=Past|VerbForm=Fin|Voice=Mid
        6       по      ADP     _
        7       наве    NOUN    Animacy=Inan|Case=Dat|Gender=Masc|Number=Sing
        8       .       PUNCT   _


   -   Untokenized sentences (one sentence per line) can be tagged as well, in this case input format should be ``"text"``

    .. code:: bash

        echo -e "Мама мыла раму.\nВаркалось, хливкие шорьки пырялись по наве." \
        > | python -m deeppavlov.models.morpho_tagger morpho_ru_syntagrus_pymorphy -i "text"

    ::

        1       Мама    NOUN    Animacy=Anim|Case=Nom|Gender=Fem|Number=Sing
        2       мыла    VERB    Aspect=Imp|Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act
        3       раму    NOUN    Animacy=Inan|Case=Acc|Gender=Fem|Number=Sing
        4       .       PUNCT   _

        1       Варкалось       NOUN    Animacy=Anim|Case=Nom|Gender=Masc|Number=Sing
        2       ,       PUNCT   _
        3       хливкие ADJ     Case=Nom|Degree=Pos|Number=Plur
        4       шорьки  NOUN    Animacy=Inan|Case=Nom|Gender=Masc|Number=Plur
        5       пырялись        VERB    Aspect=Imp|Mood=Ind|Number=Plur|Tense=Past|VerbForm=Fin|Voice=Mid
        6       по      ADP     _
        7       наве    NOUN    Animacy=Inan|Case=Dat|Gender=Masc|Number=Sing
        8       .       PUNCT   _

   - You can also obtain the output in CoNLL-U format by passing the ``-o ud`` argument:

    .. code:: bash

        echo -e "Мама мыла раму.\nВаркалось, хливкие шорьки пырялись по наве." \
        > | python -m deeppavlov.models.morpho_tagger morpho_ru_syntagrus_pymorphy -i "text" -o "ud"

    ::

        1       Мама    _       NOUN    _       Animacy=Anim|Case=Nom|Gender=Fem|Number=Sing    _       _       _       _
        2       мыла    _       VERB    _       Aspect=Imp|Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act    _       _       _       _
        3       раму    _       NOUN    _       Animacy=Inan|Case=Acc|Gender=Fem|Number=Sing    _       _       _       _
        4       .       _       PUNCT   _       _       _       _       _       _

        1       Варкалось       _       NOUN    _       Animacy=Anim|Case=Nom|Gender=Masc|Number=Sing   _       _       _       _
        2       ,       _       PUNCT   _       _       _       _       _       _
        3       хливкие _       ADJ     _       Case=Nom|Degree=Pos|Number=Plur _       _       _       _
        4       шорьки  _       NOUN    _       Animacy=Inan|Case=Nom|Gender=Masc|Number=Plur   _       _       _       _
        5       пырялись        _       VERB    _       Aspect=Imp|Mood=Ind|Number=Plur|Tense=Past|VerbForm=Fin|Voice=Mid       _       _       _       _
        6       по      _       ADP     _       _       _       _       _       _
        7       наве    _       NOUN    _       Animacy=Inan|Case=Dat|Gender=Masc|Number=Sing   _       _       _       _
        8       .       _       PUNCT   _       _       _       _       _       _


#. To evaluate ru\_syntagrus model on ru\_syntagrus test subset, run

   .. code:: bash

       python -m deeppavlov evaluate morpho_ru_syntagrus_pymorphy

#. To retrain model on ru\_syntagrus dataset, run one of the following
   (the first is for Pymorphy-enriched model)

   .. code:: bash

       python -m deeppavlov train morpho_ru_syntagrus_pymorphy
       python -m deeppavlov train morpho_ru_syntagrus

   Be careful, one epoch takes 2-60 minutes depending on your GPU.

#. To tag Russian sentences from stdin, run

   .. code:: bash

       python -m deeppavlov interact morpho_ru_syntagrus_pymorphy

Read the detailed readme below.

Task description
----------------

Morphological tagging consists in assigning labels, describing word
morphology, to a pre-tokenized sequence of words.
In the most simple case these labels are just part-of-speech (POS)
tags, hence in earlier times of NLP the task was
often referred as POS-tagging. The refined version of the problem
which we solve here performs more fine-grained
classification, also detecting the values of other morphological
features, such as case, gender and number for nouns,
mood, tense, etc. for verbs and so on. Morphological tagging is a
stage of common NLP pipeline, it generates useful
features for further tasks such as syntactic parsing, named entity
recognition or machine translation.

Common output for morphological tagging looks as below. The examples
are for Russian and English language and use the
inventory of tags and features from `Universal Dependencies
project <http://www.universaldependencies.org/guidelines.html>`__.

::

    1   Это PRON    Animacy=Inan|Case=Acc|Gender=Neut|Number=Sing
    2   чутко   ADV Degree=Pos
    3   фиксируют   VERB    Aspect=Imp|Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act
    4   энциклопедические   ADJ Case=Nom|Degree=Pos|Number=Plur
    5   издания NOUN    Animacy=Inan|Case=Nom|Gender=Neut|Number=Plur
    6   .   PUNCT   _
      
    1   Four    NUM NumType=Card
    2   months  NOUN    Number=Plur
    3   later   ADV _
    4   ,   PUNCT   _
    5   we  PRON    Case=Nom|Number=Plur|Person=1|PronType=Prs
    6   were    AUX Mood=Ind|Tense=Past|VerbForm=Fin
    7   married VERB    Tense=Past|VerbForm=Part|Voice=Pass
    8   .   PUNCT   _

The full UD format (see below) includes more columns including lemma and
syntactic information.

Training data
~~~~~~~~~~~~~

Our tagger accepts the data in `CONLL-U
format <http://universaldependencies.org/format.html>`__:

::

    1   Four    four    NUM CD  NumType=Card    2   nummod  _   _
    2   months  month   NOUN    NNS Number=Plur 3   obl:npmod   _   _
    3   later   later   ADV RB  _   7   advmod  _   SpaceAfter=No
    4   ,   ,   PUNCT   ,   _   7   punct   _   _
    5   we  we  PRON    PRP Case=Nom|Number=Plur|Person=1|PronType=Prs  7   nsubj:pass  _   _
    6   were    be  AUX VBD Mood=Ind|Tense=Past|VerbForm=Fin    7   aux:pass    _   _
    7   married marry   VERB    VBN Tense=Past|VerbForm=Part|Voice=Pass 0   root    _   SpaceAfter=No
    8   .   .   PUNCT   .   _   7   punct   _   _

It does not take into account the contents except the columns number
2, 4, 6
(the word itself, POS label and morphological tag), however, in the
default setting the reader
expects the word to be in column 2, the POS label in column 4 and the
detailed tag description
in column 6.

Test data
~~~~~~~~~

When annotating unlabeled text, our model expects the data in
10-column UD format as well. However, it does not pay attention to any column except the first one,
which should be a number, and the second, which must contain a word.
You can also pass only the words with exactly one word on each line
by adding ``"from_words": True`` to ``dataset_reader`` section.
Sentences are separated with blank lines.

You can also pass the unlemmatized text as input. In this case it is preliminarly lemmatized using the
NLTK ``word_tokenize`` function.

Algorithm description
---------------------

We adopt a neural model for morphological tagging from
`Heigold et al., 2017. An extensive empirical evaluation of
character-based morphological tagging for 14
languages <http://www.aclweb.org/anthology/E17-1048>`__.
We refer the reader to the paper for complete description of the
algorithm. The tagger consists
of two parts: a character-level network which creates embeddings for
separate words and word-level
recurrent network which transforms these embeddings to morphological
tags.

The character-level part implements the model from
`Kim et al., 2015. Character-aware language
models <https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12489/12017>`__.
First it embeds the characters into dense vectors, then passes these
vectors through multiple
parallel convolutional layers and concatenates the output of these
convolutions. The convolution
output is propagated through a highway layer to obtain the final word
representation.

You can optionally use a morphological dictionary during tagging. In
this case our model collects
a 0/1 vector with ones corresponding to the dictionary tags of a
current word. This vector is
passed through a one-layer perceptron to obtain an embedding of
dictionary information.
This embedding is concatenated with the output of character-level
network.

As a word-level network we utilize a Bidirectional LSTM, its outputs
are projected through a dense
layer with a softmax activation. In principle, several BiLSTM layers
may be stacked as well
as several convolutional or highway layers on character level;
however, we did not observed
any sufficient gain in performance and use shallow architecture
therefore.

Model configuration.
--------------------

Training configuration
~~~~~~~~~~~~~~~~~~~~~~

We distribute pre-trained models for 11 languages trained on Universal Dependencies data.
Configuration files for reproducible training are also available in
:config:`deeppavlov/configs/morpho_tagger/UD2.0 <morpho_tagger/UD2.0>`, for
example
:config:`deeppavlov/configs/morpho_tagger/UD2.0/morpho_en.json <morpho_tagger/UD2.0/morpho_en.json>`.
The configuration file consists of several parts:

Dataset Reader
^^^^^^^^^^^^^^

The dataset reader describes the instance of
:class:`~deeppavlov.dataset_readers.morphotagging_dataset_reader.MorphotaggerDatasetReader` class.

::

    "dataset_reader": {
        "class_name": "morphotagger_dataset_reader",
        "data_path": "{DOWNLOADS_PATH}/UD2.0_source",
        "language": "en", "data_types": ["train", "dev", "test"]
      }

``class_name`` field refers to the class MorphotaggerDatasetReader,
``data_path`` contains the path to data directory, the ``language``
field is used to derive the name of training and development file.
Alternatively, you can specify these files separately by full (or absolute) paths
like

::

    "dataset_reader": {
        "class_name": "morphotagger_dataset_reader",
        "data_path": ["{DOWNLOADS_PATH}/UD2.0_source/en-ud-train.conllu",
                      "{DOWNLOADS_PATH}/UD2.0_source/en-ud-dev.conllu",
                      "{DOWNLOADS_PATH}/UD2.0_source/en-ud-test.conllu"]
        "data_types": ["train", "dev", "test"]
      }

By default you need only the train file, the dev file is used to
validate
your model during training and the test file is for model evaluation
after training. Since you need some validation data anyway, without
the dev part
you need to resplit your data as described in `Dataset
Iterator <#dataset-iterator>`__ section.

Your data should be in CONLL-U format. It refers to ``predict`` mode also, but in this case only word
column is taken into account. If your data is in single word per line format and you do not want to
reformat it, add ``"from_words": True`` to ``dataset_reader`` section. You can also specify
which columns contain words, tags and detailed tags, for documentation see
:func:`Documentation <deeppavlov.dataset_readers.morphotagging_dataset_reader.read_infile>`.

Dataset iterator
^^^^^^^^^^^^^^^^

:class:`Dataset iterator <deeppavlov.dataset_iterators.morphotagger_iterator.MorphoTaggerDatasetIterator>` class
performs simple batching and shuffling.

::

    "dataset_iterator": {
        "class_name": "morphotagger_dataset"
    }

By default it has no parameters, but if your training and validation
data
are in the same file, you may specify validation split here:

::

    "dataset_iterator": {
        "class_name": "morphotagger_dataset",
        "validation_split": 0.2
    }

Chainer
^^^^^^^

The ``chainer`` part of the configuration file contains the
specification of the neural network model and supplementary things such as vocabularies.
Chainer refers to an instance of :class:`~deeppavlov.core.common.chainer.Chainer`, see
:doc:`configuration </intro/configuration>` for a complete description.

The major part of ``chainer`` is ``pipe``. The ``pipe`` contains
vocabularies and the network itself as well
as some pre- and post- processors. The first part lowercases the input
and normalizes it (see :class:`~deeppavlov.models.preprocessors.capitalization.CapitalizationPreprocessor`).

::

    "pipe": [
          {
            "id": "lowercase_preprocessor",
            "class_name": "lowercase_preprocessor",
            "in": ["x"],
            "out": ["x_processed"]
          },

The second part is the tag vocabulary which transforms tag labels the
model should predict to tag indexes.

::

    {
        "id": "tag_vocab",
        "class_name": "simple_vocab",
        "fit_on": ["y"],
        "special_tokens": ["PAD", "BEGIN", "END"],
        "save_path": "{MODELS_PATH}/morpho_tagger/UD2.0/tag_en.dict",
        "load_path": "{MODELS_PATH}/morpho_tagger/UD2.0/tag_en.dict"
      },

The third part is the character vocabulary used to represent words as sequences of indexes. Only the
symbols which occur at least ``min_freq`` times in the training set are kept.

::

     {
        "id": "char_vocab",
        "class_name": "simple_vocab",
        "min_freq": 3,
        "fit_on": ["x_processed"],
        "special_tokens": ["PAD", "BEGIN", "END"],
        "save_path": "{MODELS_PATH}/morpho_tagger/UD2.0/char_en.dict",
        "load_path": "{MODELS_PATH}/morpho_tagger/UD2.0/char_en.dict"
      },


If you want to utilize external morphological knowledge, you can do it in two ways.
The first is to use :class:`~deeppavlov.models.vectorizers.word_vectorizer.DictionaryVectorizer`.
:class:`~deeppavlov.models.vectorizers.word_vectorizer.DictionaryVectorizer` is instantiated from a dictionary file.
Each line of a dictionary file contains two columns:
a word and a space-separated list of its possible tags. Tags can be in any possible format. The config part for
:class:`~deeppavlov.models.vectorizers.word_vectorizer.DictionaryVectorizer` looks as

::

    {
        "id": "dictionary_vectorizer",
        "class_name": "dictionary_vectorizer",
        "load_path": PATH_TO_YOUR_DICTIONARY_FILE,
        "save_path": PATH_TO_YOUR_DICTIONARY_FILE,
        "in": ["x"],
        "out": ["x_possible_tags"]
    }


The second variant for external morphological dictionary, available only for Russian,
is `Pymorphy2 <http://pymorphy2.readthedocs.io>`_. In this case the vectorizer list all Pymorphy2 tags
for a given word and transforms them to UD2.0 format using
`russian-tagsets <https://github.com/kmike/russian-tagsets>`_ library. Possible UD2.0 tags
are listed in a separate distributed with the library. This part of the config look as
(see :config:`config <morpho_tagger/UD2.0/morpho_ru_syntagrus_pymorphy.json>`))

::

      {
        "id": "pymorphy_vectorizer",
        "class_name": "pymorphy_vectorizer",
        "save_path": "{MODELS_PATH}/morpho_tagger/UD2.0/ru_syntagrus/tags_russian.txt",
        "load_path": "{MODELS_PATH}/morpho_tagger/UD2.0/ru_syntagrus/tags_russian.txt",
        "max_pymorphy_variants": 5,
        "in": ["x"],
        "out": ["x_possible_tags"]
      }

The next part performs the tagging itself. Together with general parameters it describes
the input parameters of :class:`~deeppavlov.models.morpho_tagger.morpho_tagger.MorphoTagger`) class.

::

    {
        "in": ["x_processed"],
        "in_y": ["y"],
        "out": ["y_predicted"],
        "class_name": "morpho_tagger",
        "main": true,
        "save_path": "{MODELS_PATH}/morpho_tagger/UD2.0/ud_en.hdf5",
        "load_path": "{MODELS_PATH}/morpho_tagger/UD2.0/ud_en.hdf5",
        "tags": "#tag_vocab",
        "symbols": "#char_vocab",
        "verbose": 1,
        "char_embeddings_size": 32, "char_window_size": [1, 2, 3, 4, 5, 6, 7],
        "word_lstm_units": 128, "conv_dropout": 0.0, "char_conv_layers": 1,
        "char_highway_layers": 1, "highway_dropout": 0.0, "word_lstm_layers": 1,
        "char_filter_multiple": 50, "intermediate_dropout": 0.0, "word_dropout": 0.2,
        "lstm_dropout": 0.3, "regularizer": 0.01, "lm_dropout": 0.3
    }


When an additional vectorizer is used, the first line is changed to
``"in": ["x_processed", "x_possible_tags"]`` and an additional parameter
``"word_vectorizers": [["#pymorphy_vectorizer.dim", 128]]`` is appended.

Config includes general parameters of :class:`~deeppavlov.core.models.component.Component` class,
described in the :doc:`configuration </intro/configuration>` and specific
:class:`~deeppavlov.models.morpho_tagger.morpho_tagger.MorphoTagger`
parameters. The latter include

- ``tags`` - tag vocabulary. ``#tag_vocab`` refers to an already defined model with ``"id" = "tag_vocab"``.
- ``symbols`` - character vocabulary. ``#char_vocab`` refers to an already defined model with ``"id" = "char_vocab"``.

and other specific parameters of the network, available in :class:`~deeppavlov.models.morpho_tagger.morpho_tagger.MorphoTagger` documentation.

The ``"train"`` section of ``"chainer"`` contains training parameters, such as number of epochs,
batch_size and logging frequency, see general readme for more details.

**chainer** also includes the ``"prettifier"`` subsection, which describes the parameters
of :class:`~deeppavlov.core.models.morpho_tagger.common.TagOutputPrettifier`
which transforms the predictions of the tagger to a readable form.

::

    {
    "in": ["x", "y_predicted"],
    "out": ["y_prettified"],
    "class_name": "tag_output_prettifier",
    "end": "\\n"
    }


It takes two inputs — source sequence of words and predicted sequence of tags
and produces the output of the format

::

    1 Это PRON Animacy=Inan|Case=Acc|Gender=Neut|Number=Sing
    2 чутко ADV Degree=Pos
    3 фиксируют VERB
    Aspect=Imp|Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act
    4 энциклопедические ADJ Case=Nom|Degree=Pos|Number=Plur
    5 издания NOUN Animacy=Inan|Case=Nom|Gender=Neut|Number=Plur
    6 . PUNCT _

    1 Four NUM NumType=Card
    2 months NOUN Number=Plur
    3 later ADV *
    4 , PUNCT *
    5 we PRON Case=Nom|Number=Plur|Person=1|PronType=Prs
    6 were AUX Mood=Ind|Tense=Past|VerbForm=Fin
    7 married VERB Tense=Past|VerbForm=Part|Voice=Pass
    8 . PUNCT _

To generate output in 10 column CONLL-U format add ``"format_mode": "ud"`` to the described section.
