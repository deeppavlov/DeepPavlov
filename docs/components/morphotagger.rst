Neural Morphological Tagging
============================

It is an implementation of neural morphological tagger from
`Heigold et al., 2017. An extensive empirical evaluation of
character-based morphological tagging for 14
languages <http://www.aclweb.org/anthology/E17-1048>`__.
We distribute the models for 11 languages trained on `Universal
Dependencies corpora <www.universaldependencies.org>`__.
Our model achieves the state-of-the-art performance among open source
systems.

+----------------+--------------+-----------------+------------------+
|    Language    | Code         | UDPipe accuracy | Our top accuracy |
+----------------+--------------+-----------------+------------------+
| Arabic         | ar           | 88.31           | 90.85            |
+----------------+--------------+-----------------+------------------+
| Czech          | cs           | 91.86           | 94.35            |
+----------------+--------------+-----------------+------------------+
| English        | en           | 92.53           | 93.00            |
+----------------+--------------+-----------------+------------------+
| French         | fr           | 95.25           | 95.45            |
+----------------+--------------+-----------------+------------------+
| German         | de           | 76.65           | 83.83            |
+----------------+--------------+-----------------+------------------+
| Hindi          | ar           | 87.74           | 90.01            |
+----------------+--------------+-----------------+------------------+
| Hungarian      | ar           | 69.52           | 75.34            |
+----------------+--------------+-----------------+------------------+
| Italian        | it           | 96.33           | 96.47            |
+----------------+--------------+-----------------+------------------+
| Russian        | ru_syntagrus | 93.57           | 96.23            |
+----------------+--------------+-----------------+------------------+
| Spanish        | es_ancora    | 96.88           | 97.00            |
+----------------+--------------+-----------------+------------------+
| Turkish        | tr           | 86.98           | 88.03            |
+----------------+--------------+-----------------+------------------+

If you want to use our models from scratch, do the following
(all the examples are for ru\_syntagrus corpus, change the filenames accordingly to invoke models for other languages):

#. Download data

   ::

       python -m deeppavlov download morpho_ru_syntagrus_train

   To perform all downloads in runtime you can also run all subsequent
   commands with ``-d`` key,
#. To apply a pre-trained ru\_syntagrus model to ru\_syntagrus test
   data, run

   ::

       python -m deeppavlov.models.morpho_tagger morpho_ru_syntagrus_predict

   to use a basic model, or

   ::

       python -m deeppavlov.models.morpho_tagger morpho_ru_syntagrus_predict_pymorphy

   to apply a model which additionally utilizes information from
   `Pymorphy2 <http://pymorphy2.readthedocs.io>`__ library.

A subdirectory ``results`` will be created in your current working
directory
and predictions will be written to the file
``ud_ru_syntagrus_test.res`` in it.

#. To evaluate ru\_syntagrus model on ru\_syntagrus test subset, run

   ::

       python -m deeppavlov evaluate morpho_ru_syntagrus_train

#. To retrain model on ru\_syntagrus dataset, run one of the following
   (the first is for Pymorphy-enriched model)

   ::

       python -m deeppavlov train morpho_ru_syntagrus_train_pymorphy
       python -m deeppavlov train morpho_ru_syntagrus_train

   Be careful, one epoch takes 8-60 minutes depending on your GPU.
#. To tag Russian sentences from stdin, run

   ::

       python -m deeppavlov interact morpho_ru_syntagrus_predict_pymorphy

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
one-word-per-line format
with sentences separated by blank line.

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
``deeppavlov/configs/morpho_tagger/UD2.0``, for
example
``deeppavlov/configs/morpho_tagger/UD2.0/morpho_en.json``.
The configuration file consists of several parts:

Dataset Reader
^^^^^^^^^^^^^^

The dataset reader describes the instance of
:class:`~deeppavlov.dataset_readers.morphotagging_dataset_reader.MorphotaggerDatasetReader` class.

::

    "dataset_reader": {
        "name": "morphotagger_dataset_reader",
        "data_path": "UD2.0_source",
        "language": "en", "data_types": ["train", "dev", "test"]
      }

``name`` field refers to the class MorphotaggerDatasetReader,
``data_path`` contains the path to data directory, the ``language``
field is used to derive the name of training and development file.
Alternatively, you can specify these files separately by full (or absolute) paths
like

::

    "dataset_reader": {
        "name": "morphotagger_dataset_reader",
        "data_path": ["UD2.0_source/en-ud-train.conllu",
                      "UD2.0_source/en-ud-dev.conllu",
                      "UD2.0_source/en-ud-test.conllu"]
        "data_types": ["train", "dev", "test"]
      }

By default you need only the train file, the dev file is used to
validate
your model during training and the test file is for model evaluation
after training. Since you need some validation data anyway, without
the dev part
you need to resplit your data as described in `Dataset
Iterator <#dataset-iterator>`__ section.

Your data should be in CONLL-U format. It refers to `predict` mode also, but in this case only word
column is taken into account. If your data is in single word per line format and you do not want to
reformat it, add `"from_words": True` to ``dataset_reader`` section. You can also specify
which columns contain words, tags and detailed tags, for documentation see
:func:`Documentation <deeppavlov.dataset_readers.morphotagging_dataset_reader.read_infile>`.

Dataset iterator
^^^^^^^^^^^^^^^^

:class:`Dataset iterator <deeppavlov.dataset_iterators.morphotagger_iterator.MorphoTaggerDatasetIterator>` class
performs simple batching and shuffling.

::

    "dataset_iterator": {
        "name": "morphotagger_dataset"
    }

By default it has no parameters, but if your training and validation
data
are in the same file, you may specify validation split here:

::

    "dataset_iterator": {
        "name": "morphotagger_dataset",
        "validation_split": 0.2
    }

Chainer
^^^^^^^

The ``chainer`` part of the configuration file contains the
specification of the neural network model and supplementary things such as vocabularies.
Chainer refers to an instance of :class:`~deeppavlov.core.common.chainer.Chainer`, see
<intro/config_description> for a complete description.

The major part of ``chainer`` is ``pipe``. The ``pipe`` contains
vocabularies and the network itself as well
as some pre- and post- processors. The first part lowercases the input
and normalizes it (see :class:`~deeppavlov.models.preprocessors.capitalization.CapitalizationPreprocessor`).

::

    "pipe": [
          {
            "id": "lowercase_preprocessor",
            "name": "lowercase_preprocessor",
            "in": ["x"],
            "out": ["x_processed"]
          },

The second part is the tag vocabulary which transforms tag labels the
model should predict to tag indexes.

::

    {
        "id": "tag_vocab",
        "name": "default_vocab",
        "fit_on": ["y"],
        "level": "token",
        "special_tokens": ["PAD", "BEGIN", "END"],
        "save_path": "morpho_tagger/UD2.0/tag_en.dict",
        "load_path": "morpho_tagger/UD2.0/tag_en.dict"
      },

 The third part is the character vocabulary used to represent words as sequences of indexes. Only the
 symbols which occur at least ``min_freq`` times in the training set are kept.

::

     {
        "id": "char_vocab",
        "name": "default_vocab",
        "min_freq": 3,
        "fit_on": ["x_processed"],
        "special_tokens": ["PAD", "BEGIN", "END"],
        "level": "char",
        "save_path": "morpho_tagger/UD2.0/char_en.dict",
        "load_path": "morpho_tagger/UD2.0/char_en.dict"
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
        "name": "dictionary_vectorizer",
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
(see :config:`~deeppavlov/configs/morpho_tagger/UD2.0/morpho_ru_syntagrus_pymorphy.json`))

::

      {
        "id": "pymorphy_vectorizer",
        "name": "pymorphy_vectorizer",
        "save_path": "morpho_tagger/UD2.0/ru_syntagrus/tags_russian.txt",
        "load_path": "morpho_tagger/UD2.0/ru_syntagrus/tags_russian.txt",
        "max_pymorphy_variants": 5,
        "in": ["x"],
        "out": ["x_possible_tags"]
      }

The next part performs the tagging itself. Together with general parameters it describes
the input parameters of :class:`~deeppavlov.models.morpho_tagger.network.CharacterTagger`) class.

::

    {
        "in": ["x_processed"],
        "in_y": ["y"],
        "out": ["y_predicted"],
        "name": "morpho_tagger",
        "main": true,
        "save_path": "morpho_tagger/UD2.0/ud_en.hdf5",
        "load_path": "morpho_tagger/UD2.0/ud_en.hdf5",
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
`"in": ["x_processed", "x_possible_tags"]` and an additional parameter
`"word_vectorizers": [["#pymorphy_vectorizer.dim", 128]]` is appended.

Config includes general parameters of :class:`~deeppavlov.core.models.component.Component` class,
described in <intro/config_description> and specific `~deeppavlov.models.morpho_tagger.network.CharacterTagger`
parameters. The latter include

- `tags` - tag vocabulary. `#tag_vocab` refers to an already defined model with "id" = "tag_vocab".
- `symbols` - character vocabulary. `#char_vocab` refers to an already defined model with "id" = "char_vocab".

and other specific parameters of the network, available in :class:`~deeppavlov.models.morpho_tagger.network.CharacterTagger` documentation.

The `"train"` section of `"chainer"` contains training parameters, such as number of epochs,
batch_size and logging frequency, see general readme for more details.

Evaluate configuration
~~~~~~~~~~~~~~~~~~~~~~

Evaluate configuration file is almost the same as the train one, the only difference is
that **dataset_reader** reads only test part of data. Also there are no logging parameters
in the ``''train''`` subsection of **chainer**. Now it looks like

::

    "train": {
    "test\_best": true,
    "batch\_size": 16,
    "metrics": ["per\_token\_accuracy"]
    }


Predict configuration
~~~~~~~~~~~~~~~~~~~~~

In prediction configuration **chainer** includes an additional subsection for the prettifier,
which transforms the predictions of the tagger to a readable form.

::

    {
    "in": ["x", "y\_predicted"],
    "out": ["y\_prettified"],
    "name": "tag\_output\_prettifier",
    "end": "\\n"
    }


It takes two inputs -- source sequence of words and predicted sequence of tags
and produces the output of the format

::

    1 Это PRON Animacy=Inan\|Case=Acc\|Gender=Neut\|Number=Sing
    2 чутко ADV Degree=Pos
    3 фиксируют VERB
    Aspect=Imp\|Mood=Ind\|Number=Plur\|Person=3\|Tense=Pres\|VerbForm=Fin\|Voice=Act
    4 энциклопедические ADJ Case=Nom\|Degree=Pos\|Number=Plur
    5 издания NOUN Animacy=Inan\|Case=Nom\|Gender=Neut\|Number=Plur
    6 . PUNCT \_

    1 Four NUM NumType=Card
    2 months NOUN Number=Plur
    3 later ADV *
    4 , PUNCT *
    5 we PRON Case=Nom\|Number=Plur\|Person=1\|PronType=Prs
    6 were AUX Mood=Ind\|Tense=Past\|VerbForm=Fin
    7 married VERB Tense=Past\|VerbForm=Part\|Voice=Pass
    8 . PUNCT \_

You can also generate output in 10 column CONLL-U format.
For this purpose add ``format_mode`` = ``ud`` to the **prettifier** section.

The **train** section of the config is replaced by the **predict** section:

::

    "predict":
    {
    "batch\_size": 32,
    "outfile": "results/ud\_ru\_syntagrus\_test.res"
    }
