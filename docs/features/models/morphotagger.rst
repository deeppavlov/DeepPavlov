Neural Morphological Tagging
============================

It is an implementation of neural morphological tagger.
The model includes only a dense layer on the top of BERT embedder.
See the `BERT paper <http://arxiv.org/abs/1810.04805>`__
for a more complete description, as well as the
`BERT section <features/models/bert>`__ of the documentation.

The model is trained on `Universal Dependencies corpora <https://universaldependencies.org/>`__
(version 2.3).

+----------------+--------------+-----------------+-------------------------------+------------------+----------------+
|    Language    | Code         | UDPipe accuracy | UDPipe Future accuracy        | Our top accuracy | Model size (MB)|
+================+==============+=================+===============================+==================+================+
| Russian (UD2.3)| ru_syntagrus | 93.5            | 96.90                         | 97.83            |  661           |
+----------------+--------------+-----------------+-------------------------------+------------------+----------------+

===========================
Usage examples.
===========================

Before using the model make sure that all required packages are installed using the command:

.. code:: bash

    python -m deeppavlov install morpho_ru_syntagrus_bert

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
       model = build_model('morpho_ru_syntagrus_bert', download=True)
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


You may also pass the tokenized sentences instead of raw ones:

.. code:: python

    sentences = [["Я", "шёл", "домой", "по", "незнакомой", "улице", "."]]
    for parse in model(sentences):
        print(parse)

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
