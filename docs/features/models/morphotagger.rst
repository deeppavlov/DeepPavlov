Neural Morphological Tagging
============================

It is an implementation of neural morphological tagger. The model is based on BERT for token classification
<https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForTokenClassification>`__.
The model is trained on `Universal Dependencies corpora <https://universaldependencies.org/>`__ (version 2.3).

+-----------------+------------------+----------------+
| UDPipe accuracy | Our top accuracy | Model size (Gb)|
+=================+==================+================+
| 93.5            | 97.6             |      2.1       |
+-----------------+------------------+----------------+

===========================
Usage examples.
===========================

Python:
---------------------------

.. code:: python

   from deeppavlov import build_model
   model = build_model("morpho_ru_syntagrus_bert", download=True, install=True)

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
