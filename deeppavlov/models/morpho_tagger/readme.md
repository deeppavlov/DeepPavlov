# Morphological tagging

## Task description

Morphological tagging consists in assigning labels, describing word morphology, to a pre-tokenized sequence of words.
In the most simple case these labels are just part-of-speech (POS) tags, hence in earlier times of NLP the task was
often referred as POS-tagging. The refined version of the problem which we solve here performs more fine-grained 
classification, also detecting the values of other morphological features, such as case, gender and number for nouns,
mood, tense, etc. for verbs and so on. Morphological tagging is a stage of common NLP pipeline, it generates useful
features for further tasks such as syntactic parsing, named entity recognition or machine translation.

Common output for morphological tagging looks as below. The examples are for Russian and English language and use the
inventory of tags and features from [Universal Dependencies project](www.universaldependencies.org/guidelines.html).

```
1	Это	PRON	Animacy=Inan|Case=Acc|Gender=Neut|Number=Sing
2	чутко	ADV	Degree=Pos
3	фиксируют	VERB	Aspect=Imp|Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act
4	энциклопедические	ADJ	Case=Nom|Degree=Pos|Number=Plur
5	издания	NOUN	Animacy=Inan|Case=Nom|Gender=Neut|Number=Plur
6	.	PUNCT	_

1	Four	NUM	NumType=Card
2	months	NOUN	Number=Plur
3	later	ADV	_
4	,	PUNCT	_
5	we	PRON	Case=Nom|Number=Plur|Person=1|PronType=Prs
6	were	AUX	Mood=Ind|Tense=Past|VerbForm=Fin
7	married	VERB	Tense=Past|VerbForm=Part|Voice=Pass
8	.	PUNCT	_

```

The full UD format (see below) includes more columns including lemma and syntactic information.

### Training data

Our tagger accepts the data in [CONLL-U format](http://universaldependencies.org/format.html):

```
1	Four	four	NUM	CD	NumType=Card	2	nummod	_	_
2	months	month	NOUN	NNS	Number=Plur	3	obl:npmod	_	_
3	later	later	ADV	RB	_	7	advmod	_	SpaceAfter=No
4	,	,	PUNCT	,	_	7	punct	_	_
5	we	we	PRON	PRP	Case=Nom|Number=Plur|Person=1|PronType=Prs	7	nsubj:pass	_	_
6	were	be	AUX	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	7	aux:pass	_	_
7	married	marry	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	0	root	_	SpaceAfter=No
8	.	.	PUNCT	.	_	7	punct	_	_
```

It does not take into account the contents except the columns number 2, 4, 6 
(the word itself, POS label and morphological tag), however, in the default setting the reader
expects the word to be in column 2, the POS label in column 4 and the detailed tag description
in column 6.

### Test data

When annotating unlabeled text, our model expects the data in one-word-per-line format 
with sentences separated by blank line.

## Algorithm description

We adopt a neural model for morphological tagging from 
[Heigold et al., 2017. An extensive empirical evaluation of character-based morphological tagging for 14 languages](http://www.aclweb.org/anthology/E17-1048).
We refer the reader to the paper for complete description of the algorithm. The tagger consists
of two parts: a character-level network which creates embeddings for separate words and word-level
recurrent network which transforms these embeddings to morphological tags.

The character-level part implements the model from 
[Kim et al., 2015. Character-aware language models](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12489/12017).
First it embeds the characters into dense vectors, then passes these vectors through multiple
parallel convolutional layers and concatenates the output of these convolutions. The convolution
output is propagated through a highway layer to obtain the final word representation.

As a word-level network we utilize a Bidirectional LSTM, its outputs are projected through a dense
layer with a softmax activation. In principle, several BiLSTM layers may be stacked as well
as several convolutional or highway layers on character level; however, we did not observed
any sufficient gain in performance and use shallow architecture therefore.

## Model configuration.

### Training configuration

We distribute pre-trained models for Russian (ru_syntagrus corpus) and Hungarian language.
Configuration files for reproducible training are also available in a 
[separate directory](../../configs/morpho_tagger/UD2.0), for example 
[train_config.json](../../configs/morpho_tagger/UD2.0/hu/train_config.json). 
The configuration file consists of several parts:

#### Dataset Reader

The dataset reader describes the instance of 
[MorphotaggerDatasetReader class](../../dataset_readers/morphotagging_dataset_reader.py#L70).

```
"dataset_reader": {
    "name": "morphotagger_dataset_reader",
    "data_path": "UD2.0_source",
    "language": "hu", "data_types": ["train", "dev", "test"]
  }
```

"name" field refers to the class MorphotaggerDatasetReader, "data_path" contains the path to data directory, the "language"
field is used to derive the name of training and development file.
Alternatively, you can specify these files separately by full paths
like

```
"dataset_reader": {
    "name": "morphotagger_dataset_reader",
    "data_path": ["UD2.0_source/hu-ud-train.conllu",
                  "UD2.0_source/hu-ud-dev.conllu",
                  "UD2.0_source/hu-ud-test.conllu"]
    "data_types": ["train", "dev", "test"]
  }
```

By default you need only the train file, the dev file is used to validate
your model during training and the test file is for model evaluation
after training. Since you need some validation data anyway, without the dev part
you need to resplit your data as described in [Dataset Iterator](#dataset-iterator) section.

#### Dataset iterator

[Dataset iterator class](../../dataset_iterators/morphotagger_iterator.py#L59) performs simple batching and shuffling.

```
"dataset_iterator": {
    "name": "morphotagger_dataset"
}
```

By default it has no parameters, but if your training and validation data
are in the same file, you may specify validation split here:

```
"dataset_iterator": {
    "name": "morphotagger_dataset",
    "validation_split": 0.2
}
```
