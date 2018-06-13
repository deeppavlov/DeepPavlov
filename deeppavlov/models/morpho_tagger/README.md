[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Neural Morphological Tagging

It is an implementation of neural morphological tagger from
[Heigold et al., 2017. An extensive empirical evaluation of character-based morphological tagging for 14 languages](http://www.aclweb.org/anthology/E17-1048).
We distribute the model trained on ru_syntagrus corpus of [Universal Dependencies project](www.universaldependencies.org).
If you want to use it from scratch, do the following:

1. Download data
```
python -m deeppavlov download morpho_ru_syntagrus_train
```
To perform all downloads in runtime you can also run all subsequent commands with `-d` key, 
2. To apply a pre-trained ru_syntagrus model to ru_syntagrus test data, run
```
python -m deeppavlov.models.morpho_tagger morpho_ru_syntagrus_predict
```
A subdirectory ``results`` will be created in your current working directory and predictions will be written to the file ```ud_ru_syntagrus_test.res``` in it.
3. To evaluate ru_syntagrus model on ru_syntagrus test subset, run
```
python -m deeppavlov evaluate morpho_ru_syntagrus_train
```
4. To retrain model on ru_syntagrus dataset, run
```
python -m deeppavlov train morpho_ru_syntagrus_train
```
Be careful, one epoch takes 8-60 minutes depending on your GPU.
5. To tag Russian sentences from stdin, run
```
python -m deeppavlov interact morpho_ru_syntagrus_predict
```

Read the detailed readme below.

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

```

The full UD format (see below) includes more columns including lemma and syntactic information.

### Training data

Our tagger accepts the data in [CONLL-U format](http://universaldependencies.org/format.html):

```
1   Four    four    NUM CD  NumType=Card    2   nummod  _   _
2   months  month   NOUN    NNS Number=Plur 3   obl:npmod   _   _
3   later   later   ADV RB  _   7   advmod  _   SpaceAfter=No
4   ,   ,   PUNCT   ,   _   7   punct   _   _
5   we  we  PRON    PRP Case=Nom|Number=Plur|Person=1|PronType=Prs  7   nsubj:pass  _   _
6   were    be  AUX VBD Mood=Ind|Tense=Past|VerbForm=Fin    7   aux:pass    _   _
7   married marry   VERB    VBN Tense=Past|VerbForm=Part|Voice=Pass 0   root    _   SpaceAfter=No
8   .   .   PUNCT   .   _   7   punct   _   _
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
[morpho_hu_train.json](../../configs/morpho_tagger/UD2.0/hu/morpho_hu_train.json). 
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
#### Chainer

The `''chainer''` part of the configuration file contains the specification of the neural network 
model and supplementary things such as vocabularies. Chainer should be defined as follows:

```
"chainer": {
    "in": ["x"],
    "in_y": ["y"],
    "pipe": [
      ...
    ],
    "out": ["y_predicted"]
  }
```
The inputs and outputs must be specified in the pipe. "in" means regular input that is used 
for inference and train mode. "in_y" is used for training and usually contains ground truth answers. 
"out" field stands for model prediction. The model inside the pipe must have output variable with 
name "y_predicted" so that "out" knows where to get 
predictions.

The major part of "chainer" is "pipe". The "pipe" contains vocabularies and the network itself as well
as some pre- and post- processors. The first part lowercases the input and normalizes it (see 
[../../models/preprocessors/capitalization.py]()).

```
"pipe": [
      {
        "id": "lowercase_preprocessor",
        "name": "lowercase_preprocessor",
        "in": ["x"],
        "out": ["x_processed"]
      },
```

The second part is the tag vocabulary which transforms tag labels the model should predict to tag indexes.

```
    {
        "id": "tag_vocab",
        "name": "default_vocab",
        "fit_on": ["y"],
		"level": "token",
        "special_tokens": ["PAD", "BEGIN", "END"],
        "save_path": "morpho_tagger/UD2.0/tag_hu.dict",
        "load_path": "morpho_tagger/UD2.0/tag_hu.dict"
      },
 ```
 
 The third part is the character vocabulary used to represent words as sequences of indexes. Only the
 symbols which occur at least "min_freq" times in the training set are kept.
 
 ```
     {
        "id": "char_vocab",
        "name": "default_vocab",
        "min_freq": 3,
        "fit_on": ["x_processed"],
        "special_tokens": ["PAD", "BEGIN", "END"],
        "level": "char",
        "save_path": "morpho_tagger/UD2.0/char_hu.dict",
        "load_path": "morpho_tagger/UD2.0/char_hu.dict"
      },
  ```
  
  The next part performs the tagging itself. Together with general parameters it describes 
  the input parameters of [CharacterTagger](../../models/morpho_tagger/network.py#L33) class.
  ```
    {
        "in": ["x_processed"],
        "in_y": ["y"],
        "out": ["y_predicted"],
        "name": "morpho_tagger",
        "main": true,
        "save_path": "morpho_tagger/UD2.0/ud_hu.hdf5",
        "load_path": "morpho_tagger/UD2.0/ud_hu.hdf5",
        "tags": "#tag_vocab",
        "symbols": "#char_vocab",
        "verbose": 1,
        "char_embeddings_size": 32, "char_window_size": [1, 2, 3, 4, 5, 6, 7],
        "word_lstm_units": 128, "conv_dropout": 0.0, "char_conv_layers": 1,
        "char_highway_layers": 1, "highway_dropout": 0.0, "word_lstm_layers": 1,
        "char_filter_multiple": 50, "intermediate_dropout": 0.0, "word_dropout": 0.2,
        "lstm_dropout": 0.3, "regularizer": 0.01, "lm_dropout": 0.3
    }
 ```
 
General parameters are:
- **`in`** - data to be used during training. "x_processed" means
that network obtains the output of the lowercase_preprocessor as its input.
- **`in_y`** - the target to be used as gold labels during training.
- **`out`** - the name of the model output.
- **`name`** - registered name of the class [CharacterTagger](../../models/morpho_tagger/network.py#L33).
- **`main`** - (reserved for future use) a boolean parameter defining whether this is the main model. 
- **`save_path`** - where the model is saved after training.
- **`load_path`** - from where the pretrained model can be loaded if it exists.

Model parameters are:
- **`tags`** - tag vocabulary. `#tag_vocab` refers to an already defined model with "id" = "tag_vocab".
- **`symbols`** - character vocabulary. `#char_vocab` refers to an already defined model with "id" = "char_vocab".
- **`char_embeddings_size`** - the dimensionality of character embeddings (default=16)
- **`char_conv_layers`** - number of convolution layers applied to character embeddings (default=1)
- **`char_window_size`** - width of convolution filters (default=5). It can be a list if several parallel filters 
are applied, for example, [2, 3, 4, 5].
- **`char_filters`** - number of convolution filters (default=**None**). It can be a number, a list (when
there are several windows of different width on a single convolution layer), a list of lists, if there
are more than 1 convolution layers, or **None**. If **None**, a layer with width *width* contains 
min(self.char_filter_multiple * *width*, 200) filters.
- **`char_filter_multiple`** - a coefficient used to calculate number of filters depending on window size. 
- **`char_highway layer`** - number of highway layers on the top of convolutions (default=1).
- **`conv_dropout`** - ratio of dropout between convolutional layers (default=0.0).
- **`highway_dropout`** - ratio of dropout between highway layers (default=0.0).
- **`intermediate_dropout`** - ratio of dropout between last convolutional and first highway layer (default=0.0).
- **`lstm_dropout`** - ratio of dropout inside word-level LSTM (default=0.0).
- **`word_lstm_layers`** - number of word-level LSTM layers (default=1).
- **`word_lstm_units`** - number of units in word-level LSTM (default=128). It can be a list if there
are multiple layers.
- **`word_dropout`** - ratio of dropout before word-level LSTM (default=0.0).
- **`regularizer`** - the weight of l2-regularizer for output probabilities (default=None). None means
that no regularizer is applied.
- **`verbose`** - the level of verbosity during training. If it is positive, prints model summary.

The `"train"` section of `"chainer"` contains training parameters, such as number of epochs,
batch_size and logging frequency, see [general README](../../../README.md) for more details.

### Evaluate configuration

Evaluate configuration file is almost the same as the train one, the only difference is
that **dataset_reader** reads only test part of data. Also there are no logging parameters
in the ``''train''`` subsection of **chainer**. Now it looks like

```
"train": {
    "test_best": true,
    "batch_size": 16,
    "metrics": ["per_token_accuracy"]
  }
```

### Predict configuration

In prediction configuration **chainer** includes an additional subsection for the prettifier,
which transforms the predictions of the tagger to a readable form. 

```
{
    "in": ["x", "y_predicted"],
    "out": ["y_prettified"],
    "name": "tag_output_prettifier",
    "end": "\n"
}
```

It takes two inputs -- source sequence of words and predicted sequence of tags
and produces the output of the format

```
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
```

The **train** section of the config is replaced by the **predict** section:
```
"predict": 
  {
    "batch_size": 32,
    "outfile": "results/ud_ru_syntagrus_test.res"
  }
```



