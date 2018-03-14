[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Neural Named Entity Recognition and Slot Filling

This component solves Named Entity Recognition (NER) and Slot-Filling task with different neural network architectures.
 To read about NER without slot filling please address [**README_NER.md**](README_NER.md). 
 This component serves for solving DSTC 2 Slot-Filling task.
In most of the cases, NER task can be formulated as:

_Given a sequence of tokens (words, and maybe punctuation symbols) provide a tag from a predefined set of tags for each token in the sequence._

For NER task there are some common types of entities used as tags:
- persons
- locations
- organizations
- expressions of time
- quantities
- monetary values

In this component

Furthermore, to distinguish adjacent entities with the same tag many applications use BIO tagging scheme. Here "B" denotes beginning of an entity, "I" stands for "inside" and is used for all words comprising the entity except the first one, and "O" means the absence of entity. Example with dropped punctuation:

    Restaraunt  O
    in          O
    the         O
    west        B-LOC
    of          O
    the         O
    city        O
    serving     O
    modern      B-FOOD
    european    I-FOOD
    cuisine     O

In the example above, `FOOD` means food tag, `LOC` means location tag, and "B-" and "I-" are prefixes identifying beginnings and continuations of the entities.

Slot Filling can be formulated as:

_Given an entity of a certain type and a set of all possible values of this entity type provide a normalized form of the entity._

In this component, the Slot Filling task is solved by Levenshtein Distance search across all known entities of a given type. Example:

There is an entity of "food" type:

_chainese_

It is definitely misspelled. The set of all known food entities is {'chinese', 'russian', 'european'}. The nearest known entity from the given set is _chinese_. So the output of the Slot Filling system should be _chinese_.


## Configuration of the model

Configuration of the model can be performed in code or in JSON configuration file. To train 
the model you need to specify four groups of parameters:

- **`dataset_reader`**
- **`dataset`**
- **`chainer`**
- **`train`**

In the subsequent text we show the parameter specification in config file. However, the same notation can be used to specify parameters in code by replacing the JSON with python dictionary.

### Dataset Reader

The dataset reader is a class which reads and parses the data. It returns a dictionary with 
three fields: "train", "test", and "valid". The basic dataset reader is "ner_dataset_reader." 
The dataset reader config part with "ner_dataset_reader" should look like:

```json
"dataset_reader": {
    "name": "dstc2_datasetreader",
    "data_path": "dstc2"
} 
```

where "name" refers to the basic ner dataset reader class and data_path is the path to the 
folder with DSTC 2 dataset.

### Dataset

For simple batching and shuffling you can use "dstc2_ner_dataset". The part of the 
configuration file for the dataset looks like:
 ```json
"dataset": {
    "name": "dstc2_ner_dataset"
}
```

There are no additional parameters in this part.

### Chainer

The chainer part of the configuration file contains the specification of the neural network 
model and supplementary things such as vocabularies.  The chainer part must have the following form:

```json
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
for inference and train mode. "in_y" is used for training and usually contains ground truth answers. "out" field stands for model prediction. The model inside the pipe must have output variable with name "y_predicted" so that "out" knows where to get
predictions.

The major part of "chainer" is "pipe". The "pipe" contains the model and vocabularies. Firstly 
we define vocabularies needed to build the neural network:

```json
"pipe": [
    {
        "id": "word_vocab",
        "name": "default_vocab",
        "fit_on": ["x"],
        "level": "token",
        "save_path": "ner_conll2003_model/word.dict",
        "load_path": "ner_conll2003_model/word.dict"
    },
    {
        "id": "tag_vocab",
        "name": "default_vocab",
        "fit_on": ["y"],
        "level": "token",
        "save_path": "ner_conll2003_model/tag.dict",
        "load_path": "ner_conll2003_model/tag.dict"
    },
    {
        "id": "char_vocab",
        "name": "default_vocab",
        "fit_on": ["x"],
        "level": "char",
        "save_path": "ner_conll2003_model/char.dict",
        "load_path": "ner_conll2003_model/char.dict"
    },
    ...
]
```
Parameters for vocabulary are:

- **`id`** - the name of the vocabulary which will be used in other models
- **`name`** - always equal to `"default_vocab"`
- **`fit_on`** - on which data part of the data the vocabulary should be fitted (built), 
possible options are ["x"] or ["y"]
- **`level`** - char-level or token-level tokenization
- **`save_path`** - path to a new file to save the vocabulary
- **`load_path`** - path to an existing vocabulary

Vocabularies are used for holding sets of tokens, tags, or characters. They assign indices to 
elements of given sets an allow conversion from tokens to indices and vice versa. Conversion of such kind 
is needed to perform lookup in embeddings matrices and compute cross-entropy between predicted 
probabilities and target values. For each vocabulary "default_vocab" model is used. "fit_on" 
parameter defines on which part of the data the vocabulary is built. [\"x\"] stands for the x part of the data (tokens) and [\"y\"] stands for the y part (tags). We can also assemble character-level vocabularies by changing the value of "level" parameter: "char" instead of "token". 

The network is defined by the following part of JSON config:
```json
"pipe": [
    ...
    {
        "in": ["x"],
        "in_y": ["y"],
        "out": ["y_predicted"],
        "main": true,
        "name": "dstc_slotfilling",
        "learning_rate": 1e-3,
        "save_path": "ner/dstc_ner_model",
        "load_path": "ner/dstc_ner_model",
        "word_vocab": "#word_vocab",
        "tag_vocab": "#tag_vocab",
        "char_vocab": "#char_vocab",
        "filter_width": 7,
        "embeddings_dropout": true,
        "n_filters": [
          64,
          64
        ],
        "token_embeddings_dim": 64,
        "char_embeddings_dim": 32,
        "use_batch_norm": true,
        "use_crf": true
    }
]
```

All network parameters are:
- **`in`** - the input to be taken from shared memory. Treated as x. It is used both 
during the training and inference
- **`in_y`** - the target or y input to be taken from shared memory. This input is used during
 the training.
- **`name`** - the name of the model to be used. In this case we use 'ner' model originally 
imported from deeppavlov.models.ner.ner. We use only 'ner' name relying on the @registry 
decorator.
- **`main`** - (reserved for future use) a boolean parameter defining whether this is the main model. 
- **`save_path`** - path to the new file where the model will be saved
- **`load_path`** - path to a pretrained model from where it will be loaded
- **`token_embeddings_dim`** - token embeddings dimensionality (must agree with embeddings 
if they are provided), typical values are from 100 to 300
- **`word_vocab`** - in this field a link to word vocabulary from the pipe should be provided. To address
the vocabulary we use "#word_vocab" expression, where _word_vocab_ is the name of other vocabulary
defined in the pipe before
- **`net_type`** - type of the network, either 'cnn' or 'rnn'
- **`tag_vocab`** - in this field a link to tag vocabulary from the pipe should be provided. In this case
"#tag_vocab" reference is used, addressing previously defined tag vocabulary
- **`char_vocab`** - in this field a link to char vocabulary from the pipe should be provided. In this case
"#char_vocab" reference is used, addressing previously defined char vocabulary
- **`filter_width`** - the width of the convolutional kernel for Convolutional Neural Networks
- **`embeddings_dropout`** - boolean, whether to use dropout on embeddings or not
- **`n_filters`** - a list of output feature dimensionality for each layer. A value `[100, 200]`
means that there will be two layers with 100 and 200 units, respectively.
- **`token_embeddings_dim`** - dimensionality of token embeddings. If embeddings are trained on
the go, this parameter determines dimensionality of the embedding matrix. If the pre-trained 
embeddings this argument must agree with the dimensionality of pre-trained embeddings
- **`char_embeddings_dim`** - character embeddings dimensionality, typical values are 25 - 100
- **`use_crf`** - boolean, whether to use Conditional Random Fields on top of the network (recommended)
- **`use_batch_norm`** - boolean, whether to use Batch Normalization or not. Affects only CNN networks
- **`use_capitalization`** - boolean, whether to include capitalization binary features to the input 
of the network. If True, a binary feature indicating whether the word starts with a capital 
letter will be concatenated to the word embeddings.
- **`dropout_rate`** - probability of dropping the hidden state, values from 0 to 1. 0.5 
works well in most of the cases
- **`learning_rate`**: learning rate to use during the training (usually from 0.01 to 0.0001)

After the "chainer" part you should specify the "train" part:

```json
"train": {
    "epochs": 100,
    "batch_size": 64,

    "metrics": ["ner_f1"],
    "validation_patience": 5,
    "val_every_n_epochs": 5,

    "log_every_n_epochs": 1,
    "show_examples": false
}
```
 
training parameters are:
- **`epochs`** - number of epochs (usually 10 - 100)
- **`batch_size`** - number of samples in a batch (usually 4 - 64)
- **`metrics`** - metrics to validate the model. For NER task we recommend using ["ner_f1"]
- **`validation_patience`** - parameter of early stopping: for how many epochs the training can continue without improvement of metric value on the validation set
- **`val_every_n_epochs`** - how often the metrics should be computed on the validation set
- **`log_every_n_epochs`** - how often the results should be logged
- **`show_examples`** - whether to show results of the network predictions

And now all parts together:
```json
{
  "dataset_reader": {
    "name": "dstc2_datasetreader",
    "data_path": "dstc2"
  },
  "dataset": {
    "name": "dstc2_ner_dataset",
    "dataset_path": "dstc2"
  },
  "chainer": {
    "in": ["x"],
    "in_y": ["y"],
    "pipe": [
      {
        "id": "word_vocab",
        "name": "default_vocab",
        "fit_on": ["x"],
		"level": "token",
        "save_path": "vocabs/word.dict",
        "load_path": "vocabs/word.dict"
      },
      {
        "id": "tag_vocab",
        "name": "default_vocab",
        "fit_on": ["y"],
		"level": "token",
        "save_path": "vocabs/tag.dict",
        "load_path": "vocabs/tag.dict"
      },
      {
        "id": "char_vocab",
        "name": "default_vocab",
        "fit_on": ["x"],
		"level": "char",
        "save_path": "vocabs/char.dict",
        "load_path": "vocabs/char.dict"
      },
      {
        "in": ["x"],
        "in_y": ["y"],
        "out": ["y_predicted"],
        "main": true,
        "name": "dstc_slotfilling",
        "learning_rate": 1e-3,
        "save_path": "ner/dstc_ner_model",
        "load_path": "ner/dstc_ner_model",
        "word_vocab": "#word_vocab",
        "tag_vocab": "#tag_vocab",
        "char_vocab": "#char_vocab",
        "verbouse": true,
        "filter_width": 7,
        "embeddings_dropout": true,
        "n_filters": [
          64,
          64
        ],
        "token_embeddings_dim": 64,
        "char_embeddings_dim": 32,
        "use_batch_norm": true,
        "use_crf": true
      }
    ],
    "out": ["y_predicted"]
  },
  "train": {
    "epochs": 100,
    "batch_size": 64,

    "metrics": ["slots_accuracy"],
    "validation_patience": 5,
    "val_every_n_epochs": 5,

    "log_every_n_epochs": 1,
    "show_examples": false
  }
}
```

## Train and use the model

Please see an example of training a NER model and using it for prediction:

```python
from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.infer import interact_model
PIPELINE_CONFIG_PATH = 'configs/ner/slotfill_dstc2.json'
train_model_from_config(PIPELINE_CONFIG_PATH)
interact_model(PIPELINE_CONFIG_PATH)
```

This example assumes that the working directory is deeppavlov.
