[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Neural Named Entity Recognition and Slot Filling

This component solves Named Entity Recognition (NER) and Slot-Filling task with different neural network architectures.
 To read about NER without slot filling please address [**README_NER.md**](README_NER.md). 
 This component serves for solving DSTC 2 Slot-Filling task.
In most of the cases, NER task can be formulated as:

_Given a sequence of tokens (words, and may be punctuation symbols) provide a tag from predefined set of tags for each token in the sequence._

For NER task there are some common types of entities which essentially are tags:
- persons
- locations
- organizations
- expressions of time
- quantities
- monetary values

In this component

Furthermore, to distinguish consequent entities with the same tags BIO tagging scheme is used. "B" stands for beginning,
"I" stands for the continuation of an entity and "O" means the absence of entity. Example with dropped punctuation:

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

In the example above, `FOOD` means food tag (and `LOC` means location tag), and "B-" and "I-" are prefixes identifying beginnings and continuations of the entities.

Slot Filling can be formulated as:

_Given some entity of certain type and a set of all possible values of this entity type provide a normalized form of the entity._

In this component, the Slot Filling task is solved by Levenshtein Distance search across all known entities of given type. Example:

There is an entity of "food" type:

_chainese_

Definitely, it is misspelled. The set of all known food entities is {'chinese', 'russian', 'european'}. The nearest known entity from the given set is _chinese_. So the output of the Slot Filling system should be _chinese_.


## Configuration of the model

Configuration of the model can be performed in code or in JSON configuration file. To train 
the model four groups of parameters must be specified:

- **`dataset_reader`**
- **`dataset`**
- **`chainer`**
- **`train`**

The following parts assume that config file is used. However, it can be used in the code
replacing the JSON with python dictionary.

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

In order to perform simple batching and shuffling "dstc2_ner_dataset" is used. The part of the 
configuration file for the dataset looks like:
 ```json
"dataset": {
    "name": "dstc2_ner_dataset"
}
```

There is no additional parameters in this part.

### Chainer

The chainer part of the configuration file contains the specification of the neural network 
model and supplementary things such as vocabularies. The chainer part must have the following 
form:

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
The inputs and outputs must be specified in the pype. "in" means regular input that is used 
for inference and train mode. "in_y" is used for training and regularly 
contains ground truth answers. "out" field stands for model prediction. The model inside the 
pipe must have output variable with name "y_predicted" so that "out" knows where to get 
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
possible options: ["x"] or ["y"]
- **`level`** - char or token level tokenization
- **`save_path`** - path to the vocabulary where it will be saved
- **`load_path`** - path to load existing vocabulary

Vocabularies are used for holding sets of tokens, tags, or characters. They assign indexes to 
elements of given sets an allow conversion from tokens to indices and vice versa. Conversion of such kind 
is needed to perform lookup in embeddings matrices and compute cross-entropy between predicted 
probabilities and target values. For each vocabulary "default_vocab" model is used. "fit_on" 
parameter defines on which part of the data build (assemble or train) the vocabulary. [\"x\"] 
stends for x part of the data (tokens) and [\"y\"] stands for y part (tags). We also can 
assemble character level vocabularies by defining level": "char" instead of "token".

The following part is network part. The network is defined by the following part of JSON 
config:
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
- **`in`** - the input to be taken from shared memory. Treated as x. So it is used both 
during the training and inference
- **`in_y`** - the target or y input to be taken from shared memory. This input is used during
 the training.
- **`name`** - the name of the model to be used. In this case we use 'ner' model originally 
imported from deeppavlov.models.ner.ner. We use only 'ner' name relying on the @registry 
decorator.
- **`main`** - a boolean parameter defining whether this is the main model. Only the main 
model is trained during the training phase.
- **`save_path`** - path to the model where it will be saved
- **`load_path`** - path to load pretrained model
- **`token_embeddings_dim`** - token embeddings dimensionality (must agree with embeddings 
if they are provided), typical values are from 100 to 300
- **`word_vocab`** - in this field a link to word vocabulary from pipe must be provided. To address
the vocabulary we use "#word_vocab" expression, where _word_vocab_ is the name of other vocabulary
defined in the pipe before
- **`net_type`** - type of the network, either 'cnn' or 'rnn'
- **`tag_vocab`** - in this field a link to tag vocabulary from pipe must be provided. In this case
"#tag_vocab" reference is used, addressing previously defined tag vocabulary
- **`char_vocab`** - in this field a link to char vocabulary from pipe must be provided. In this case
"#char_vocab" reference is used, addressing previously defined char vocabulary
- **`filter_width`** - the width of the convolutional kernel for Convolutional Neural Networks
- **`embeddings_dropout`** - boolean, whether to use dropout on embeddings or not
- **`n_filters`** - list of output feature dimensionality for each layer. For [100, 200] 
there will be two layers with 100 and 200 number of units respectively. 
- **`token_embeddings_dim`** - dimensionality of token embeddings. If embeddings are trained on
the go this parameter determine dimensionality of embedding matrix. If the case of pre-trained 
embeddings this argument must agree with pre-trained dimensionality
- **`char_embeddings_dim`** - character embeddings dimensionality, typical values are 25 - 100
- **`use_crf`** - whether to use Conditional Random Fields on the top (suggested to always use
 True)
- **`use_batch_norm`** - whether to use Batch Normalization or not. Affects only CNN networks
- **`use_capitalization`** - whether to include capitalization binary features to the input 
of the network. If True than binary feature indicating whether the word starts with a capital 
letter will be concatenated to the word embeddings.
- **`dropout_rate`** - probability of dropping the hidden state, values from 0 to 1. 0.5 
works well in most of the cases
- **`learning_rate`**: learning rate to use durint the training (0.01 - 0.0001 typical)

After defining the "chainer" the "train" part must be specified:

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
- **`epochs`** - number of epochs (10 - 100 typical)
- **`batch_size`** - number of samples in the batch (4 - 64 typical)
- **`metrics`** - metrics to validated the model. For NER task ["ner_f1"] should be used
- **`validation_patience`** - how many epochs continue training without improvement of metric
- **`val_every_n_epochs`** - how often calculate metrics on validation set
- **`log_every_n_epochs`** - how often to log the results
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

To train and use model for prediction the following code can be used:

```python
from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.infer import interact_model
PIPELINE_CONFIG_PATH = 'configs/ner/slotfill_dstc2.json'
train_model_from_config(PIPELINE_CONFIG_PATH)
interact_model(PIPELINE_CONFIG_PATH)
```

This example assumes that the working directory is deeppavlov.