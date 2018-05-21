[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Neural Named Entity Recognition and Slot Filling

This component solves Named Entity Recognition (NER) and Slot-Filling task with different neural network architectures.
 To read about NER without slot filling please address [**README_NER.md**](../ner/README_NER.md). 
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

```
"dataset_reader": {
    "name": "dstc2_datasetreader",
    "data_path": "dstc2"
} 
```

where "name" refers to the basic ner dataset reader class and data_path is the path to the 
folder with DSTC 2 dataset.

### Dataset Iterator

For simple batching and shuffling you can use "dstc2_ner_iterator". The part of the 
configuration file for the dataset iterator looks like:
 ```
"dataset": {
    "name": "dstc2_ner_iterator"
}
```

There are no additional parameters in this part.

### Chainer

The chainer part of the configuration file contains the specification of the neural network 
model and supplementary things such as vocabularies.  The chainer part must have the following form:

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
for inference and train mode. "in_y" is used for training and usually contains ground truth answers. "out" field stands for model prediction. The model inside the pipe must have output variable with name "y_predicted" so that "out" knows where to get
predictions.

The major part of "chainer" is "pipe". The "pipe" contains the pre-processing modules, vocabularies and model. Firstly 
we define pre-processing:

```
"pipe": [
      {
        "in": ["x"],
        "name": "lazy_tokenizer",
        "out": ["x"]
      },
      {
        "in": ["x"],
        "name": "str_lower",
        "out": ["x_lower"]
      },
      {
        "in": ["x"],
        "name": "mask",
        "out": ["mask"]
      },
]
```
Module str_lower performs lowercasing. Module lazy_tokenizer performes tokenization if the elements of the batch are 
strings but not tokens. The mask module prepares masks for the network. It serves to cope with different lengths inputs
inside the batch. The mask is a matrix filled with ones and zeros. For instance, for two sentences batch with lengths 2
and 3 the mask will be \[\[1, 1, 0\],\[1, 1, 1\]\].

Then vocabularies must be defined:

```
"pipe": [
      ...
      {
        "in": ["x_lower"],
        "id": "word_vocab",
        "name": "simple_vocab",
        "pad_with_zeros": true,
        "fit_on": ["x_lower"],
        "save_path": "slotfill_dstc2/word.dict",
        "load_path": "slotfill_dstc2/word.dict",
        "out": ["x_tok_ind"]
      },
      {
        "in": ["y"],
        "id": "tag_vocab",
        "name": "simple_vocab",
        "pad_with_zeros": true,
        "fit_on": ["y"],
        "save_path": "slotfill_dstc2/tag.dict",
        "load_path": "slotfill_dstc2/tag.dict",
        "out": ["y_ind"]
      },
      ...
]
```

Parameters for vocabulary are:

- **`id`** - the name of the vocabulary which will be used in other models
- **`name`** - always equal to `"simple_vocab"`
- **`fit_on`** - on which data part of the data the vocabulary should be fitted (built), 
possible options are ["x"] or ["y"]
- **`save_path`** - path to a new file to save the vocabulary
- **`load_path`** - path to an existing vocabulary (ignored if there is no files)
- **`pad_with_zeros`**: whether to pad the resulting index array with zeros or not

Vocabularies are used for holding sets of tokens, tags, or characters. They assign indices to 
elements of given sets an allow conversion from tokens to indices and vice versa. Conversion of such kind 
is needed to perform lookup in embeddings matrices and compute cross-entropy between predicted 
probabilities and target values. For each vocabulary "simple_vocab" model is used. "fit_on" 
parameter defines on which part of the data the vocabulary is built. [\"x\"] stands for the x part of the 
data (tokens) and [\"y\"] stands for the y part (tags).

Then the embeddings matrix must be initialized:
```
"pipe": [
    ...
    {
      "name": "random_emb_mat",
      "id": "embeddings",
      "vocab_len": "#word_vocab.len",
      "emb_dim": 100
    },
    ...
]
```

The component `random_emb_mat` creates a matrix of embeddings filled with scaled gaussian random variables. Scaling is
similar to Xavier initialization.

Then the network is defined by the following part of JSON config:
```
"pipe": [
    ...
    {
        "in": ["x_tok_ind", "mask"],
        "in_y": ["y_ind"],
        "out": ["y_predicted"],
        "name": "ner",
        "main": true,
        "token_emb_mat": "#embeddings.emb_mat",
        "n_hidden_list": [64, 64],
        "net_type": "cnn",
        "n_tags": "#tag_vocab.len",
        "save_path": "slotfill_dstc2/model",
        "load_path": "slotfill_dstc2/model",
        "embeddings_dropout": true,
        "top_dropout": true,
        "intra_layer_dropout": true,
        "use_batch_norm": true,
        "learning_rate": 1e-2,
        "dropout_keep_prob": 0.5
    },
    ...
]
```

All network parameters are:
- **`in`** - inputs to be taken from the shared memory. Treated as x. They are used both 
during the training and inference.
- **`in_y`** - the target or y input to be taken from shared memory. This input is used during
 the training.
- **`name`** - the name of the model to be used. In this case we use 'ner' model originally 
imported from `deeppavlov.models.ner.ner`. We use only 'ner' name relying on the @registry 
decorator.
- **`main`** - (reserved for future use) a boolean parameter defining whether this is the main model. 
- **`save_path`** - path to the new file where the model will be saved
- **`load_path`** - path to a pretrained model from where it will be loaded
- **`token_emb_mat`** - token embeddings matrix
- **`net_type`** - type of the network, either 'cnn' or 'rnn'
- **`n_tags`** - number of tags in the tag vocabulary
- **`filter_width`** - the width of the convolutional kernel for Convolutional Neural Networks
- **`embeddings_dropout`** - boolean, whether to use dropout on embeddings or not
- **`intra_layer_dropout`** - boolean, whether to use dropout between layers or not
- **`top_dropout`** - boolean, whether to use dropout on output units of the network or not
- **`n_hidden_list`** - a list of output feature dimensionality for each layer. A value `[100, 200]`
means that there will be two layers with 100 and 200 units, respectively.
- **`use_crf`** - boolean, whether to use Conditional Random Fields on top of the network (recommended)
- **`use_batch_norm`** - boolean, whether to use Batch Normalization or not. Affects only CNN networks
- **`dropout_keep_prob`** - probability of keeping the hidden state, values from 0 to 1. 0.5 
works well in most of the cases
- **`learning_rate`**: learning rate to use during the training (usually from 0.1 to 0.0001)

The output of the network are indices of tags predicted by the network. They must be converted back to the tag strings.
This operation is performed by already created vocabulary:

```
"pipe": [
    ...
      {
        "ref": "tag_vocab",
        "in": ["y_predicted"],
        "out": ["tags"]
      }
    ...
```
In this part of config reusing pattern is used. The `ref` parameter serves to refer to already existing component via
 `id`. This part also illustrate omidirectionality of the vocabulary. When strings are passed to the vocab, it convert 
 them into indices. When the indices are passed to the vocab, they are converted to the tag strings.

The last component in the pipeline is the `slotfiller`:

```
"pipe": [
    {
        "in": ["x_lower", "tags"],
        "name": "dstc_slotfilling",
        "save_path": "slotfill_dstc2/dstc_slot_vals.json",
        "load_path": "slotfill_dstc2/dstc_slot_vals.json",
        "out": ["slots"]
    }
```
The `slotfiller` takes the tags and tokens to perform normalization of extracted entities. The normalization is
performed via fuzzy Levenshtein search in dstc_slot_vals dictionary. The output of this component is dictionary of
slot values found in the input utterances.

After the "chainer" part you should specify the "train" part:

```
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

The last part of the config is metadata:
```
"metadata": {
    "labels": {
      "telegram_utils": "NERModel"
    },
    "download": [
      "http://lnsigo.mipt.ru/export/deeppavlov_data/slotfill_dstc2.tar.gz"
    ]
  }
```
It contains information for deployment of the model and urls for download pre-trained models.

And now all parts together:
```json
{
  "dataset_reader": {
    "name": "dstc2_reader",
    "data_path": "dstc2"
  },
  "dataset_iterator": {
    "name": "dstc2_ner_iterator",
    "dataset_path": "dstc2"
  },
  "chainer": {
    "in": ["x"],
    "in_y": ["y"],
    "pipe": [
      {
        "in": ["x"],
        "name": "lazy_tokenizer",
        "out": ["x"]
      },
      {
        "in": ["x"],
        "name": "str_lower",
        "out": ["x_lower"]
      },
      {
        "in": ["x"],
        "name": "mask",
        "out": ["mask"]
      },
      {
        "in": ["x_lower"],
        "id": "word_vocab",
        "name": "simple_vocab",
        "pad_with_zeros": true,
        "fit_on": ["x_lower"],
        "save_path": "slotfill_dstc2/word.dict",
        "load_path": "slotfill_dstc2/word.dict",
        "out": ["x_tok_ind"]
      },
      {
        "in": ["y"],
        "id": "tag_vocab",
        "name": "simple_vocab",
        "pad_with_zeros": true,
        "fit_on": ["y"],
        "save_path": "slotfill_dstc2/tag.dict",
        "load_path": "slotfill_dstc2/tag.dict",
        "out": ["y_ind"]
      },
      {
        "name": "random_emb_mat",
        "id": "embeddings",
        "vocab_len": "#word_vocab.len",
        "emb_dim": 100
      },
      {
        "in": ["x_tok_ind", "mask"],
        "in_y": ["y_ind"],
        "out": ["y_predicted"],
        "name": "ner",
        "main": true,
        "token_emb_mat": "#embeddings.emb_mat",
        "n_hidden_list": [64, 64],
        "net_type": "cnn",
        "n_tags": "#tag_vocab.len",
        "save_path": "slotfill_dstc2/model",
        "load_path": "slotfill_dstc2/model",
        "embeddings_dropout": true,
        "top_dropout": true,
        "intra_layer_dropout": false,
        "use_batch_norm": true,
        "learning_rate": 1e-2,
        "dropout_keep_prob": 0.5
      },
      {
        "ref": "tag_vocab",
        "in": ["y_predicted"],
        "out": ["tags"]
      },
      {
        "in": ["x_lower", "tags"],
        "name": "dstc_slotfilling",
        "save_path": "slotfill_dstc2/dstc_slot_vals.json",
        "load_path": "slotfill_dstc2/dstc_slot_vals.json",
        "out": ["slots"]
      }
    ],
    "out": ["tags"]
  },
  "train": {
    "epochs": 100,
    "batch_size": 64,

    "metrics": ["ner_f1", "per_item_accuracy"],
    "validation_patience": 5,
    "val_every_n_epochs": 5,

    "log_every_n_batches": 100,
    "show_examples": false
  },
  "metadata": {
    "labels": {
      "telegram_utils": "NERModel"
    },
    "download": [
      "http://lnsigo.mipt.ru/export/deeppavlov_data/slotfill_dstc2.tar.gz"
    ]
  }
}
```

## Train and use the model

Please see an example of training a Slot Filling model and using it for prediction:

```python
from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.infer import interact_model
PIPELINE_CONFIG_PATH = 'configs/ner/slotfill_dstc2.json'
train_model_from_config(PIPELINE_CONFIG_PATH)
interact_model(PIPELINE_CONFIG_PATH)
```

This example assumes that the working directory is deeppavlov.


## Slotfilling withot NER

An alternative approach to Slot Filling problem could be fuzzy search for each instance of each slot value inside the
 text. This approach is realized in `slotfill_raw` component. The component uses needle in haystack 
 
The main advantage of this approach is elimination of a separate Named Entity Recognition module. However,
absence of NER module make this model less robust to noise (words with similar spelling) especially for long 
utterances.

Usage example:

```python
from deeppavlov.core.commands.infer import interact_model
PIPELINE_CONFIG_PATH = 'configs/ner/slotfill_dstc2_raw.json'
interact_model(PIPELINE_CONFIG_PATH)
```