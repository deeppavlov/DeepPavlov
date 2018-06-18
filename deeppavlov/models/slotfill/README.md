[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Neural Named Entity Recognition and Slot Filling

This component solves Slot-Filling task using Levenshtein search and different neural network architectures for NER.
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

Slot Filling is a typical step after the NER. It can be formulated as:

_Given an entity of a certain type and a set of all possible values of this entity type provide a normalized form of the entity._

In this component, the Slot Filling task is solved by Levenshtein Distance search across all known entities of a given type. Example:

There is an entity of "food" type:

_chainese_

It is definitely misspelled. The set of all known food entities is {'chinese', 'russian', 'european'}. The nearest known entity from the given set is _chinese_. So the output of the Slot Filling system will be _chinese_.


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

The major part of "chainer" is "pipe". The "pipe" contains the pre-processing modules, vocabularies and model. However,
we can use existing pipelines:

```
"pipe": [
      {
        "in": ["x"],
        "name": "lazy_tokenizer",
        "out": ["x"]
      },
      {
        "in": ["x"],
        "config_path": "../deeppavlov/configs/ner/ner_dstc2.json",
        "out": ["tags"]
      },
      ...
]
```

This part will initialize already existing pre-trained NER module. The only thing need to be specified is path to existing
config. The preceding lazy tokenizer serves to extract tokens for raw string of text. 

The following component in the pipeline is the `slotfiller`:

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

The main part of the `dstc_slotfilling` componet is the slot values dictionary. The dicttionary has the following 
structure:
```
{
    "entity_type_0": {
        "entity_value_0": [
            "entity_value_0_variation_0",
            "entity_value_0_variation_1",
            "entity_value_0_variation_2"
        ],
        "entity_value_1": [
            "entity_value_1_variation_0"
        ],
        ...
    }
    "entity_type_1": {
        ...
```

Slotfiller will perform fuzzy search through the all variations of all entity values of given entity type. The entity type is determined 
by the NER component.

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

## Usage of the model

Please see an example of training a Slot Filling model and using it for prediction:

```python
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.download import deep_download
import json
PIPELINE_CONFIG_PATH = 'deeppavlov/configs/ner/slotfill_dstc2.json'
with open(PIPELINE_CONFIG_PATH) as f:
    config = json.load(f)
deep_download(['-c', PIPELINE_CONFIG_PATH])
slotfill_model = build_model_from_config(config)
slotfill_model(['I would like some chinese food', 'The west part of the city would be nice'])
```

This example assumes that the working directory is the root of the project.


## Slotfilling without NER

An alternative approach to Slot Filling problem could be fuzzy search for each instance of each slot value inside the
 text. This approach is realized in `slotfill_raw` component. The component uses needle in haystack 
 
The main advantage of this approach is elimination of a separate Named Entity Recognition module. However,
absence of NER module make this model less robust to noise (words with similar spelling) especially for long 
utterances.

Usage example:

```python
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.download import deep_download
import json
PIPELINE_CONFIG_PATH = 'deeppavlov/configs/ner/slotfill_dstc2_raw.json'
with open(PIPELINE_CONFIG_PATH) as f:
    config = json.load(f)
deep_download(['-c', PIPELINE_CONFIG_PATH])
slotfill_model = build_model_from_config(config)
slotfill_model(['I would like some chinese food', 'The west part of the city would be nice'])
```