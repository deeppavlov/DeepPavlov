[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Neural Named Entity Recognition and Slot Filling

This component solves Named Entity Recognition (NER) and Slot Filling task with different neural network architectures. This component serves for solving DSTC 2 Slot Filling task.
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

In the example above PER means person tag, and "B-" and "I-" are prefixes identifying beginnings and continuations of the entities.

Slot Filling can be formulated as:

_Given some entity of certain type and a set of all possible values of this entity type provide a normalized form of the entity._

In this component the Slot Filling task is solved by Levenshtein Distance search across all known entities of given type. Example:

There is an entity of "food" type:

_chainese_

Definitely it is misspelled. The set of all known food entities is {'chinese', 'russian', 'european'}. The nearest known entity from the given set is _chinese_. So the output of the Slot Filling system should be _chinese_.


## Assemble the model

The system contains two main parts: Slot Filling System and Named Entity Recognition system. The first one is not trainable and the NER network is trainable. The NER network is an integral part of the Slot Filling component.

The config of the model must have the following fields:
```json
{
"model": {
    "name": "dstc_slotfilling",
    "train_now": true,
    "model_path": "models/ner/model/",
    "ner_network": {
        "name": "ner_tagging_network"
    }
}
```
wher "name" is always "dstc_slotfilling", reffering to the DstcSlotFillingNetwork class, "train_now" determine whether to train NER network or not, "model_path" defines the path to the files needed for slot filling, "ner_network" is a refference to the NerNetwork class, which has its own parametrs.

The NER network is a separate model and it has its own initialization parameters, namely:
```json
{
"ner_network": {
    "name": "ner_tagging_network",
    "vocabs": ["token_vocab", "tag_vocab", "char_vocab"],
    "model_path": "models/ner/model/",
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
}
```
- "name" is always equal to "ner_tagging_network",
- "vocabs" is equal to ["token_vocab", "tag_vocab", "char_vocab"] and specify vocabularies needed to construct the network (which will be mentioned below),
- "model_path" defines the path to save the network parameters files,
-"verbouse" - show supplimentary information about the network or not,
- "filter_width" - the width of convolutional kernel
- "embeddings_dropout" - whether to use dropout for embeddings or not
- "n_filters" - number of filters in convolutional network
- "token_embeddings_dim" - dimensionality of token embeddings
- "char_embeddings_dim" - dimensionality of character embeddings
- "use_batch_norm" - whether to use batch normalization or not
- "use_crf" - whether to apply Conditional Random Fields on top of the network

To perform convertation between tokens and indices there are three vocabularies in the config file:
```json
 { "vocabs": {
  "token_vocab": {
      "name": "default_vocab",
      "inputs": ["x"],
      "level": "token",
      "model_dir": "data/dstc2",
      "model_file": "token.dict"
    },
  "tag_vocab": {
      "name": "default_vocab",
      "inputs": ["y"],
      "level": "token",
      "model_dir": "data/dstc2",
      "model_file": "tag.dict"
    },
  "char_vocab":{
      "name": "default_vocab",
      "inputs": ["x"],
      "level": "char",
      "model_dir": "data/dstc2",
      "model_file": "char.dict"
    }
  }
 }
```
These vocabularies are used by NER network to perform converation between tokens and indices and vice versa. Indices are fed into the network to perform embeddings lookup. The are refference to this vocabularies in the NER network configuration part. These vocabularies are built before initialization of the network and provides parameters (number of tokens, number of tags, ...) to initialize lookup matrices. They are also used for convertation of batch tokens to batch indices.

To build vocabularies and train the network two data components must be specified. Namely: dataset_reader and datset:
```json
{"dataset_reader": {
      "name": "dstc2_datasetreader",
      "data_path": "/home/user/data/dstc2"
  },
  "dataset": {
      "name": "dstc2_ner_dataset",
      "dataset_path": "/home/user/data/dstc2"
  }
}
```
For DSTC 2 Slot Filling task only dataset_path should be specified. While "dstc2_datasetreader" performs downloading of the raw dataset, "dstc2_ner_dataset" extracts entities from the raw data forming pairs of samples (utterance_tokens_list, utterance_tags_list) and forms dataset, which is essentialy a python dict with fields "train", "valid", and "test". In eacch field a list of samples is stored. The samples from the dataset is used to build dictionaries.

## Training

To train the model the config file specifed above must be formed. The field "train_now" in the "model" component must be set to true. Then the following script can be used to train the model:

This model expects a sentence string with spaced-separated tokens in lowercase as it's input and returns the same string with corrected words

```python
from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.infer import interact_model
from deeppavlov.core.commands.utils import set_usr_dir

MODEL_CONFIG_PATH = 'models/ner/config.json'
usr_dir = set_usr_dir(MODEL_CONFIG_PATH)
train_model_from_config(MODEL_CONFIG_PATH)
```


## Inference

After training the following code will launch interaction with the model:

```python
from deeppavlov.core.commands.infer import interact_model
from deeppavlov.core.commands.utils import set_usr_dir

CONFIG_PATH = 'deeppavlov/models/ner/config.json'
usr_dir = set_usr_dir(CONFIG_PATH)
interact_model(MODEL_CONFIG_PATH)
```

## Results

The NER network component reproduce architecture from the paper "_Application of a Hybrid Bi-LSTM-CRF model to the task of Russian Named Entity Recognition_" https://arxiv.org/pdf/1709.09686.pdf, which is inspired by LSTM+CRF architecture from https://arxiv.org/pdf/1603.01360.pdf.

Bi-LSTM architecture of NER network were tested on three datasets:
- Gareev corpus [1] (obtainable by request to authors)
- FactRuEval 2016 [2]
- Persons-1000 [3]

The F1 measure for the model along with other published solution provided in the table below:

| Models                | Gareev’s dataset | Persons-1000 | FactRuEval 2016 |
|---------------------- |:----------------:|:------------:|:---------------:|
| Gareev et al. [1]     | 75.05            |              |                 |
| Malykh et al. [4]     | 62.49            |              |                 |
| Trofimov  [5]         |                  | 95.57        |                 |
| Rubaylo et al. [6]    |                  |              | 78.13           |
| Sysoev et al. [7]     |                  |              | 74.67           |
| Ivanitsky et al.  [7] |                  |              | **87.88**       |
| Mozharova et al.  [8] |                  | 97.21        |                 |
| Our (Bi-LSTM+CRF)     | **87.17**        | **99.26**    | 82.10           ||

## Literature

[1] - Rinat Gareev, Maksim Tkachenko, Valery Solovyev, Andrey Simanovsky, Vladimir Ivanov: Introducing Baselines for Russian Named Entity Recognition. Computational Linguistics and Intelligent Text Processing, 329 -- 342 (2013).

[2] - https://github.com/dialogue-evaluation/factRuEval-2016

[3] - http://ai-center.botik.ru/Airec/index.php/ru/collections/28-persons-1000

[4] -  Reproducing Russian NER Baseline Quality without Additional Data. In proceedings of the 3rd International Workshop on ConceptDiscovery in Unstructured Data, Moscow, Russia, 54 – 59 (2016)

[5] - Rubaylo A. V., Kosenko M. Y.: Software utilities for natural language information
retrievial. Almanac of modern science and education, Volume 12 (114), 87 – 92.(2016)

[6] - Sysoev A. A., Andrianov I. A.: Named Entity Recognition in Russian: the Power of Wiki-Based Approach. dialog-21.ru

[7] - Ivanitskiy Roman, Alexander Shipilo, Liubov Kovriguina: Russian Named Entities Recognition and Classification Using Distributed Word and Phrase Representations. In SIMBig, 150 – 156. (2016).

[8] - Mozharova V., Loukachevitch N.: Two-stage approach in Russian named entity recognition. In Intelligence, Social Media and Web (ISMW FRUCT), 2016 International FRUCT Conference, 1 – 6 (2016)