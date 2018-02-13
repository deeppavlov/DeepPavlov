# Named Entity Recognition (NER)

## NER task

Named Entity Recognition (NER) is one of the most common tasks in natural language processing. 
In most of the cases, NER task can be formulated as: 

_Given a sequence of tokens (words, and maybe punctuation symbols) provide a tag from predefined set of tags for each token in the sequence._

For NER task there are some common types of entities which essentially are tags:
- persons
- locations
- organizations
- expressions of time
- quantities
- monetary values 

Furthermore, to distinguish consequent entities with the same tags BIO tagging scheme is used. "B" stands for beginning, 
"I" stands for the continuation of an entity and "O" means the absence of entity. Example with dropped punctuation:

    Bernhard        B-PER
    Riemann         I-PER
    Carl            B-PER
    Friedrich       I-PER
    Gauss           I-PER
    and             O
    Leonhard        B-PER
    Euler           I-PER

In the example above PER means person tag, and "B-" and "I-" are prefixes identifying beginnings and continuations of the entities. Without such prefixes, it is impossible to separate Bernhard Riemann from Carl Friedrich Gauss.

## Training data
To train the neural network, you need to have a dataset in the following format:

    EU B-ORG
    rejects O
    the O
    call O
    of O
    Germany B-LOC
    to O
    boycott O
    lamb O
    from O
    Great B-LOC
    Britain I-LOC
    . O
    
    China B-LOC
    says O
    time O
    right O
    for O
    Taiwan B-LOC
    talks O
    . O

    ...

The source text is tokenized and tagged. For each token, there is a separate tag with BIO markup. Tags are separated from tokens with whitespaces. Sentences are separated by empty lines.

The dataset is a text file or a set of text files.
The dataset must be split into three partitions: train, test, and validation. The train set is used for training the network, namely adjusting the weights with gradient descent. The validation set is used for monitoring learning progress and early stopping. The test set is used for final estimation of model quality. Typical partitions of train, validation, and test are 80%, 10%, 10% respectively.


## Configuration of the model

Configuration of the model can be performed in code or in JSON configuration file. To train the model four groups of parameters must be specified:

- **`dataset_reader`**
- **`dataset`**
- **`model`**
- **`vocabs`**

The following parts assume that config file is used. However, it can be used in the code.

### Dataset Reader

The dataset reader is a class which reads and parses the data. It returns a dictionary with three fields: "train", "test", and "valid". The basic dataset reader is "ner_dataset_reader." The dataset reader config part with "ner_dataset_reader" should look like:
```json
"dataset_reader": {
    "name": "ner_dataset_reader",
    "data_path": "/home/user/Data/conll2003/"
} 
```

where "name" refers to the basic ner dataset reader class and data_path is the path to the folder with three files, namely: "train.txt", "valid.txt", and "test.txt". Each file contains data in the format presented in 
*Training data* section. Each line in the file may contain additional information such as POS tags. However, the token must be first in line and tags must be last.

### Dataset

In order to perform simple batching and shuffling "basic_dataset" is used. The part of the configuration file for the dataset looks like:
 ```json
"dataset": {
    "name": "basic_dataset"
}
```

There is no additional parameters to this part.

### Model

The model part of the configuration file contains the specification of the neural network model. 
 ```json
"model": {
    "name": "ner",
    "train_now": true,
    "epochs": 3,
    "save_path": "ner_model/ner_model",
    "load_path": "ner_model/ner_model",
    "vocabs": ["token_vocab", "tag_vocab", "char_vocab"],
    "verbouse": true,
    "filter_width": 7,
    "embeddings_dropout": true,
    "n_filters": [
      128,
      128
    ],
    "token_embeddings_dim": 64,
    "char_embeddings_dim": 32,
    "use_batch_norm": true,
    "use_crf": true,
    "learning_rate": 1e-3,
    "dropout_rate": 0.5
}
```

There are network parameters and leaning parameters. Network parameters are:

- **`token_embeddings_dim`** - token embeddings dimensionality (must agree with embeddings if they are provided), typical values are from 100 to 300
- **`char_embeddings_dim`** - character embeddings dimensionality, typical values are 25 - 100
- **`use_crf`** - whether to use Conditional Random Fields on the top (suggested to always use True)
- **`use_capitalization`** - whether to include capitalization binary features to the input of the network. If True than binary feature indicating whether the word starts with a capital letter will be concatenated to the word embeddings.
- **`_n_filters_`** - list of output feature dimensionality for each layer. For [100, 200] there will be two layers with 100 and 200 number of units respectively. 
- **`embeddings_dropout`** - whether to use dropout on embeddings
- **`save_path`** - path to the model where it will be saved
- **`load_path`** - path to load pretrained model

The training parameters are:

- **`dropout_rate`** - probability of dropping the hidden state a value from 0 to 1. 0.5 Works well in most of the cases
- **`epochs`** - number of epochs (10 - 100 typical)
- **`learning_rate`**: learning rate (0.01 - 0.0001 typical)
- **`batch_size`**: number of samples in the batch (4 - 64 typical)
- **`learning_rate_decay`** - multiple factor of decreasing learning rate every epoch (1 - 0.5 typical)

### Vocabularies

Vocabularies are used for holding sets of tokens, tags, or characters. They assign indexes to given sets an allow conversion from tokens to indices and vice versa. Conversion of such kind is needed to perform lookup in embeddings matrices and compute cross-entropy between predicted probabilities and target values.

Vocabularies form a block of the following form:
```json
"vocabs": {
    "word_vocab": {
        "name": "default_vocab",
        "inputs": ["x"],
        "train_now": true,
        "level": "token",
        "save_path": "ner_model/word.dict",
        "load_path": "ner_model/word.dict"
      },
    "tag_vocab": {
        "name": "default_vocab",
        "inputs": ["y"],
        "train_now": true,
        "level": "token",
        "save_path": "ner_model/tag.dict",
        "load_path": "ner_model/tag.dict"
      },
    "char_vocab":{
        "name": "default_vocab",
        "inputs": ["x"],
        "train_now": true,
        "level": "char",
        "save_path": "ner_model/char.dict",
        "load_path": "ner_model/char.dict"
      }
}
```

wher each vocabulary ("word_vocab", "tag_vocab", or "char_vocab") there are parameters:

- **`name`** - always equal to `"default_vocab"`
- **`inputs`** -  which input is use: x or y (tokens or tags), can be `["x"]` for tokens or `["y"]` for tags.
- **`train_now`** - whether to train (build) vocab or not. If false there assumed to be a vocabulary in `load_path`
- **`level`** -char or token level tokenization
- **`save_path`** - path to the model where it will be saved
- **`load_path`** - path to load pretrained model

## Training

To train the network there must be a json configuration file like described above. Then the following code can be used 
for training and interacting the model:

```python
from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.infer import interact_model
from deeppavlov.core.commands.utils import set_usr_dir, get_usr_dir

PIPELINE_CONFIG_PATH = 'models/ner/ner_config.json'
set_usr_dir(PIPELINE_CONFIG_PATH)
train_model_from_config(PIPELINE_CONFIG_PATH)
interact_model(PIPELINE_CONFIG_PATH)
```

## Inference

Like in training stage the configuration file must be used. The only difference from training in the absence of training stage.

```python
from deeppavlov.core.commands.infer import interact_model
from deeppavlov.core.commands.utils import set_usr_dir

PIPELINE_CONFIG_PATH = 'models/ner/ner_config.json'
set_usr_dir(PIPELINE_CONFIG_PATH)
interact_model(PIPELINE_CONFIG_PATH)
```

## Results

The NER network component reproduces architecture from the paper "_Application of a Hybrid Bi-LSTM-CRF model to the task of Russian Named Entity Recognition_" https://arxiv.org/pdf/1709.09686.pdf, which is inspired by LSTM+CRF architecture from https://arxiv.org/pdf/1603.01360.pdf.

Bi-LSTM architecture of NER network was tested on three datasets:
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
