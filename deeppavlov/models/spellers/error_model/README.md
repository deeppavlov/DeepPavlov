[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Automatic spelling correction component

Automatic spelling correction component is based on
[An Improved Error Model for Noisy Channel Spelling Correction](http://www.aclweb.org/anthology/P00-1037)
by Eric Brill and Rober C. Moore and uses statistics based error model,
a static dictionary and an ARPA language model to correct spelling errors.  
We provide everything you need to build a spelling correction module for russian and english languages
and some guidelines for how to collect appropriate datasets for other languages.

## Usage

model parameters:  
* `name` always equals to `"spelling_error_model"`
* `train_now` — without this flag set to `true` train phase of an error model will be skipped
* `model_file` — name of the file that the model will be saved to and loaded from, defaults to `"error_model.tsv"` 
* `window` — window size for the error model from `0` to `4`, defaults to `1`
* `lm_file` — path to the ARPA language model file. If omitted, all of the dictionary words will be handled as equally probable
* `dictionary`
    *
    *

This model expects a sentence string with spaced-separated tokens in lowercase as it's input and returns the same string with corrected words

```python
import json
import sys

from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.commands.utils import set_usr_dir

CONFIG_PATH = 'deeppavlov/models/spellers/error_model/config_ru_custom_vocab.json'
usr_dir = set_usr_dir(CONFIG_PATH)

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

model = build_model_from_config(config)
for line in sys.stdin:
    print(model.infer(line), flush=True)
```

## Training

#### Error model

!how to train an error model:  
!write your own dataset_reader, run train

```python
from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.utils import set_usr_dir

MODEL_CONFIG_PATH = 'deeppavlov/models/spellers/error_model/config_ru_custom_vocab.json'
usr_dir = set_usr_dir(MODEL_CONFIG_PATH)
train_model_from_config(MODEL_CONFIG_PATH)
```

#### Language model

This model uses [KenLM](http://kheafield.com/code/kenlm/) to process language models, so if you want to build your own,
we suggest you consult with it's website. We do also provide our own language models for
[english](http://lnsigo.mipt.ru/export/lang_models/en_wiki_no_punkt.arpa.binary.gz) \(5.5GB\) and
[russian](http://lnsigo.mipt.ru/export/lang_models/ru_wiyalen_no_punkt.arpa.binary.gz) \(5GB\) languages.

## Ways to improve

* locate bottlenecks in code and rewrite them in Cython to improve performance
* find a way to add skipped spaces and remove superfluous ones
* find or learn a proper balance between an error model and a language model scores when ranking candidates
* implement [Discriminative Reranking for Spelling Correction](http://www.aclweb.org/anthology/Y06-1009)
by Yang Zhang, Pilian He, Wei Xiang and Mu Li