[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Automatic spelling correction component

Automatic spelling correction component is based on
[An Improved Error Model for Noisy Channel Spelling Correction](http://www.aclweb.org/anthology/P00-1037)
by Eric Brill and Robert C. Moore and uses statistics based error model,
a static dictionary and an ARPA language model to correct spelling errors.  
We provide everything you need to build a spelling correction module for russian and english languages
and some hints on how to collect appropriate datasets for other languages.

## Usage

#### Config parameters:  
* `name` always equals to `"spelling_error_model"`
* `train_now` — without this flag set to `true` train phase of an error model will be skipped
* `save_path` — path where the model will be saved at after a training session
* `load_path` — path to the pretrained model
* `window` — window size for the error model from `0` to `4`, defaults to `1`
* `lm_file` — path to the ARPA language model file. If omitted, all of the dictionary words will be handled as equally probable
* `dictionary` — description of a static dictionary model, instance of (or inherited from) `deeppavlov.vocabs.static_dictionary.StaticDictionary`
    * `name` — `"static_dictionary"` for a custom dictionary or one of two provided:
        * `"russian_words_vocab"` to automatically download and use a list of russian words from [https://github.com/danakt/russian-words/](https://github.com/danakt/russian-words/)  
        * `"wikitionary_100K_vocab"` to automatically download a list of most common words from Project Gutenberg from [Wiktionary](https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists#Project_Gutenberg)
     
    * `dictionary_name` — name of a directory where a dictionary will be built to and loaded from, defaults to `"dictionary"` for static_dictionary
    * `raw_dictionary_path` — path to a file with a line-separated list of dictionary words, required for static_dictionary

A working config could look like this:

```json
{
  "model": {
    "name": "spelling_error_model",
    "save_path": "error_model/error_model.tsv",
    "load_path": "error_model/error_model.tsv",
    "train_now": true,
    "window": 1,
    "dictionary": {
      "name": "wikitionary_100K_vocab"
    },
    "lm_file": "/data/data/enwiki_no_punkt.arpa.binary"
  }
}
```

#### Usage example
This model expects a sentence string with space-separated tokens in lowercase as its input and returns the same string with corrected words.
Here's an example code that will read input data from stdin line by line and output resulting text to the `output.txt` file:

```python
import json
import sys

from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.commands.utils import set_usr_dir

CONFIG_PATH = 'configs/error_model/config_ru_custom_vocab.json'
set_usr_dir(CONFIG_PATH)

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

model = build_model_from_config(config)
with open('output.txt', 'w') as f:
    for line in sys.stdin:
        print(model.infer(line), file=f, flush=True)
```

if we save it as `example.py` then it could be used like so:

```bash
cat input.txt | python3 example.py
```

## Training

#### Error model

For the training phase config file needs to also include these parameters:

* `dataset` — it should always be set like `"dataset": {"name": "typos_dataset"}`
* `dataset_reader`
    * `name` — `typos_custom_reader` for a custom dataset or one of two provided:
        * `typos_kartaslov_reader` to automatically download and process misspellings dataset for russian language from
         [https://github.com/dkulagin/kartaslov/tree/master/dataset/orfo_and_typos](https://github.com/dkulagin/kartaslov/tree/master/dataset/orfo_and_typos)
        * `typos_wikipedia_reader` to automatically download and process
         [a list of common misspellings from english Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines)
    * `data_path` — required for typos_custom_reader as a path to a dataset file,
     where each line contains a misspelling and a correct spelling of a word separated by a tab symbol

A working training config could look something like:

```json
{
  "model": {
    "name": "spelling_error_model",
    "save_path": "error_model/error_model.tsv",
    "load_path": "error_model/error_model.tsv",
    "window": 1,
    "train_now": true,
    "dictionary": {
      "name": "wikitionary_100K_vocab"
    }
  },
  "dataset_reader": {
    "name": "typos_wikipedia_reader"
  },
  "dataset": {
    "name": "typos_dataset"
  }
}
```

And a script to use this config:

```python
from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.utils import set_usr_dir

MODEL_CONFIG_PATH = 'configs/error_model/config_en.json'
set_usr_dir(MODEL_CONFIG_PATH)
train_model_from_config(MODEL_CONFIG_PATH)
```

#### Language model

This model uses [KenLM](http://kheafield.com/code/kenlm/) to process language models, so if you want to build your own,
we suggest you consult with its website. We do also provide our own language models for
[english](http://lnsigo.mipt.ru/export/lang_models/en_wiki_no_punkt.arpa.binary.gz) \(5.5GB\) and
[russian](http://lnsigo.mipt.ru/export/lang_models/ru_wiyalen_no_punkt.arpa.binary.gz) \(3.1GB\) languages.

## Comparison

We compared this module with [Yandex.Speller](http://api.yandex.ru/speller/) and [GNU Aspell](http://aspell.net/)
on the [test set](http://www.dialog-21.ru/media/3838/test_sample_testset.txt)
for the [SpellRuEval competition](http://www.dialog-21.ru/en/evaluation/2016/spelling_correction/) on Automatic Spelling Correction for Russian:

| Correction method                          | Precision | Recall | F-measure | 
|--------------------------------------------|-----------|--------|-----------|
| Yandex.Speller                             | 83.09     | 59.86  | 69.59     | 
| Our model with the provided language model | 51.92     | 53.94  | 52.91     | 
| Our model with no language model           | 41.42     | 37.21  | 39.20     | 
| GNU Aspell, always first candidate         | 27.85     | 34.07  | 30.65     |

## Ways to improve

* locate bottlenecks in code and rewrite them in Cython to improve performance
* find a way to add skipped spaces and remove superfluous ones
* find or learn a proper balance between an error model and a language model scores when ranking candidates
* implement [Discriminative Reranking for Spelling Correction](http://www.aclweb.org/anthology/Y06-1009)
by Yang Zhang, Pilian He, Wei Xiang and Mu Li
* use a better dataset for getting misspellings statistics
* add handcrafted features to use phonetic information
