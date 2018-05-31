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

#### Component config parameters:  
* `in` — list with one element: name of this component's input in chainer's shared memory
* `out` — list with one element: name for this component's output in chainer's shared memory
* `name` always equals to `"spelling_error_model"`
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

This module expects sentence strings with space-separated tokens in lowercase as its input, so it is advised to add
appropriate preprocessing in chainer.

A working config could look like this:

```json
{
  "chainer":{
    "in": ["x"],
    "pipe": [
      {
        "name": "str_lower",
        "in": ["x"],
        "out": ["x_lower"]
      },
      {
        "name": "nltk_tokenizer",
        "in": ["x_lower"],
        "out": ["x_tokens"]
      },
      {
        "in": ["x_tokens"],
        "out": ["y_predicted"],
        "name": "spelling_error_model",
        "window": 1,
        "save_path": "error_model/error_model.tsv",
        "load_path": "error_model/error_model.tsv",
        "dictionary": {
          "name": "wikitionary_100K_vocab"
        },
        "lm_file": "/data/data/enwiki_no_punkt.arpa.binary"
      }
    ],
    "out": ["y_predicted"]
  }
}
```

#### Usage example
This model expects a sentence string with space-separated tokens in lowercase as its input and returns the same string with corrected words.
Here's an example code that will read input data from stdin line by line and output resulting text into stdout:

```python
import json
import sys

from deeppavlov.core.commands.infer import build_model_from_config

CONFIG_PATH = 'configs/error_model/brillmoore_kartaslov_ru.json'

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

model = build_model_from_config(config)
for line in sys.stdin:
    print(model([line])[0], flush=True)
```

if we save it as `example.py` then it could be used like so:

```bash
cat input.txt | python3 example.py > out.txt
```

## Training

#### Error model

For the training phase config file needs to also include these parameters:

* `dataset_iterator` — it should always be set like `"dataset_iterator": {"name": "typos_iterator"}`
    * `name` always equals to `typos_iterator`
    * `test_ratio` — ratio of test data to train, from `0.` to `1.`, defaults to `0.`
* `dataset_reader`
    * `name` — `typos_custom_reader` for a custom dataset or one of two provided:
        * `typos_kartaslov_reader` to automatically download and process misspellings dataset for russian language from
         [https://github.com/dkulagin/kartaslov/tree/master/dataset/orfo_and_typos](https://github.com/dkulagin/kartaslov/tree/master/dataset/orfo_and_typos)
        * `typos_wikipedia_reader` to automatically download and process
         [a list of common misspellings from english Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines)
    * `data_path` — required for typos_custom_reader as a path to a dataset file,
     where each line contains a misspelling and a correct spelling of a word separated by a tab symbol

Component's configuration also has to have as `fit_on` parameter — list of two elements: names of component's input
and true output in chainer's shared memory

A working training config could look something like:

```json
{
  "dataset_reader": {
    "name": "typos_wikipedia_reader"
  },
  "dataset_iterator": {
    "name": "typos_iterator",
    "test_ratio": 0.05
  },
  "chainer":{
    "in": ["x"],
    "in_y": ["y"],
    "pipe": [
      {
        "name": "str_lower",
        "id": "lower",
        "in": ["x"],
        "out": ["x_lower"]
      },
      {
        "name": "nltk_tokenizer",
        "id": "tokenizer",
        "in": ["x_lower"],
        "out": ["x_tokens"]
      },
      {
        "ref": "lower",
        "in": ["y"],
        "out": ["y_lower"]
      },
      {
        "ref": "tokenizer",
        "in": ["y"],
        "out": ["y_tokens"]
      },
      {
        "fit_on": ["x_tokens", "y_tokens"],
        "in": ["x_tokens"],
        "out": ["y_predicted"],
        "name": "spelling_error_model",
        "window": 1,
        "dictionary": {
          "name": "wikitionary_100K_vocab"
        },
        "save_path": "error_model/error_model.tsv",
        "load_path": "error_model/error_model.tsv"
      }
    ],
    "out": ["y_predicted"]
  },
  "train": {
    "validate_best": false,
    "test_best": true
  }
}
```

And a script to use this config:

```python
from deeppavlov.core.commands.train import train_model_from_config

MODEL_CONFIG_PATH = 'configs/error_model/brillmoore_wikitypos_en.json'
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
* use multiprocessing or multithreading for batch elements
* find a way to add skipped spaces and remove superfluous ones
* find or learn a proper balance between an error model and a language model scores when ranking candidates
* implement [Discriminative Reranking for Spelling Correction](http://www.aclweb.org/anthology/Y06-1009)
by Yang Zhang, Pilian He, Wei Xiang and Mu Li
* use a better dataset for getting misspellings statistics
* add handcrafted features to use phonetic information
