[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Automatic spelling correction pipelines

We provide two types of pipelines for spelling correction: [levenstein_corrector](#levenstein_corrector)
 uses simple Damerau-Levenstein distance to find correction candidates and [brillmoore](#brillmoore)
 uses statistics based error model for it. In both cases correction candidates are chosen based on context
 with the help of a [kenlm language model](#language-model).  
You can find [the comparison](#comparison) of these and other approaches near the end of this readme.

## Quick start

You can run the following command to try provided pipelines out:

```
python -m deeppavlov interact <path_to_config> [-d]
```

where `<path_to_config>` is one of the [provided config files](/deeppavlov/configs/spelling_correction).  
With the optional `-d` parameter all the data required to run selected pipeline will be downloaded, including
 an appropriate language model.
 

After downloading the required files you can use these configs in your python code.
 For example, this code will read lines from stdin and print corrected lines to stdout:

```python
import json
import sys

from deeppavlov.core.commands.infer import build_model_from_config

CONFIG_PATH = 'deeppavlov/configs/spelling_correction/brillmoore_kartaslov_ru.json'

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

model = build_model_from_config(config)
for line in sys.stdin:
    print(model([line])[0], flush=True)
```

## levenstein_corrector

[This component](levenstein/searcher_component.py) finds all the candidates in a static dictionary
 on set Damerau-Levenstein distance.  
It can separate one token into two but it will not work the other way around.

#### Component config parameters:
* `in` — list with one element: name of this component's input in chainer's shared memory
* `out` — list with one element: name for this component's output in chainer's shared memory
* `name` always equals to `"spelling_levenstein"`. Optional if `class` attribute is present
* `class` always equals to `deeppavlov.models.spelling_correction.levenstein.searcher_component:LevensteinSearcherComponent`. Optional if `name` attribute is present
* `words` — list of all correct words (should be a reference)
* `max_distance` — maximum allowed Damerau-Levenstein distance between source words and candidates
* `error_probability` — assigned probability for every edit


## brillmoore

[This component](brillmoore/error_model.py) is based on
[An Improved Error Model for Noisy Channel Spelling Correction](http://www.aclweb.org/anthology/P00-1037)
by Eric Brill and Robert C. Moore and uses statistics based error model to find best candidates in a static dictionary.

#### Component config parameters:  
* `in` — list with one element: name of this component's input in chainer's shared memory
* `out` — list with one element: name for this component's output in chainer's shared memory
* `name` always equals to `"spelling_error_model"`. Optional if `class` attribute is present
* `class` always equals to `deeppavlov.models.spelling_correction.brillmoore.error_model:ErrorModel`. Optional if `name` attribute is present
* `save_path` — path where the model will be saved at after a training session
* `load_path` — path to the pretrained model
* `window` — window size for the error model from `0` to `4`, defaults to `1`
* `candidates_count` — maximum allowed count of candidates for every source token
* `dictionary` — description of a static dictionary model, instance of (or inherited from) `deeppavlov.vocabs.static_dictionary.StaticDictionary`
    * `name` — `"static_dictionary"` for a custom dictionary or one of two provided:
        * `"russian_words_vocab"` to automatically download and use a list of russian words from [https://github.com/danakt/russian-words/](https://github.com/danakt/russian-words/)  
        * `"wikitionary_100K_vocab"` to automatically download a list of most common words from Project Gutenberg from [Wiktionary](https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists#Project_Gutenberg)
     
    * `dictionary_name` — name of a directory where a dictionary will be built to and loaded from, defaults to `"dictionary"` for static_dictionary
    * `raw_dictionary_path` — path to a file with a line-separated list of dictionary words, required for static_dictionary


#### Training configuration
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

Component's configuration for `spelling_error_model` also has to have as `fit_on` parameter — list of two elements:
 names of component's input and true output in chainer's shared memory.


## Language model

Provided pipelines use [KenLM](http://kheafield.com/code/kenlm/) to process language models, so if you want to build your own,
we suggest you consult its website. We do also provide our own language models for
[english](http://lnsigo.mipt.ru/export/lang_models/en_wiki_no_punkt.arpa.binary.gz) \(5.5GB\) and
[russian](http://lnsigo.mipt.ru/export/lang_models/ru_wiyalen_no_punkt.arpa.binary.gz) \(3.1GB\) languages.

## Comparison

We compared our pipelines with [Yandex.Speller](http://api.yandex.ru/speller/),
[JamSpell](https://github.com/bakwc/JamSpell) that was trained on biggest part of our Russian texts corpus that JamSpell could handle and [PyHunSpell](https://github.com/blatinier/pyhunspell)
on the [test set](http://www.dialog-21.ru/media/3838/test_sample_testset.txt)
for the [SpellRuEval competition](http://www.dialog-21.ru/en/evaluation/2016/spelling_correction/) on Automatic Spelling Correction for Russian:

| Correction method                                                                                | Precision | Recall | F-measure | Speed (sentences/s) |
|--------------------------------------------------------------------------------------------------|-----------|--------|-----------|---------------------|
| Yandex.Speller                                                                                   | 83.09     | 59.86  | 69.59     | 5.                  |
| [Damerau Levenstein 1 + lm](/deeppavlov/configs/spelling_correction/levenstein_corrector_ru.json)| 53.26     | 53.74  | 53.50     | 29.3                |
| [Brill Moore top 4 + lm](/deeppavlov/configs/spelling_correction/brillmoore_kartaslov_ru.json)   | 51.92     | 53.94  | 52.91     | 0.6                 |
| Hunspell + lm                                                                                    | 41.03     | 48.89  | 44.61     | 2.1                 |
| JamSpell                                                                                         | 44.57     | 35.69  | 39.64     | 136.2               |
| [Brill Moore top 1](/deeppavlov/configs/spelling_correction/brillmoore_kartaslov_ru_nolm.json)   | 41.29     | 37.26  | 39.17     | 2.4                 |
| Hunspell                                                                                         | 30.30     | 34.02  | 32.06     | 20.3                |
