[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/deepmipt/DeepPavlov/blob/master/LICENSE)
![Python 3.6, 3.7](https://img.shields.io/badge/python-3.6%20%7C%203.7-green.svg)
[![Downloads](https://pepy.tech/badge/deeppavlov)](https://pepy.tech/project/deeppavlov)
<img align="right" height="27%" width="27%" src="docs/_static/deeppavlov_logo.png"/>

DeepPavlov is an open-source conversational AI library built on [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/)
and [PyTorch](https://pytorch.org/).

DeepPavlov is designed for
* development of production ready chat-bots and complex conversational systems,
* research in the area of NLP and, particularly, of dialog systems.

## Quick Links

* Demo [*demo.deeppavlov.ai*](https://demo.deeppavlov.ai/)
* Documentation [*docs.deeppavlov.ai*](http://docs.deeppavlov.ai/)
    * Model List [*docs:features/*](http://docs.deeppavlov.ai/en/master/features/overview.html)
    * Contribution Guide [*docs:contribution_guide/*](http://docs.deeppavlov.ai/en/master/devguides/contribution_guide.html)
* Issues [*github/issues/*](https://github.com/deepmipt/DeepPavlov/issues)
* Forum [*forum.deeppavlov.ai*](https://forum.deeppavlov.ai/)
* Blogs [*medium.com/deeppavlov*](https://medium.com/deeppavlov)
* Tutorials [*examples/*](https://github.com/deepmipt/DeepPavlov/tree/master/examples) and [extended colab tutorials](https://github.com/deepmipt/dp_tutorials)
* Docker Hub [*hub.docker.com/u/deeppavlov/*](https://hub.docker.com/u/deeppavlov/) 
    * Docker Images Documentation [*docs:docker-images/*](http://docs.deeppavlov.ai/en/master/intro/installation.html#docker-images)

Please leave us [your feedback](https://forms.gle/i64fowQmiVhMMC7f9) on how we can improve the DeepPavlov framework.

**Models**

[Named Entity Recognition](http://docs.deeppavlov.ai/en/master/features/models/ner.html) | [Slot filling](http://docs.deeppavlov.ai/en/master/features/models/slot_filling.html)

[Intent/Sentence Classification](http://docs.deeppavlov.ai/en/master/features/models/classifiers.html) |  [Question Answering over Text (SQuAD)](http://docs.deeppavlov.ai/en/master/features/models/squad.html) 

[Knowledge Base Question Answering](http://docs.deeppavlov.ai/en/master/features/models/kbqa.html)

[Sentence Similarity/Ranking](http://docs.deeppavlov.ai/en/master/features/models/neural_ranking.html) | [TF-IDF Ranking](http://docs.deeppavlov.ai/en/master/features/models/tfidf_ranking.html) 

[Morphological tagging](http://docs.deeppavlov.ai/en/master/features/models/morphotagger.html) | [Syntactic parsing](http://docs.deeppavlov.ai/en/master/features/models/syntaxparser.html)

[Automatic Spelling Correction](http://docs.deeppavlov.ai/en/master/features/models/spelling_correction.html) | [ELMo training and fine-tuning](http://docs.deeppavlov.ai/en/master/apiref/models/elmo.html)

[Speech recognition and synthesis (ASR and TTS)](http://docs.deeppavlov.ai/en/master/features/models/nemo.html) based on [NVIDIA NeMo](https://nvidia.github.io/NeMo/index.html)

[Entity Linking](http://docs.deeppavlov.ai/en/master/features/models/entity_linking.html) | [Multitask BERT](http://docs.deeppavlov.ai/en/master/features/models/multitask_bert.html)

**Skills**

[Goal(Task)-oriented Bot](http://docs.deeppavlov.ai/en/master/features/skills/go_bot.html) | [Open Domain Questions Answering](http://docs.deeppavlov.ai/en/master/features/skills/odqa.html)

[Frequently Asked Questions Answering](http://docs.deeppavlov.ai/en/master/features/skills/faq.html)

**Embeddings**

[BERT embeddings for the Russian, Polish, Bulgarian, Czech, and informal English](http://docs.deeppavlov.ai/en/master/features/pretrained_vectors.html#bert)

[ELMo embeddings for the Russian language](http://docs.deeppavlov.ai/en/master/features/pretrained_vectors.html#elmo)

[FastText embeddings for the Russian language](http://docs.deeppavlov.ai/en/master/features/pretrained_vectors.html#fasttext)

**Auto ML**

[Tuning Models](http://docs.deeppavlov.ai/en/master/features/hypersearch.html)

**Integrations**

[REST API](http://docs.deeppavlov.ai/en/master/integrations/rest_api.html) | [Socket API](http://docs.deeppavlov.ai/en/master/integrations/socket_api.html) | [Yandex Alice](http://docs.deeppavlov.ai/en/master/integrations/yandex_alice.html)

[Telegram](http://docs.deeppavlov.ai/en/master/integrations/telegram.html) | [Microsoft Bot Framework](http://docs.deeppavlov.ai/en/master/integrations/ms_bot.html)

[Amazon Alexa](http://docs.deeppavlov.ai/en/master/integrations/amazon_alexa.html) | [Amazon AWS](http://docs.deeppavlov.ai/en/master/integrations/aws_ec2.html)

## Installation

0. We support `Linux` and `Windows` platforms, `Python 3.6` and `Python 3.7`
    * **`Python 3.5` is not supported!**
    * **installation for `Windows` requires `Git`(for example, [git](https://git-scm.com/download/win)) and  `Visual Studio 2015/2017` with `C++` build tools installed!**

1. Create and activate a virtual environment:
    * `Linux`
    ```
    python -m venv env
    source ./env/bin/activate
    ```
    * `Windows`
    ```
    python -m venv env
    .\env\Scripts\activate.bat
    ```
2. Install the package inside the environment:
    ```
    pip install deeppavlov
    ```

## QuickStart

There is a bunch of great pre-trained NLP models in DeepPavlov. Each model is
determined by its config file.

List of models is available on
[the doc page](http://docs.deeppavlov.ai/en/master/features/overview.html) in
the `deeppavlov.configs` (Python):

```python
from deeppavlov import configs
```

When you're decided on the model (+ config file), there are two ways to train,
evaluate and infer it:

* via [Command line interface (CLI)](#command-line-interface-cli) and
* via [Python](#python).

#### GPU requirements

To run supported DeepPavlov models on GPU you should have [CUDA](https://developer.nvidia.com/cuda-toolkit) 10.0
installed on your host machine and TensorFlow with GPU support (`tensorflow-gpu`)
installed in your python environment. Current supported TensorFlow version is 1.15.2.
Run

```
pip install tensorflow-gpu==1.15.2
```

before installing model's package requirements to install supported `tensorflow-gpu` version.


Before making choice of an interface, install model's package requirements
(CLI):

```bash
python -m deeppavlov install <config_path>
```

* where `<config_path>` is path to the chosen model's config file (e.g.
  `deeppavlov/configs/ner/slotfill_dstc2.json`) or just name without
  *.json* extension (e.g. `slotfill_dstc2`)


### Command line interface (CLI)

To get predictions from a model interactively through CLI, run

```bash
python -m deeppavlov interact <config_path> [-d]
```

* `-d` downloads required data -- pretrained model files and embeddings
  (optional).

You can train it in the same simple way:

```bash
python -m deeppavlov train <config_path> [-d]
```

Dataset will be downloaded regardless of whether there was `-d` flag or not.

To train on your own data you need to modify dataset reader path in the
[train config doc](http://docs.deeppavlov.ai/en/master/intro/config_description.html#train-config).
The data format is specified in the corresponding model doc page. 

There are even more actions you can perform with configs:

```bash
python -m deeppavlov <action> <config_path> [-d]
```

* `<action>` can be
    * `download` to download model's data (same as `-d`),
    * `train` to train the model on the data specified in the config file,
    * `evaluate` to calculate metrics on the same dataset,
    * `interact` to interact via CLI,
    * `riseapi` to run a REST API server (see
    [doc](http://docs.deeppavlov.ai/en/master/integrations/rest_api.html)),
    * `telegram` to run as a Telegram bot (see
    [doc](http://docs.deeppavlov.ai/en/master/integrations/telegram.html)),
    * `msbot` to run a Miscrosoft Bot Framework server (see
    [doc](http://docs.deeppavlov.ai/en/master/integrations/ms_bot.html)),
    * `predict` to get prediction for samples from *stdin* or from
      *<file_path>* if `-f <file_path>` is specified.
* `<config_path>` specifies path (or name) of model's config file
* `-d` downloads required data


### Python

To get predictions from a model interactively through Python, run

```python
from deeppavlov import build_model

model = build_model(<config_path>, download=True)

# get predictions for 'input_text1', 'input_text2'
model(['input_text1', 'input_text2'])
```

* where `download=True` downloads required data from web -- pretrained model
  files and embeddings (optional),
* `<config_path>` is path to the chosen model's config file (e.g.
  `"deeppavlov/configs/ner/ner_ontonotes_bert_mult.json"`) or
  `deeppavlov.configs` attribute (e.g.
  `deeppavlov.configs.ner.ner_ontonotes_bert_mult` without quotation marks).

You can train it in the same simple way:

```python
from deeppavlov import train_model 

model = train_model(<config_path>, download=True)
```

* `download=True` downloads pretrained model, therefore the pretrained
model will be, first, loaded and then train (optional).

Dataset will be downloaded regardless of whether there was ``-d`` flag or
not.

To train on your own data you need to modify dataset reader path in the
[train config doc](http://docs.deeppavlov.ai/en/master/intro/config_description.html#train-config).
The data format is specified in the corresponding model doc page. 

You can also calculate metrics on the dataset specified in your config file:

```python
from deeppavlov import evaluate_model 

model = evaluate_model(<config_path>, download=True)
```

There are also available integrations with various messengers, see
[Telegram Bot doc page](http://docs.deeppavlov.ai/en/master/integrations/telegram.html)
and others in the Integrations section for more info.


## Breaking Changes

**Breaking changes in version 0.15.0**
- [bert_as_summarizer](https://github.com/deepmipt/DeepPavlov/pull/1391), [seq2seq_go_bot](https://github.com/deepmipt/DeepPavlov/pull/1434) and all deeppavlov.deprecated components were removed
- hyperparameter optimization by neural evolution was [removed](https://github.com/deepmipt/DeepPavlov/pull/1436)

**Breaking changes in version 0.7.0**
- in dialog logger config file [dialog_logger_config.json](deeppavlov/utils/settings/dialog_logger_config.json) `agent_name` parameter was renamed to `logger_name`,
  the default value was changed
- Agent, Skill, eCommerce Bot and Pattern Matching classes were moved to [deeppavlov.deprecated](deeppavlov/deprecated)
- [AIML Skill](http://docs.deeppavlov.ai/en/0.7.0/features/skills/aiml_skill.html),
  [RASA Skill](http://docs.deeppavlov.ai/en/0.7.0/features/skills/rasa_skill.html),
  [Yandex Alice](http://docs.deeppavlov.ai/en/0.7.0/integrations/yandex_alice.html),
  [Amazon Alexa](http://docs.deeppavlov.ai/en/0.7.0/integrations/amazon_alexa.html),
  [Microsoft Bot Framework](http://docs.deeppavlov.ai/en/0.7.0/integrations/ms_bot.html) and
  [Telegram integration](http://docs.deeppavlov.ai/en/0.7.0/integrations/telegram.html) interfaces were changed
- `/start` and `/help` Telegram messages were moved from `models_info.json` to [server_config.json](deeppavlov/utils/settings/server_config.json)
- [risesocket](http://docs.deeppavlov.ai/en/0.7.0/integrations/socket_api.html) request and response format was changed
- [riseapi](http://docs.deeppavlov.ai/en/0.7.0/integrations/rest_api.html#advanced-configuration) and
  [risesocket](http://docs.deeppavlov.ai/en/0.7.0/integrations/socket_api.html#advanced-configuration) model-specific
  properties parametrization was changed

**Breaking changes in version 0.6.0**
- [REST API](http://docs.deeppavlov.ai/en/0.6.0/integrations/rest_api.html):
  - all models default endpoints were renamed to `/model`
  - by default model arguments names are taken from `chainer.in`
    [configuration parameter](http://docs.deeppavlov.ai/en/0.6.0/intro/configuration.html) instead of pre-set names
    from a [settings file](http://docs.deeppavlov.ai/en/0.6.0/integrations/settings.html)
  - swagger api endpoint moved from `/apidocs` to `/docs`
- when using `"max_proba": true` in
  a [`proba2labels` component](http://docs.deeppavlov.ai/en/0.6.0/apiref/models/classifiers.html) for classification,
  it will return single label for every batch element instead of a list. One can set `"top_n": 1`
  to get batches of single item lists as before

**Breaking changes in version 0.5.0**
- dependencies have to be reinstalled for most pipeline configurations
- models depending on `tensorflow` require `CUDA 10.0` to run on GPU instead of `CUDA 9.0`
- scikit-learn models have to be redownloaded or retrained

**Breaking changes in version 0.4.0!**
- default target variable name for [neural evolution](https://docs.deeppavlov.ai/en/0.4.0/intro/hypersearch.html#parameters-evolution-for-deeppavlov-models)
was changed from `MODELS_PATH` to `MODEL_PATH`.

**Breaking changes in version 0.3.0!**
- component option `fit_on_batch` in configuration files was removed and replaced with adaptive usage of the `fit_on` parameter.

**Breaking changes in version 0.2.0!**
- `utils` module was moved from repository root in to `deeppavlov` module
- `ms_bot_framework_utils`,`server_utils`, `telegram utils` modules was renamed to `ms_bot_framework`, `server` and `telegram` correspondingly
- rename metric functions `exact_match` to `squad_v2_em` and  `squad_f1` to `squad_v2_f1`
- replace dashes in configs name with underscores

**Breaking changes in version 0.1.0!**
- As of `version 0.1.0` all models, embeddings and other downloaded data for provided configurations are
 by default downloaded to the `.deeppavlov` directory in current user's home directory.
 This can be changed on per-model basis by modifying
 a `ROOT_PATH` [variable](http://docs.deeppavlov.ai/en/master/intro/configuration.html#variables)
 or related fields one by one in model's configuration file.
 
- In configuration files, for all features/models, dataset readers and iterators `"name"` and `"class"` fields are combined
into the `"class_name"` field.

- `deeppavlov.core.commands.infer.build_model_from_config()` was renamed to `build_model` and can be imported from the
 `deeppavlov` module directly.

- The way arguments are passed to metrics functions during training and evaluation was changed and
 [documented](http://docs.deeppavlov.ai/en/0.4.0/intro/config_description.html#metrics).

## License

DeepPavlov is Apache 2.0 - licensed.

## The Team

DeepPavlov is built and maintained by [Neural Networks and Deep Learning Lab](https://www.facebook.com/deepmipt/)
at [MIPT](https://mipt.ru/english/).

<p align="center">
<img src="docs/_static/ipavlov_footer.png" width="50%" height="50%"/>
</p>
