[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/deepmipt/DeepPavlov/blob/master/LICENSE)
![Python 3.6, 3.7](https://img.shields.io/badge/python-3.6%20%7C%203.7-green.svg)
[![Downloads](https://pepy.tech/badge/deeppavlov)](https://pepy.tech/project/deeppavlov)
<img align="right" height="27%" width="27%" src="docs/_static/ipavlov_logo.png"/>

DeepPavlov is an open-source conversational AI library built on [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/).

DeepPavlov is designed for
 * development of production ready chat-bots and complex conversational systems,
 * research in the area of NLP and, particularly, of dialog systems.

## Quick Links

* Demo [*demo.ipavlov.ai*](https://demo.ipavlov.ai/)
* Documentation [*docs.deeppavlov.ai*](http://docs.deeppavlov.ai/)
    * Feature List [*docs:features/*](http://docs.deeppavlov.ai/en/master/intro/features.html)
    * Contribution Guide [*docs:contribution_guide/*](http://docs.deeppavlov.ai/en/master/devguides/contribution_guide.html)
* Issues [*github/issues/*](https://github.com/deepmipt/DeepPavlov/issues)
* Forum [*forum.ipavlov.ai*](https://forum.ipavlov.ai/)
* Blogs [*ipavlov.ai/#rec108281800*](http://ipavlov.ai/#rec108281800)
* Tutorials [*examples/*](https://github.com/deepmipt/DeepPavlov/tree/master/examples)
* Docker Hub [*hub.docker.com/u/deeppavlov/*](https://hub.docker.com/u/deeppavlov/) 
    * Docker Images Documentation [*docs:docker-images/*](http://docs.deeppavlov.ai/en/master/intro/installation.html#docker-images)

## Key Features

**Models**

[Named Entity Recognition](http://docs.deeppavlov.ai/en/latest/components/ner.html) | [Slot filling](http://docs.deeppavlov.ai/en/latest/components/slot_filling.html)

[Intent/Sentence Classification](http://docs.deeppavlov.ai/en/latest/components/classifiers.html) |  [Question Answering over Text (SQuAD)](http://docs.deeppavlov.ai/en/latest/components/squad.html) 

[Sentence Similarity/Ranking](http://docs.deeppavlov.ai/en/latest/components/neural_ranking.html) | [TF-IDF Ranking](http://docs.deeppavlov.ai/en/latest/components/tfidf_ranking.html) 

[Morphological tagging](http://docs.deeppavlov.ai/en/latest/components/morphotagger.html) | [Automatic Spelling Correction](http://docs.deeppavlov.ai/en/latest/components/spelling_correction.html)

[ELMo training and fine-tuning](http://docs.deeppavlov.ai/en/latest/apiref/models/elmo.html)


**Skills**

[Goal(Task)-oriented Bot](http://docs.deeppavlov.ai/en/latest/skills/go_bot.html) | [Seq2seq Goal-Oriented bot](http://docs.deeppavlov.ai/en/latest/skills/seq2seq_go_bot.html)

[Open Domain Questions Answering](http://docs.deeppavlov.ai/en/latest/skills/odqa.html) | [eCommerce Bot](http://docs.deeppavlov.ai/en/master/skills/ecommerce.html) 

[Frequently Asked Questions Answering](http://docs.deeppavlov.ai/en/latest/skills/faq.html) | [Pattern Matching](http://docs.deeppavlov.ai/en/latest/skills/pattern_matching.html) 

**Embeddings**

[ELMo embeddings for the Russian language](http://docs.deeppavlov.ai/en/latest/apiref/models/embedders.html#deeppavlov.models.embedders.elmo_embedder.ELMoEmbedder)

[FastText embeddings for the Russian language](http://docs.deeppavlov.ai/en/latest/intro/pretrained_vectors.html)

**Auto ML**

[Tuning Models with Evolutionary Algorithm](http://docs.deeppavlov.ai/en/latest/intro/hypersearch.html)

## QuickStart

To use our pre-trained models, you should first install their requirements:
```
python -m deeppavlov install <path_to_config>
```
  
Then download the models and data for them:
```
python -m deeppavlov download <path_to_config>
```
or you can use additional key `-d` to automatically download all required models and data with any command like `interact`, `riseapi`, etc.

Then you can interact with the models or train them with the following command:

```
python -m deeppavlov <mode> <path_to_config> [-d]
```

* `<mode>` can be `train`, `predict`, `interact`, `interactbot`, `interactmsbot` or `riseapi`
* `<path_to_config>` should be a path to an NLP pipeline json config (e.g. `deeppavlov/configs/ner/slotfill_dstc2.json`)
or a name without the `.json` extension of one of the config files [provided](deeppavlov/configs) in this repository (e.g. `slotfill_dstc2`)

You can also serve DeepPavlov models for:
* Microsoft Bot Framework ([see developer guide for the detailed instructions](http://docs.deeppavlov.ai/en/latest/devguides/ms_bot_integration.html)) 
* Amazon Alexa ([see developer guide for the detailed instructions](http://docs.deeppavlov.ai/en/latest/devguides/amazon_alexa.html)) 

For `riseapi` mode you should specify api settings (host, port, etc.) in [*deeppavlov/utils/settings/server_config.json*](deeppavlov/utils/settings/server_config.json) configuration file. If provided, values from *model_defaults* section override values for the same parameters from *common_defaults* section. Model names in *model_defaults* section should be similar to the class names of the models main component.
Here is [detailed info on the DeepPavlov REST API](http://docs.deeppavlov.ai/en/latest/devguides/rest_api.html)

All DeepPavlov settings files are stored in `deeppavlov/utils/settings` by default. You can get full path to it with `python -m deeppavlov.settings settings`. Also you can move it with with `python -m deeppavlov.settings settings -p <new/configs/dir/path>` (all your configuration settings will be preserved) or move it to default location with `python -m deeppavlov.settings settings -d` (all your configuration settings will be RESET to default ones).

For `predict` you can specify path to input file with `-f` or `--input-file` parameter, otherwise, data will be taken
from stdin.  
Every line of input text will be used as a pipeline input parameter, so one example will consist of as many lines,
as many input parameters your pipeline expects.  
You can also specify batch size with `-b` or `--batch-size` parameter.

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
## Breaking Changes

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
 a `ROOT_PATH` [variable](http://docs.deeppavlov.ai/en/latest/intro/config_description.html#variables)
 or related fields one by one in model's configuration file.
 
- In configuration files, for all components, dataset readers and iterators `"name"` and `"class"` fields are combined
into the `"class_name"` field.

- `deeppavlov.core.commands.infer.build_model_from_config()` was renamed to `build_model` and can be imported from the
 `deeppavlov` module directly.

- The way arguments are passed to metrics functions during training and evaluation was changed and
 [documented](http://docs.deeppavlov.ai/en/latest/intro/config_description.html#metrics).

## License

DeepPavlov is Apache 2.0 - licensed.

## The Team

DeepPavlov is built and maintained by [Neural Networks and Deep Learning Lab](https://mipt.ru/english/research/labs/neural-networks-and-deep-learning-lab) at [MIPT](https://mipt.ru/english/) within [iPavlov](http://ipavlov.ai/) project (part of [National Technology Initiative](https://asi.ru/eng/nti/)) and in partnership with [Sberbank](http://www.sberbank.com/).

<p align="center">
<img src="docs/_static/ipavlov_footer.png" width="50%" height="50%"/>
</p>
