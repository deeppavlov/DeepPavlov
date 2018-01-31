[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
<div style="text-align: justify">

# DeepPavlov (pre alfa)
An Apache 2.0 NLP research library, built on TensorFlow and Keras, for 
 * Building complicated NLP pipelines
 * Training and infering NLP algorithms
 
## Features
 * Goal-oriented dialog agent
 * Slot filling component
 * Intent classification component
 * Automatic spelling correction component
 * Pretrained embeddings library

### Contents

 * [Installation](#installation)
 * [Quick start](#quick-start)
 * [Support](#support)
 * [The Team](#the-team)
 * [License](#license)
 * [Deeppavlov overview](#deeppavlov-overview)
    * [Project modules](#project-modules)
    * [Config](#config)
    * [DatasetReader](#datasetreader)
    * [Dataset](#dataset)
    * [Vocab](#vocab)
    * [Model](#model)
    * [Training](#training)
    * [Inferring](#inferring)
 * [Suggested models](#suggested-models)

## Installation
1. Create a virtual environment with `Python 3.6`
2. Activate the environment.
3. `cd` to the project root.
4. Install the requirements:
    ```
    python install.py
    ```
5. Install `spacy` requirements:

    ```
    python -m spacy download en
    ```

## Quick start
With the purpose to interact with our pretrained models, they should be downloaded first:

```
cd deeppavlov/
python download.py [-all]
```
* `[-all]` option is not required for basic examples; it will download **all** our pretrained models.

* Warning! `[-all]` requires about 10 GB of free space on disk.

Then the models can be interacted or trained with the following command:

```
python deep.py <mode> <path_to_config>
```

* `<mode>` can be 'train', 'interact' or 'interactbot'
* `<path_to_config>` should be a path to an NLP pipeline json config

For 'interactbot' mode you should specify Telegram bot token in `-t` parameter or in `TELEGRAM_TOKEN` environment variable.


Available model configs are:

*skills/go_bot/config.json*

*models/classifiers/intents/config_dstc2.json*

*models/ner/config.json*

*models/spellers/error_model/config_en.json*

## Support

If you have any questions, bug reports or feature requests, please don't hesitate to post on our Github Issues page.

## The Team

DeepPavlov is currently maintained by ...

## License

DeepPavlov is Apache 2.0 - licensed.

---

## DeepPavlov overview

### Project modules

<table>
<tr>
    <td><b> deeppavlov.core.commands </b></td>
    <td> basic training and inferring functions  </td>
</tr>
<tr>
    <td><b> deeppavlov.core.common </b></td>
    <td> registration and classes initialization functionality, class method decorators </td>
</tr>
<tr>
    <td><b> deeppavlov.core.data </b></td>
    <td> basic <font color=#ff6666><i><b>Dataset</b></i></font>,
     <font color=#ff6666><i><b>DatasetReader</b></i></font> and <font color=#ff6666><i><b>Vocab</b></i></font> classes </td>
</tr>
<tr>
    <td><b> deeppavlov.core.models </b></td>
    <td> abstract model classes and interfaces </td>
</tr>
<tr>
    <td><b> deeppavlov.dataset_readers </b></td>
    <td> concrete <font color=#ff6666><i><b>DatasetReader</b></i></font> classes </td>
</tr>
<tr>
    <td><b> deeppavlov.datasets </b></td>
    <td> concrete <font color=#ff6666><i><b>Dataset</b></i></font> classes </td>
</tr>
<tr>
    <td><b> deeppavlov.models </b></td>
    <td> concrete <font color=#ff6666><i><b>Model</b></i></font> classes </td>
</tr>
<tr>
    <td><b> deeppavlov.skills </b></td>
    <td> <font color=#ff6666><i><b>Skill</b></i></font> classes. Skills are dialog models.</td>
</tr>
<tr>
    <td><b> deeppavlov.vocabs </b></td>
    <td> concrete <font color=#ff6666><i><b>Vocab</b></i></font> classes </td>
</tr>
</table>

### Config

An NLP pipeline config is a JSON file, which consists of four required elements:

```javascript
{
  "dataset_reader": {
  },
  "dataset": {
  },
  "vocabs": {
  },
  "model": {
  }
}
```

Each class in the config has `name` parameter, which is its registered codename
 and can have any other parameters, repeating its `__init__()` method arguments.
 Default values of `__init__()` arguments will be overriden with the config values
 during class instance initialization.

### DatasetReader

`DatasetReader` class reads data and returns it in a specified format.
A concrete `DatasetReader` class should be inherited from base
`deeppavlov.data.dataset_reader.DatasetReader` class and registered with a codename:

```python
@register('dstc2_datasetreader')
class DSTC2DatasetReader(DatasetReader):
```

### Dataset

`Dataset` forms needed sets of data ('train', 'valid', 'test') and forms data batches.
A concrete `Dataset` class should be registered and can be inherited from
`deeppavlov.data.dataset_reader.Dataset` class. `deeppavlov.data.dataset_reader.Dataset`
is not an abstract class and can be used as `Dataset` as well.

### Vocab

`Vocab` is a trainable class, which forms and serialize vocabs. Vocabs index any data.
For example, tokens to indices and backwards, chars to indices, classes to indices, etc.
It can index X (features) and y (answers) types of data. A concrete `Vocab` class
should be registered and can be inherited from `deeppavlov.data.vocab.DefaultVocabulary` class.
`deeppavlov.data.vocab.DefaultVocabulary` is not an abstrat class and can be used as `Vocab` as well.

### Model

`Model` is the main class which rules the training/infering process and feature generation.
If a model requires other models to produce features, they need to be passed in its constructor
and config. All models can be nested as much as needed. For example, a skeleton of
`deeppavlov.skills.go_bot.go_bot.GoalOrientedBot` consists of 11 separate model classes,
3 of which are neural networks:

```javascript
{
  "model": {
    "name": "go_bot",
    "network": {
      "name": "go_bot_rnn"
    },
    "slot_filler": {
      "name": "dstc_slotfilling",
      "ner_network": {
         "name": "ner_tagging_network",
      }
    },
    "intent_classifier": {
      "name": "intent_model",
      "embedder": {
        "name": "fasttext"
      },
      "tokenizer": {
        "name": "nltk_tokenizer"
      }
    },
    "embedder": {
      "name": "fasttext"
    },
    "bow_encoder": {
      "name": "bow"
    },
    "tokenizer": {
      "name": "spacy_tokenizer"
    },
    "tracker": {
      "name": "featurized_tracker"
    }
  }
}
```

All models should be registered and inherited from `deeppavlov.core.models.inferable.Inferable`
or from both `Inferable` and `deeppavlov.core.models.trainable.Trainable` interfaces.
Models inherited from `Trainable` interface can be trained. Models inherited from `Inferable`
interface can be only inferred. Usually `Inferable` models are rule-based models or
pretraned models that we import from third-party libraries (like `NLTK`, `Spacy`, etc.).

### Training

All models inherited from `deeppavlov.core.models.trainable.Trainable` interface can be trained.
The training process should be described in `train()` method:

 ```python
 @register("my_model")
 class MyModel(Inferable, Trainable):

    def train(*args, **kwargs):
        """
        Implement training here.
        """
 ```

All parameters for training which can be changed during experiments (like *num of epochs*,
*batch size*, *patience*, *learning rate*, *optimizer*) should be passed to a model's
`__init__()`. The default parameters values from `__init__()` are overridden with JSON config values.
To change these values, there is no need to rewrite the code, only the config should be changed.

The training process is managed by `train_now` attribute. If `train_now` is *True*,
a model is being trained. This parameter is useful when using `Vocab`, because in a single
model run some vocabs can be trained, while some only inferred by other models in pipeline.
The training parameters in JSON config can look like this:

```javascript
{
  "model": {
    "name": "my_model",
    "train_now": true,
    "optimizer": "Adam",
    "learning_rate": 0.2,
    "num_epochs": 1000
  }
}
```

Training is triggered by `deeppavlov.core.commands.train.train_model_from_config()` function.

### Inferring

All models inherited from `deeppavlov.core.models.inferable.Inferable` interface can be inferred.
The `infer()` method should return what a model can do. For example, a *tokenizer* should return
*tokens*, a *NER recognizer* should return *recognized entities*, a *bot* should return a *replica*.
A particular format of returned data should be defined in `infer()`.

Inferring is triggered by `deeppavlov.core.commands.train.infer_model_from_config()` function.
There is no need in s separate JSON for inferring. `train_now` parameter is ignored during
inferring.

## Suggested models
</div>