[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/deepmipt/DeepPavlov/blob/master/LICENSE)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# <center>DeepPavlov</center>
### *We are in a really early Alpha release. You should be ready for hard adventures.*
### *If you have updated to version 0.0.2 - please re-download all pre-trained models*
An open-source conversational AI library is built on TensorFlow and Keras and designed for
 * NLP and dialog systems research
 * implementation and evaluation of complex conversational systems
 
Our goal is to provide researchers with:
 * a framework for implementing and testing their own dialog models
 * set of pre-trained NLP models, pre-defined dialog system components (ML/DL/Rule-based) and pipeline templates
 * benchmarking environment for conversational models and uniform access to relevant datasets
 
and AI-application developers with:
 * a framework for building conversational software
 * tools for application integration with adjacent infrastructure (messengers, helpdesk software etc.)

## Features


| Component | Description |
| --------- | ----------- |
| [Slot filling and NER components](deeppavlov/models/ner/README.md) | Based on neural Named Entity Recognition network and fuzzy Levenshtein search to extract normalized slot values from text. The NER component reproduces architecture from the paper [Application of a Hybrid Bi-LSTM-CRF model to the task of Russian Named Entity Recognition](https://arxiv.org/pdf/1709.09686.pdf) which is inspired by Bi-LSTM+CRF architecture from https://arxiv.org/pdf/1603.01360.pdf. |
| [Intent classification component](deeppavlov/models/classifiers/intents/README.md) | Based on shallow-and-wide Convolutional Neural Network architecture from [Kim Y. Convolutional neural networks for sentence classification – 2014](https://arxiv.org/pdf/1408.5882). The model allows multilabel classification of sentences. |
| [Automatic spelling correction component](deeppavlov/models/spellers/error_model/README.md) | Based on [An Improved Error Model for Noisy Channel Spelling Correction by Eric Brill and Robert C. Moore](http://www.aclweb.org/anthology/P00-1037) and uses statistics based error model, a static dictionary and an ARPA language model to correct spelling errors. |
| **Skill** |  |
| [Goal-oriented bot](deeppavlov/skills/go_bot/README.md) | Based on Hybrid Code Networks (HCNs) architecture from [Jason D. Williams, Kavosh Asadi, Geoffrey Zweig, Hybrid Code Networks: practical and efficient end-to-end dialog control with supervised and reinforcement learning – 2017](https://arxiv.org/abs/1702.03274). It allows to predict responses in goal-oriented dialog. The model is customizable: embeddings, slot filler and intent classifier can switched on and off on demand. |
| **Embeddings** |  |
| [Pre-trained embeddings for the Russian language](pretrained-vectors.md) | Word vectors for the Russian language trained on joint [Russian Wikipedia](https://ru.wikipedia.org/wiki/%D0%97%D0%B0%D0%B3%D0%BB%D0%B0%D0%B2%D0%BD%D0%B0%D1%8F_%D1%81%D1%82%D1%80%D0%B0%D0%BD%D0%B8%D1%86%D0%B0) and [Lenta.ru](https://lenta.ru/) corpora. |

## Basic examples

View video demo of deploy a goal-oriented bot and a slot-filling model with Telegram UI

[![Alt text for your video](https://img.youtube.com/vi/yzoiCa_sMuY/0.jpg)](https://youtu.be/yzoiCa_sMuY)
          
 * Run goal-oriented bot with Telegram interface:
 ```
 python deep.py interactbot configs/go_bot/gobot_dstc2.json -t <TELEGRAM_TOKEN>
 ```
 * Run goal-oriented bot with console interface:
 ```
 python deep.py interact configs/go_bot/gobot_dstc2.json
 ```
 * Run slot-filling model with Telegram interface:
 ```
 python deep.py interactbot configs/ner/slotfill_dstc2.json -t <TELEGRAM_TOKEN>
 ```
 * Run slot-filling model with console interface:
 ```
 python deep.py interact configs/ner/slotfill_dstc2.json
 ```
## Conceptual overview

### Principles
The library is designed according to the following principles:
 * end-to-end deep learning architecture as a long-term goal
 * hybrid ML/DL/Rule-based architecture as a current approach
 * modular dialog system architecture
 * component-based software engineering, maximization of reusability
 * easy extension and benchmarking
 * multiple solutions for one NLP task for flexible data-driven configuration

### Target Architecture
Target architecture of our library:
<p align="left">
<img src="http://lnsigo.mipt.ru/export/images/deeppavlov_architecture.png" width="50%" height="50%"/>
</p>
DeepPavlov is built on top machine learning frameworks (TensorFlow, Keras). Other external libraries can be used to build basic components.

### Key Concepts
 * `Agent` - a conversational agent communicating with users in natural language (text)
 * `Skill` - a unit of interaction that fulfills user’s needs. Typically, a user’s need is fulfilled by presenting information or completing a transaction (e.g. answer question by FAQ, booking tickets etc.); however, for some tasks success is defined as continuous engagement (e.g. chit-chat)
 * `Components` - atomic functionality blocks
   * `Rule-based Components` - cannot be trained
   * `Machine Learning Components` - can be trained only separately
   * `Deep Learning Components` - can be trained separately and in end-to-end mode being joined in chain
 * `Switcher` - mechanism which is used by agent to rank and select the final response shown to user
 * `Components Chainer` - tool for building an agent/component pipeline from heterogeneous components (rule-based/ml/dl). Allows to train and infer from pipeline as a whole.


### Contents

 * [Installation](#installation)
 * [Quick start](#quick-start)
 * [Technical overview](#technical-overview)
    * [Project modules](#project-modules)
    * [Config](#config)
    * [Training](#training)
    * [Train config](#train-config)
    * [Train parameters](#train-parameters)
    * [DatasetReader](#datasetreader)
    * [Dataset](#dataset)
    * [Inferring](#inferring)
 * [License](#license)
 * [Support and collaboration](#support-and-collaboration)
 * [The Team](#the-team)
 

## Installation
1. Create a virtual environment with `Python 3.6`
    ```
    virtualenv env
    ```
2. Activate the environment.
    ```
    source ./env/bin/activate
    ```
3. Clone the repo and `cd` to project root
   ```
   git clone https://github.com/deepmipt/DeepPavlov.git
   cd DeepPavlov
   ```
4. Install the requirements:
    ```
    python setup.py develop
    ```
5. Install `spacy` dependencies:
    ```
    python -m spacy download en
    ```

## Quick start

To use our pre-trained models, you should first download them:
```
python download.py [-all] 
```
* running this command without options will download basic examples, `[-all]` option will download **all** our pre-trained models.
* Warning! `[-all]` requires about 10 GB of free space on disk.
    
Then you can interact with the models or train them with the following command:

```
python deep.py <mode> <path_to_config>
```

* `<mode>` can be 'train', 'interact' or 'interactbot'
* `<path_to_config>` should be a path to an NLP pipeline json config

For 'interactbot' mode you should specify Telegram bot token in `-t` parameter or in `TELEGRAM_TOKEN` environment variable.


Available model configs are:

*configs/go_bot/gobot_dstc2.json*

*configs/intents/intents_dstc2.json*

*configs/ner/slotfill_dstc2.json*

*configs/error_model/brillmoore_wikitypos_en.json*

---

## Technical overview

### Project modules

<table>
<tr>
    <td><b> deeppavlov.core.commands </b></td>
    <td> basic training and inference functions  </td>
</tr>
<tr>
    <td><b> deeppavlov.core.common </b></td>
    <td> registration and classes initialization functionality, class method decorators </td>
</tr>
<tr>
    <td><b> deeppavlov.core.data </b></td>
    <td> basic <b><i>Dataset</i></b>, <b><i>DatasetReader</i></b> and <b><i>Vocab</i></b> classes </td>
</tr>
<tr>
    <td><b> deeppavlov.core.models </b></td>
    <td> abstract model classes and interfaces </td>
</tr>
<tr>
    <td><b> deeppavlov.dataset_readers </b></td>
    <td> concrete <b><i>DatasetReader</i></b> classes </td>
</tr>
<tr>
    <td><b> deeppavlov.datasets </b></td>
    <td> concrete <b><i>Dataset</i></b> classes </td>
</tr>
<tr>
    <td><b> deeppavlov.models </b></td>
    <td> concrete <b><i>Model</i></b> classes </td>
</tr>
<tr>
    <td><b> deeppavlov.skills </b></td>
    <td> <b><i>Skill</i></b> classes. Skills are dialog models.</td>
</tr>
<tr>
    <td><b> deeppavlov.vocabs </b></td>
    <td> concrete <b><i>Vocab</i></b> classes </td>
</tr>
</table>

### Config

An NLP pipeline config is a JSON file that contains one required element `chainer`:

```json
{
  "chainer": {
    "in": ["x"],
    "in_y": ["y"],
    "pipe": [
      ...
    ],
    "out": ["y_predicted"]
  }
}
```

Chainer is a core concept of DeepPavlov library: chainer builds a pipeline from heterogeneous components
(rule-based/ml/dl) and allows to train or infer from pipeline as a whole. Each component in the pipeline specifies
its inputs and outputs as arrays of names, for example: `"in": ["tokens", "features"]` and `"out": ["token_embeddings", "features_embeddings"]` and you can chain outputs of one components with inputs of other components:
```json
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
```
Each [Component](deeppavlov/core/models/component.py) in the pipeline must implement method `__call__` and has `name` parameter, which is its registered codename. It can also have any other parameters which repeat its `__init__()` method arguments.
 Default values of `__init__()` arguments will be overridden with the config values during the initialization of a class instance.
 
You can reuse components in the pipeline to process different parts of data with the help of `id` and `ref` parameters:
```json
{
  "name": "nltk_tokenizer",
  "id": "tokenizer",
  "in": ["x_lower"],
  "out": ["x_tokens"]
},
{
  "ref": "tokenizer",
  "in": ["y"],
  "out": ["y_tokens"]
},
```
 
### Training

There are two abstract classes for trainable components: **Estimator** and **NNModel**.  
[**Estimators**](deeppavlov/core/models/estimator.py) are fit once on any data with no batching or early stopping,
so it can be safely done at the time of pipeline initialization. `fit` method has to be implemented for each Estimator. An example of Estimator is [Vocab](deeppavlov/core/data/vocab.py).
[**NNModel**](deeppavlov/core/models/nn_model.py) requires more complex training. It can only be trained in a supervised mode (as opposed to **Estimator** which can be trained in both supervised and unsupervised settings). This process takes multiple epochs with periodic validation and logging.
`train_on_batch` method has to be implemented for each NNModel.

Training is triggered by `deeppavlov.core.commands.train.train_model_from_config()` function.

### Train config

Estimators that are trained should also have `fit_on` parameter which contains a list of input parameter names.
An NNModel should have the `in_y` parameter which contains a list of ground truth answer names. For example:

```json
[
  {
    "id": "classes_vocab",
    "name": "default_vocab",
    "fit_on": ["y"],
    "level": "token",
    "save_path": "vocabs/classes.dict",
    "load_path": "vocabs/classes.dict"
  },
  {
    "in": ["x"],
    "in_y": ["y"],
    "out": ["y_predicted"],
    "name": "intent_model",
    "save_path": "intents/intent_cnn",
    "load_path": "intents/intent_cnn",
    "classes_vocab": {
      "ref": "classes_vocab"
    }
  }
]
```

The config for training the pipeline should have three additional elements: `dataset_reader`, `dataset` and `train`:

```json
{
  "dataset_reader": {
    "name": ...,
    ...
  }
  "dataset": {
    "name": ...,
    ...
  },
  "chainer": {
    ...
  }
  "train": {
    ...
  }
}
```

### Train Parameters
* `epochs` — maximum number of epochs to train NNModel, defaults to `-1` (infinite)
* `batch_size`,
* `metrics` — list of names of [registered metrics](deeppavlov/metrics) to evaluate the model. The first metric in the list
is used for early stopping
* `metric_optimization` — `maximize` or `minimize` a metric, defaults to `maximize`
* `validation_patience` — how many times in a row the validation metric has to not improve for early stopping, defaults to `5`
* `val_every_n_epochs` — how often to validate the pipe, defaults to `-1` (never)
* `log_every_n_batches`, `log_every_n_epochs` — how often to calculate metrics for train data, defaults to `-1` (never)
* `validate_best`, `test_best` flags to infer the best saved model on valid and test data, defaults to `true`

### DatasetReader

`DatasetReader` class reads data and returns it in a specified format.
A concrete `DatasetReader` class should be inherited from the base
`deeppavlov.data.dataset_reader.DatasetReader` class and registered with a codename:

```python
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader

@register('dstc2_datasetreader')
class DSTC2DatasetReader(DatasetReader):
```

### Dataset

`Dataset` forms the sets of data ('train', 'valid', 'test') needed for training/inference and divides it into batches.
A concrete `Dataset` class should be registered and can be inherited from
`deeppavlov.data.dataset_reader.Dataset` class. `deeppavlov.data.dataset_reader.Dataset`
is not an abstract class and can be used as a `Dataset` as well.

### Inference

All components inherited from `deeppavlov.core.models.component.Component` abstract class can be used for inference. The `__call__()` method should return standard output of a component. For example, a *tokenizer* should return
*tokens*, a *NER recognizer* should return *recognized entities*, a *bot* should return an *utterance*.
A particular format of returned data should be defined in `__call__()`.

Inference is triggered by `deeppavlov.core.commands.infer.interact_model()` function. There is no need in a separate JSON for inference. 

## License

DeepPavlov is Apache 2.0 - licensed.

## Support and collaboration

If you have any questions, bug reports or feature requests, please feel free to post on our [Github Issues](https://github.com/deepmipt/DeepPavlov/issues) page. Please tag your issue with 'bug', 'feature request', or 'question'.  Also we’ll be glad to see your pull requests to add new datasets, models, embeddings, etc.

## The Team

DeepPavlov is built and maintained by [Neural Networks and Deep Learning Lab](https://mipt.ru/english/research/labs/neural-networks-and-deep-learning-lab) at [MIPT](https://mipt.ru/english/) within [iPavlov](http://ipavlov.ai/) project (part of [National Technology Initiative](https://asi.ru/eng/nti/)) and in partnership with [Sberbank](http://www.sberbank.com/).

<p align="center">
<img src="http://ipavlov.ai/img/ipavlov_footer.png" width="50%" height="50%"/>
</p>

