# DeepPavlov 1.0

[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python 3.6, 3.7, 3.8, 3.9, 3.10, 3.11](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-green.svg)
[![Downloads](https://pepy.tech/badge/deeppavlov)](https://pepy.tech/project/deeppavlov)
[![Static Badge](https://img.shields.io/badge/DeepPavlov%20Community-blue)](https://forum.deeppavlov.ai/)
[![Static Badge](https://img.shields.io/badge/DeepPavlov%20Demo-blue)](https://demo.deeppavlov.ai/)


DeepPavlov 1.0 is an open-source NLP framework built on [PyTorch](https://pytorch.org/) and [transformers](https://github.com/huggingface/transformers). DeepPavlov 1.0 is created for modular and configuration-driven development of state-of-the-art NLP models and supports a wide range of NLP model applications. DeepPavlov 1.0 is designed for practitioners with limited knowledge of NLP/ML.

## Quick Links

|name|Description|
|--|--|
| ⭐️ [*Demo*](https://demo.deeppavlov.ai/)|Check out our NLP models in the online demo|
| 📚 [*Documentation*](http://docs.deeppavlov.ai/)|How to use DeepPavlov 1.0 and its features|
| 🚀 [*Model List*](http://docs.deeppavlov.ai/en/master/features/overview.html)|Find the NLP model you need in the list of available models|
| 🪐 [*Contribution Guide*](http://docs.deeppavlov.ai/en/master/devguides/contribution_guide.html)|Please read the contribution guidelines before making a contribution|
| 🎛 [*Issues*](https://github.com/deeppavlov/DeepPavlov/issues)|If you have an issue with DeepPavlov, please let us know|
| ⏩ [*Forum*](https://forum.deeppavlov.ai/)|Please let us know if you have a problem with DeepPavlov|
| 📦 [*Blogs*](https://medium.com/deeppavlov)|Read about our current development|
| 🦙 [Extended colab tutorials](https://github.com/deeppavlov/dp_tutorials)|Check out the code tutorials for our models|
| 🌌 [*Docker Hub*](https://hub.docker.com/u/deeppavlov/)|Check out the Docker images for rapid deployment|
| 👩‍🏫 [*Feedback*](https://forms.gle/i64fowQmiVhMMC7f9)|Please leave us your feedback to make DeepPavlov better|


## Installation

0. DeepPavlov supports `Linux`, `Windows 10+` (through WSL/WSL2), `MacOS` (Big Sur+) platforms, `Python 3.6`, `3.7`, `3.8`, `3.9` and `3.10`.
    Depending on the model used, you may need from 4 to 16 GB RAM.

1. Create and activate a virtual environment:
    * `Linux`

    ```
    python -m venv env
    source ./env/bin/activate
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

By default, DeepPavlov installs models requirements from PyPI. PyTorch from PyPI could not support your device CUDA
capability. To run supported DeepPavlov models on GPU you should have [CUDA](https://developer.nvidia.com/cuda-toolkit)
compatible with used GPU and [PyTorch version](deeppavlov/requirements/pytorch.txt) required by DeepPavlov models.
See [docs](https://docs.deeppavlov.ai/en/master/intro/quick_start.html#using-gpu) for details.
GPU with Pascal or newer architecture and 4+ GB VRAM is recommended.

### Command line interface (CLI)

To get predictions from a model interactively through CLI, run

```bash
python -m deeppavlov interact <config_path> [-d] [-i]
```

* `-d` downloads required data - pretrained model files and embeddings (optional).
* `-i` installs model requirements (optional).

You can train it in the same simple way:

```bash
python -m deeppavlov train <config_path> [-d] [-i]
```

Dataset will be downloaded regardless of whether there was `-d` flag or not.

To train on your own data you need to modify dataset reader path in the
[train config doc](http://docs.deeppavlov.ai/en/master/intro/config_description.html#train-config).
The data format is specified in the corresponding model doc page.

There are even more actions you can perform with configs:

```bash
python -m deeppavlov <action> <config_path> [-d] [-i]
```

* `<action>` can be
  * `install` to install model requirements (same as `-i`),
  * `download` to download model's data (same as `-d`),
  * `train` to train the model on the data specified in the config file,
  * `evaluate` to calculate metrics on the same dataset,
  * `interact` to interact via CLI,
  * `riseapi` to run a REST API server (see
    [doc](http://docs.deeppavlov.ai/en/master/integrations/rest_api.html)),
  * `predict` to get prediction for samples from *stdin* or from
      *<file_path>* if `-f <file_path>` is specified.
* `<config_path>` specifies path (or name) of model's config file
* `-d` downloads required data
* `-i` installs model requirements

### Python

To get predictions from a model interactively through Python, run

```python
from deeppavlov import build_model

model = build_model(<config_path>, install=True, download=True)

# get predictions for 'input_text1', 'input_text2'
model(['input_text1', 'input_text2'])
```

where

* `install=True` installs model requirements (optional),
* `download=True` downloads required data from web - pretrained model files and embeddings (optional),
* `<config_path>` is model name (e.g. `'ner_ontonotes_bert_mult'`), path to the chosen model's config file (e.g.
  `"deeppavlov/configs/ner/ner_ontonotes_bert_mult.json"`),  or `deeppavlov.configs` attribute (e.g.
  `deeppavlov.configs.ner.ner_ontonotes_bert_mult` without quotation marks).

You can train it in the same simple way:

```python
from deeppavlov import train_model 

model = train_model(<config_path>, install=True, download=True)
```

To train on your own data you need to modify dataset reader path in the
[train config doc](http://docs.deeppavlov.ai/en/master/intro/config_description.html#train-config).
The data format is specified in the corresponding model doc page.

You can also calculate metrics on the dataset specified in your config file:

```python
from deeppavlov import evaluate_model 

model = evaluate_model(<config_path>, install=True, download=True)
```

DeepPavlov also [allows](https://docs.deeppavlov.ai/en/master/intro/python.html) to build a model from components for
inference using Python.

## License

DeepPavlov is Apache 2.0 - licensed.
