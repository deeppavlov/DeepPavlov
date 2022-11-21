[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python 3.6, 3.7, 3.8, 3.9](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-green.svg)
[![Downloads](https://pepy.tech/badge/deeppavlov)](https://pepy.tech/project/deeppavlov)
<img align="right" height="27%" width="27%" src="docs/_static/deeppavlov_logo.png"/>

DeepPavlov is an open-source conversational AI library built on [PyTorch](https://pytorch.org/).

DeepPavlov is designed for
* development of production ready chat-bots and complex conversational systems,
* research in the area of NLP and, particularly, of dialog systems.

## Quick Links

* Demo [*demo.deeppavlov.ai*](https://demo.deeppavlov.ai/)
* Documentation [*docs.deeppavlov.ai*](http://docs.deeppavlov.ai/)
    * Model List [*docs:features/*](http://docs.deeppavlov.ai/en/master/features/overview.html)
    * Contribution Guide [*docs:contribution_guide/*](http://docs.deeppavlov.ai/en/master/devguides/contribution_guide.html)
* Issues [*github/issues/*](https://github.com/deeppavlov/DeepPavlov/issues)
* Forum [*forum.deeppavlov.ai*](https://forum.deeppavlov.ai/)
* Blogs [*medium.com/deeppavlov*](https://medium.com/deeppavlov)
* [Extended colab tutorials](https://github.com/deeppavlov/dp_tutorials)
* Docker Hub [*hub.docker.com/u/deeppavlov/*](https://hub.docker.com/u/deeppavlov/) 
    * Docker Images Documentation [*docs:docker-images/*](http://docs.deeppavlov.ai/en/master/intro/installation.html#docker-images)

Please leave us [your feedback](https://forms.gle/i64fowQmiVhMMC7f9) on how we can improve the DeepPavlov framework.

**Models**

[Named Entity Recognition](http://docs.deeppavlov.ai/en/master/features/models/NER.html) | [Intent/Sentence Classification](http://docs.deeppavlov.ai/en/master/features/models/classifiers.html) |

[Question Answering over Text (SQuAD)](http://docs.deeppavlov.ai/en/master/features/models/SQuAD.html) | [Knowledge Base Question Answering](http://docs.deeppavlov.ai/en/master/features/models/kbqa.html)

[Sentence Similarity/Ranking](http://docs.deeppavlov.ai/en/master/features/models/neural_ranking.html) | [TF-IDF Ranking](http://docs.deeppavlov.ai/en/master/features/models/tfidf_ranking.html) 

[Automatic Spelling Correction](http://docs.deeppavlov.ai/en/master/features/models/spelling_correction.html) | [Entity Linking](http://docs.deeppavlov.ai/en/master/features/models/entity_linking.html)

[Open Domain Questions Answering](http://docs.deeppavlov.ai/en/master/features/models/odqa.html) | [Frequently Asked Questions Answering](http://docs.deeppavlov.ai/en/master/features/models/faq.html)

[Russian SuperGLUE](http://docs.deeppavlov.ai/en/master/features/models/superglue.html)

**Embeddings**

[BERT embeddings for the Russian, Polish, Bulgarian, Czech, and informal English](http://docs.deeppavlov.ai/en/master/features/pretrained_vectors.html#bert)

[ELMo embeddings for the Russian language](http://docs.deeppavlov.ai/en/master/features/pretrained_vectors.html#elmo)

[FastText embeddings for the Russian language](http://docs.deeppavlov.ai/en/master/features/pretrained_vectors.html#fasttext)

**Auto ML**

[Tuning Models](http://docs.deeppavlov.ai/en/master/features/hypersearch.html)

**Integrations**

[REST API](http://docs.deeppavlov.ai/en/master/integrations/rest_api.html) | [Socket API](http://docs.deeppavlov.ai/en/master/integrations/socket_api.html)

[Amazon AWS](http://docs.deeppavlov.ai/en/master/integrations/aws_ec2.html)

## Installation

0. We support `Linux` platform, `Python 3.6`, `3.7`, `3.8` and `3.9`
    * **`Python 3.5` is not supported!**

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

## License

DeepPavlov is Apache 2.0 - licensed.
