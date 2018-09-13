[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/deepmipt/DeepPavlov/blob/master/LICENSE)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

__In version 0.0.6 everything from package `deeppavlov.skills` except `deeppavlov.skills.pattern_matching_skill` was moved to `deeppavlov.models` so your imports might break__  


DeepPavlov is an open-source conversational AI library built on [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/). It is designed for
 * development of production ready chat-bots and complex conversational systems,
 * NLP and dialog systems research.

# Hello Bot in DeepPavlov

Import key components to build HelloBot. 
```python
from deeppavlov.core.agent import Agent, HighestConfidenceSelector
from deeppavlov.skills.pattern_matching_skill import PatternMatchingSkill
```

Create skills as pre-defined responses for a user's input containing specific keywords. Every skill returns response and confidence.
```python
hello = PatternMatchingSkill(responses=['Hello world! :)'], patterns=["hi", "hello", "good day"])
bye = PatternMatchingSkill(['Goodbye world! :(', 'See you around.'], ["bye", "chao", "see you"])
fallback = PatternMatchingSkill(["I don't understand, sorry :/", 'I can say "Hello world!" 8)'])
```

Agent executes skills and then takes response from the skill with the highest confidence.
```python
HelloBot = Agent([hello, bye, fallback], skills_selector=HighestConfidenceSelector())
```

Give the floor to the HelloBot!
```python
print(HelloBot(['Hello!', 'Boo...', 'Bye.']))
```

[Jupyther notebook with HelloBot example.](docs/intro/hello_bot.ipynb)


# Features

**Components**

[Named Entity Recognition](http://docs.deeppavlov.ai/en/latest/components/ner.html) | [Slot filling](http://docs.deeppavlov.ai/en/latest/components/slot_filling.html)

[Intent/Sentence Classification](http://docs.deeppavlov.ai/en/latest/components/classifiers.html) |  [Sentence Similarity/Ranking](http://docs.deeppavlov.ai/en/latest/components/neural_ranking.html)

[Question Answering over Text (SQuAD)](http://docs.deeppavlov.ai/en/latest/components/squad.html) 

[Morphological tagging](http://docs.deeppavlov.ai/en/latest/components/morphotagger.html) | [Automatic Spelling Correction](http://docs.deeppavlov.ai/en/latest/components/spelling_correction.html)

**Skills**

[Goal(Task)-oriented Bot](http://docs.deeppavlov.ai/en/latest/skills/go_bot.html) | [Seq2seq Goal-Oriented bot](http://docs.deeppavlov.ai/en/latest/skills/seq2seq_go_bot.html)

[Open Domain Questions Answering](http://docs.deeppavlov.ai/en/latest/skills/odqa.html)

[Frequently Asked Questions Answering](http://docs.deeppavlov.ai/en/latest/skills/faq.html)

**Embeddings**

[ELMo embeddings for the Russian language](http://docs.deeppavlov.ai/en/master/apiref/models/embedders.html#deeppavlov.models.embedders.elmo_embedder.ELMoEmbedder)

[FastText embeddings for the Russian language](http://docs.deeppavlov.ai/en/latest/intro/pretrained_vectors.html)

**Auto ML**

[Tuning Models with Evolutionary Algorithm](http://docs.deeppavlov.ai/en/latest/intro/parameters_evolution.html)

# Installation

0. Currently we support only `Linux` platform and `Python 3.6` (**`Python 3.5` is not supported!**)

1. Create a virtual environment with `Python 3.6`:
    ```
    virtualenv env
    ```
2. Activate the environment:
    ```
    source ./env/bin/activate
    ```
3. Install the package inside this virtual environment:
    ```
    pip install deeppavlov
    ```

# Demo 

Demo of selected features is available at [demo.ipavlov.ai](https://demo.ipavlov.ai/)

# Quick start

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

For the `interactbot` mode you should specify Telegram bot token in `-t` parameter or in `TELEGRAM_TOKEN` environment variable. Also if you want to get custom `/start` and `/help` Telegram messages for the running model you should:
* Add section to `utils/telegram_utils/model_info.json` with your custom Telegram messages
* In model config file specify `metadata.labels.telegram_utils` parameter with name which refers to the added section of `utils/telegram_utils/model_info.json`

For the `interactmsbot` mode you should specify **Microsoft app id** in `-i` and **Microsoft app secret** in `-s`. Also before launch you should specify api deployment settings (host, port) in [*utils/server_config.json*](utils/server_utils/server_config.json) configuration file. Note, that Microsoft Bot Framework requires `https` endpoint with valid certificate from CA.
Here is [detailed info on the Microsoft Bot Framework integration](http://docs.deeppavlov.ai/en/latest/devguides/ms_bot_integration.html) 

For `riseapi` mode you should specify api settings (host, port, etc.) in [*utils/server_config.json*](utils/server_utils/server_config.json) configuration file. If provided, values from *model_defaults* section override values for the same parameters from *common_defaults* section. Model names in *model_defaults* section should be similar to the class names of the models main component.
Here is [detailed info on the DeepPavlov REST API](http://docs.deeppavlov.ai/en/latest/devguides/rest_api.html)

For `predict` you can specify path to input file with `-f` or `--input-file` parameter, otherwise, data will be taken
from stdin.  
Every line of input text will be used as a pipeline input parameter, so one example will consist of as many lines,
as many input parameters your pipeline expects.  
You can also specify batch size with `-b` or `--batch-size` parameter.

# Documentation

[docs.deeppavlov.ai](http://docs.deeppavlov.ai/)

# Tutorials

Jupyter notebooks and videos explaining how to use DeepPalov for different tasks can be found in [/examples/tutorials/](examples/tutorials/)

# License

DeepPavlov is Apache 2.0 - licensed.

# Support and collaboration

If you have any questions, bug reports or feature requests, please feel free to post on our [Github Issues](https://github.com/deepmipt/DeepPavlov/issues) page. Please tag your issue with `bug`, `feature request`, or `question`.  Also we’ll be glad to see your pull requests to add new datasets, models, embeddings, etc.

# The Team

DeepPavlov is built and maintained by [Neural Networks and Deep Learning Lab](https://mipt.ru/english/research/labs/neural-networks-and-deep-learning-lab) at [MIPT](https://mipt.ru/english/) within [iPavlov](http://ipavlov.ai/) project (part of [National Technology Initiative](https://asi.ru/eng/nti/)) and in partnership with [Sberbank](http://www.sberbank.com/).

<p align="center">
<img src="https://ipavlov.ai/img/ipavlov_footer.png" width="50%" height="50%"/>
</p>
