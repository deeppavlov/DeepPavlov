[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/deepmipt/DeepPavlov/blob/master/LICENSE)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

_We are still in a really early Alpha release._  
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


# Installation

0. Currently we support only `Linux` platform and `Python 3.6` (**`Python 3.5` is not supported!**)

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
4. Install basic requirements:
    ```
    python setup.py develop
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

* `<mode>` can be `train`, `predict`, `interact`, `interactbot` or `riseapi`
* `<path_to_config>` should be a path to an NLP pipeline json config (e.g. `deeppavlov/configs/ner/slotfill_dstc2.json`)
or a name without the `.json` extension of one of the config files [provided](deeppavlov/configs) in this repository (e.g. `slotfill_dstc2`)

For the `interactbot` mode you should specify Telegram bot token in `-t` parameter or in `TELEGRAM_TOKEN` environment variable. Also if you want to get custom `/start` and `/help` Telegram messages for the running model you should:
* Add section to `utils/telegram_utils/model_info.json` with your custom Telegram messages
* In model config file specify `metadata.labels.telegram_utils` parameter with name which refers to the added section of `utils/telegram_utils/model_info.json`

For `riseapi` mode you should specify api settings (host, port, etc.) in [*utils/server_utils/server_config.json*](utils/server_utils/server_config.json) configuration file. If provided, values from *model_defaults* section override values for the same parameters from *common_defaults* section. Model names in *model_defaults* section should be similar to the class names of the models main component.

For `predict` you can specify path to input file with `-f` or `--input-file` parameter, otherwise, data will be taken
from stdin.  
Every line of input text will be used as a pipeline input parameter, so one example will consist of as many lines,
as many input parameters your pipeline expects.  
You can also specify batch size with `-b` or `--batch-size` parameter.

# Features

| Component | Description |
| --------- | ----------- |
| [NER component](docs/components/ner.rst) | Based on neural Named Entity Recognition network. The NER component reproduces architecture from the paper [Application of a Hybrid Bi-LSTM-CRF model to the task of Russian Named Entity Recognition](https://arxiv.org/pdf/1709.09686.pdf) which is inspired by Bi-LSTM+CRF architecture from https://arxiv.org/pdf/1603.01360.pdf. |
| [Slot filling components](docs/components/slot_filling.rst) | Based on fuzzy Levenshtein search to extract normalized slot values from text. The components either rely on NER results or perform needle in haystack search.|
| [Classification component](docs/components/classifiers.rst) | Component for classification tasks (intents, sentiment, etc) on word-level. Shallow-and-wide CNN, Deep CNN, BiLSTM,  BiLSTM with self-attention and other models are presented. The model allows multilabel classification of sentences. |
| [Goal-oriented bot](docs/components/go_bot.rst) | Based on Hybrid Code Networks (HCNs) architecture from [Jason D. Williams, Kavosh Asadi, Geoffrey Zweig, Hybrid Code Networks: practical and efficient end-to-end dialog control with supervised and reinforcement learning – 2017](https://arxiv.org/abs/1702.03274). It allows to predict responses in goal-oriented dialog. The model is customizable: embeddings, slot filler and intent classifier can switched on and off on demand.  |
| [Seq2seq goal-oriented bot](docs/components/seq2seq_go_bot.rst) | Dialogue agent predicts responses in a goal-oriented dialog and is able to handle multiple domains (pretrained bot allows calendar scheduling, weather information retrieval, and point-of-interest navigation). The model is end-to-end differentiable and does not need to explicitly model dialogue state or belief trackers. |
| [Automatic spelling correction component](docs/components/spelling_correction.rst) | Pipelines that use candidates search in a static dictionary and an ARPA language model to correct spelling errors. |
| [Ranking component](docs/components/neural_ranking.rst) |  Based on [LSTM-based deep learning models for non-factoid answer selection](https://arxiv.org/abs/1511.04108). The model performs ranking of responses or contexts from some database by their relevance for the given context. |
| [Question Answering component](docs/components/squad.rst) | Based on [R-NET: Machine Reading Comprehension with Self-matching Networks](https://www.microsoft.com/en-us/research/publication/mrc/). The model solves the task of looking for an answer on a question in a given context ([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) task format). |
| [Morphological tagging component](docs/components/morphotagger.rst) | Based on character-based approach to morphological tagging [Heigold et al., 2017. An extensive empirical evaluation of character-based morphological tagging for 14 languages](http://www.aclweb.org/anthology/E17-1048). A state-of-the-art model for Russian and several other languages. Model assigns morphological tags in UD format to sequences of words.|
| **Skills** |  |
|[ODQA](docs/skills/odqa.rst) | An open domain question answering skill. The skill accepts free-form questions about the world and outputs an answer based on its Wikipedia knowledge.|
| **Parameters Evolution** |  |
| [Parameters evolution for models](docs/intro/parameters_evolution.rst) | Implementation of parameters evolution for DeepPavlov models that requires only some small changes in a config file. |
| **Embeddings** |  |
| [Pre-trained embeddings for the Russian language](docs/intro/pretrained_vectors.rst) | Word vectors for the Russian language trained on joint [Russian Wikipedia](https://ru.wikipedia.org/) and [Lenta.ru](https://lenta.ru/) corpora. |

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
