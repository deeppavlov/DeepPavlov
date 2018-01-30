[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

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

## Installation
1. Create a virtual environment with `Python 3.6`
2. Activate the environment.
3. `cd` to the project root.
4. `pip install -r requirements.txt`
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
```bash
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

### Describe modules here (skills, models, data, etc.)

### Config

An NLP pipeline config is a JSON file, which consists of four required elements:

```json
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
### DatasetReader
`DatasetReader` class reads data and returns it in a specified format.
A concrete `DatasetReader` class should be inherited from base
`deeppavlov.data.dataset_reader.DatasetReader` class and registered with a codename:

```python
@register('dstc2_datasetreader')
class DSTC2DatasetReader(DatasetReader):
```

### Dataset

### Vocab

### Model

## Suggested models
