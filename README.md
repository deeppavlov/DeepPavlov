# DeepPavlov (pre alfa)
An Apache 2.0 NLP research library, built on TensorFlow and Keras, for 
 * building complicated natural language processing pipelines
 * training and infering natural language processing algorithms
 
## Features
 * Goal-oriented dialog agent
 * Slot filling commponent
 * Intnet classification component
 * Automatic spelling correction component
 * Pretrained embeddings library

## Installation
1. Create a virtual environment with `Python 3.6`
2. Activate the environment.
3. `cd` to the project root.
4. `pip install -r requirements.txt`

## Basic examples
With the purpose to interact with our trained models, they should be downloaded first:

```
cd deeppavlov/
python download.py -all
```

Then the models can be interacted or trained with the following command:

```
python deep.py <mode> <path_to_config>
```
* `<mode>` can be 'train' or 'interact'
* `<path_to_config>` should be a path to an NLP pipeline json config

Available model configs are:

*skills/hcn_new/config.json*

*models/classifiers/intents/config.json*

*models/ner/config.json*

*models/spellers/error_model/config_en.json*

## Support

If you have any questions, bug reports or feature requests, please don't hesitate to post on our Github Issues page.

## The Team

DeepPavlov is currently maintained by ...

## License

DeepPavlov is Apache 2.0 - licensed.
