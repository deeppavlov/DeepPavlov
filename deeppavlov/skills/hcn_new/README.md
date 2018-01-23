[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![tensorflow 1.4](https://img.shields.io/badge/tensorflow-1.4-green.svg)

# Dialogue Manager for goal-oriented task 


The dialogue manager is based on [1] which introduces Hybrid Code Networks (HCNs) that combine an RNN with domain-specific knowledge and system action templates.

Compared to existing end-to-end approaches, HCNs considerably reduce the amount of training data required, while retaining the key benefit of inferring a latent representation of dialog state.

Here is an simple example of interaction with a trained dialogue bot:

```
::
>> Hello, welcome to the Phystech restaurant system. You can ask for restaurants by area, price range or food type. How may I help you?
:: hi, i want some food
>> What part of town do you have in mind?
:: north of Dolgoprudny
>> What kind of food would you like?
:: tasty =)
>> "Time to eat" is a great restaurant.
:: ok, is it cheap?
>> "Time to eat" is a nice place in the north of town and the prices are cheap.
:: give me their address then
>> Sure, "Time to eat" is on Sobina Square, 1/4.
:: and phone number. In case, i wanted to order something for delivery
>> The phone number of "Time to eat" is +7(965)173-37-33.
:: it's useless, but tell me their post code
>> The post code of "Time to eat" is 141700.
:: thanks
>> "Time to eat" is a nice place in the north of town and the prices are cheap.
:: good bye then
>> You are welcome!
```

## Usage

#### Requirements

#### Config parameters:
* `name` always equals to `"hcn_new"`
* `train_now` — `true` or `false`(default) depending on whether you are training or using a model _(optional)_
* `num_epochs` — maximum number of epochs during training _(optional)_
* `val_patience` — stop training after `val_patience` epochs without improvement of turn accuracy on validation dialogs _(optional)_
* `template_path` — map from actions to text templates for response generation
* `vocabs` — vocabs used in model
   * `word_vocab` — vocabulary of tokens from context utterances
   * `train_now` — whether to train it on the current dataset, or use pretrained
   * `name` — `"default_vocab"`,
   * `inputs` — `[ "x" ]`,
   * `level` — `"token"`,
   * `ser_path` — `"../data/hcn/token.dict"`
* `tokenizer` — description of a tokenizer from `from deeppavlov.models.tokenizers` module
   * `name` — tokenizer name
   * other arguments specific to your tokenizer
* `bow_encoder` — description of a bag-of-words encoder from `deeppavlov.models.encoders.bow` module
   * `name` — encoder name
   * other arguments specific to your encoder
* `embedder` — description of an embedder from `deeppavlov.models.embedders` module
   * `name` — embedder name
   * `mean` — must be set to `true`
   * other arguments specific to your embedder
* `tracker` — dialogue state tracker from `deeppavlov.models.trackers`
   * `name` — tracker name (`featurized_tracker` recommended)
   * other arguments specific to your tracker
* `network` — reccurent network that handles dialogue policy management
   * `name` — `"custom_rnn"`,
   * `train_now` — `true` or `false`(default) depending on whether you are training or using a model _(optional)_
   * `ser_path` — name of the file that the model will be saved to and loaded from
   * `learning_rate` — learning rate during training
   * `hidden_dim` — hidden state dimension
   * `obs_size` — input observation size
   * `use_action_mask` — in case of true, action mask is applied to probability distribution
   * `action_size` — output action size
* `slot_filler` — model that predicts slot values for a given utterance
   * `name` — slot filler name, `"dstc_slotfilling"` recommended
   * other slot filler arguments
* `intent_classifier` — model that outputs intents probability disctibution for a given utterance
   * `name` — slot filler name, `"intent_model"` recommended
   * classifier's other arguments
* `debug` — whether to display debug output (defaults to `false`) _(optional)_

For a working exemplary config see `deeeppavlov/skills/hcn_new/config.json`.

#### Usage example

## Training

#### Config parameters

To be used for training, your config json file should include the following parameters:

* `dataset_reader`
   * `name` — `"your_reader_here"` for a custom dataset or `"dstc2_datasetreader"` to use DSTC2
   * `ser_path` — a path to a dataset file, which in case of `"dstc2_datasetreader"` will be automatically downloaded from 
   internet and placed to `ser_path`
* `dataset` — it should always be set to `{"name": "dialog_dataset"}`

#TODO: rename dstc2_dialog_dataset to dialog_dataset

Do not forget to set `train_now` parameters to `true` for `vocabs.word_vocab`, `model` and `model.network` sections.

See `deeppavlov/skills/hcn_new/config.json` for details.

#### Train run

The easiest way to run the training is by using `deeppavlov/run_model.py` script:

1. set `MODEL_CONFIG_PATH` to your config path relative to the deeppavlov library
(for example, `'skills/hcn_new/config.json'`)
2. then run the script by `python3 run_model.py`

The model will be trained according to your configuration and afterwards an interaction with the model will be run.

## Datasets

#### DSTC2

#TODO: note on dstc2 modification

#### Your data

Constructing intents from DSTC 2 makes IntentDataset a bit difficult. Therefore, another dataset reader ClassificationDatasetReader and dataset ClassificationDataset to work with .csv files are also provided in deeppavlov/dataset_readers and deeppavlov/datasets.

Training data files train.csv (and, if exists, valid.csv) should be presented in the following form:

## Comparison

As far as out dataset is a modified version of official DSTC2-dataset[2], resulting metrics can't be compared with model evaluations on the original dataset.

But comparisons for hcn model modifications trained on out DSTC2-dataset are presented:

|                   Model                      |  Action accuracy  |  Turn accuracy  |  Dialog accuracy |
|----------------------------------------------|-------------------|-----------------|------------------|
|basic hcn			                             |                   |                 |                  |
|hcn with ner slot-filler			              |                   |                 |                  |
|hcn with ner slot-filler & fasttext embeddings|                   |                 |                  |
|hcn with ner slot-filler & fasttext & intents |                   |                 |                  |

#TODO: add metrics values

# References

[1] [Jason D. Williams, Kavosh Asadi, Geoffrey Zweig, Hybrid Code Networks: practical and efficient end-to-end dialog control with supervised and reinforcement learning – 2017](https://arxiv.org/abs/1702.03274)

[2] [Dialog State Tracking Challenge 2 dataset](http://camdial.org/~mh521/dstc/)
