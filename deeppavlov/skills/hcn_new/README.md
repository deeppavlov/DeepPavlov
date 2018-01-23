[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![tensorflow 1.4](https://img.shields.io/badge/tensorflow-1.4-green.svg)

# Dialogue Manager for goal-oriented task 
The dialogue manager is based on [[1]](#references) which introduces Hybrid Code Networks (HCNs) that combine an RNN with domain-specific knowledge and system action templates.

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
      * `name` — `"default_vocab"` (for vocabulary's implementation see [`deeppavlov.core.data.vocab`](../../core/data/vocab.py))
      * `inputs` — `[ "x" ]`,
      * `level` — `"token"`,
      * `ser_path` — `"../data/hcn/token.dict"`
* `tokenizer` — one of tokenizers from [`deeppavlov.models.tokenizers`](../../models/tokenizers) module
   * `name` — tokenizer name
   * other arguments specific to your tokenizer
* `bow_encoder` — one of bag-of-words encoders from [`deeppavlov.models.encoders.bow`](../../models/encoders/bow) module
   * `name` — encoder name
   * other arguments specific to your encoder
* `embedder` — pne of embedders from [`deeppavlov.models.embedders`](../../models/embedders) module
   * `name` — embedder name (`"fasttext"` recommended, see [`deeppavlov.models.embedders.fasttext_embedder`](../../models/embedders/fasttext_embedder.py))
   * `mean` — must be set to `true`
   * other arguments specific to your embedder
* `tracker` — dialogue state tracker from [`deeppavlov.models.trackers`](../../models/trackers)
   * `name` — tracker name (`"default_tracker"` or `"featurized_tracker"` recommended)
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
   * `name` — slot filler name (`"dstc_slotfilling"` recommended, for implementation see [`deeppavlov.models.ner`](../../models/ner))
   * other slot filler arguments
* `intent_classifier` — model that outputs intents probability disctibution for a given utterance
   * `name` — slot filler name (`"intent_model"` recommended, for implementation see [`deeppavlov.models.classifiers.intents`](../../models/classifiers/intents))
   * classifier's other arguments
* `debug` — whether to display debug output (defaults to `false`) _(optional)_

For a working exemplary config see [`deeeppavlov/skills/hcn_new/config.json`](config.json).

#### Usage example

## Training

#### Config parameters
To be used for training, your config json file should include the following parameters:

* `dataset_reader`
   * `name` — `"your_reader_here"` for a custom dataset or `"dstc2_datasetreader"` to use DSTC2 (for implementation see [`deeppavlov.dataset_readers.dstc2_dataset_reader`](../../dataset_readers/dstc2_datasetreader.py))
   * `data_path` — a path to a dataset file, which in case of `"dstc2_datasetreader"` will be automatically downloaded from 
   internet and placed to `data_path` directory
* `dataset` — it should always be set to `{"name": "dialog_dataset"}` (for implementation see [`deeppavlov.datasets.dstc2_datasets.py`](../../datasets/dstc2_datasets.py))

#TODO: rename dstc2_dialog_dataset to dialog_dataset

Do not forget to set `train_now` parameters to `true` for `vocabs.word_vocab`, `model` and `model.network` sections.

See [`deeeppavlov/skills/hcn_new/config.json`](config.json) for details.

#### Train run
The easiest way to run the training is by using [`deeppavlov/run_model.py`](../../run_model.py) script:

1. set `PIPELINE_CONFIG_PATH` to your config path relative to the deeppavlov library
(for example, `'skills/hcn_new/config.json'`)
2. then run the script by `python3 run_model.py`

The model will be trained according to your configuration and afterwards an interaction with the model will be run.

## Datasets

#### DSTC2
The Hybrid Code Network model was trained and evaluated on a modification of a dataset from Dialogue State Tracking Challenge 2[[2]](#references). The modifications were as follows:
* **new actions**
    * bot dialog actions were concatenated into one action (example: `{"dialog_acts": ["ask", "request"]}` -> `{"dialog_acts": ["ask_request"]}`)
    * if a slot key was associated with the dialog action, the new act was a concatenation of an act and a slot key (example: `{"dialog_acts": ["ask"], "slot_vals": ["area"]}` -> `{"dialog_acts": ["ask_area"]}`)
* **new train/dev/test split**
    * original dstc2 consisted of three different MDP polices, the original train and dev datasets (consisting of two polices) were merged and randomly split into train/dev/test
* **minor fixes**
    * fixed several dialogs, where actions were wrongly annotated
    * uppercased first letter of bot responses
    * unified punctuation for bot responses'

#### Your data
If your model uses DSTC2 and relies on `dstc2_datasetreader` [`DatasetReader`](../../core/data/dataset_reader.py), all needed files, if not present in the `dataset_reader.data_path` directory, will be downloaded from internet.

If your model needs be trained on different data, you have several ways of achieving that (sorted by increase in the amount of code):

1. Use `"dialog_dataset"` in dataset config section and `"dstc2_datasetreader"` in dataset reader config section (**the simplest, but not the best way**):
    * set `dataset.data_path` to your data directory;
    * your data files should have the same format as expected in [`deeppavlov.dataset_readers.dstc2_dataset_reader:DSTC2DatasetReader.read()`](../../dataset_readers/dstc2_dataset_reader.py) function.

2. Use `"dialog_dataset"` in dataset config section and `"your_dataset_reader"` in dataset reader config section (**recommended**): 
    * clone [`deeppavlov.dataset_readers.dstc2_dataset_reader:DSTC2DatasetReader`](../../dataset_readers/dstc2_dataset_reader.py) to `YourDatasetReader`;
    * register as `"your_dataset_reader"`;
    * rewrite so that it implements the same interface as the origin. Particularly, `YourDatasetReader.read()` must have the same output as `DSTC2DatasetReader.read()`:
      * `train` — training dialog turns consisting of tuples:
         * first tuple element contains first user's utterance info
            * `text` — utterance string
            * `intents` — list of string intents, associated with user's utterance
            * `db_result` — a database response _(optional)_
            * `episode_done` — set to `true`, if current utterance is the start of a new dialog, and `false` (or skipped) otherwise _(optional)_
         * second tuple element contains second user's response info
            * `text` — utterance string
            * `act` — an act, associated with the user's utterance
      * `valid` — validation dialog turns in the same format
      * `test` — test dialog turns in the same format
      
#TODO: change str `act` to a list `acts`

3. Use your own dataset and dataset reader (**if 2. doesn't work for you**):
    * your `YourDataset.iter()` class method output should match the input format for [`HybridCodeNetworkBot.train()`](hcn.py).

## Comparison
As far as out dataset is a modified version of official DSTC2-dataset [[2]](#references), resulting metrics can't be compared with model evaluations on the original dataset.

But comparisons for hcn model modifications trained on out DSTC2-dataset are presented:

|                   Model                      |  Action accuracy  |  Turn accuracy  |  Dialog accuracy |
|----------------------------------------------|-------------------|-----------------|------------------|
|basic hcn			                               |                   |                 |                  |
|hcn with ner slot-filler			                 |                   |                 |                  |
|hcn with ner slot-filler & fasttext embeddings|                   |                 |                  |
|hcn with ner slot-filler & fasttext & intents |                   |                 |                  |

#TODO: add metrics values

# References
[1] [Jason D. Williams, Kavosh Asadi, Geoffrey Zweig, Hybrid Code Networks: practical and efficient end-to-end dialog control with supervised and reinforcement learning – 2017](https://arxiv.org/abs/1702.03274)

[2] [Dialog State Tracking Challenge 2 dataset](http://camdial.org/~mh521/dstc/)
