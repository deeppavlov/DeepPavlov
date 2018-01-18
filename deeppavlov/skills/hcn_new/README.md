[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![tensorflow 1.4](https://img.shields.io/badge/tensorflow-1.4-green.svg)

# Dialogue Manager for goal-oriented task 

The dialogue manager is based on
["Hybrid Code Networks: practical and efficient end-to-end dialog control with supervised and reinforcement learning" (Jason D. Williams et al., 2017)](https://arxiv.org/abs/1702.03274)
which introduces Hybrid Code Networks (HCNs) that combine an RNN with domain-specific knowledge and system action templates. 

Compared to existing end-to-end approaches, HCNs considerably reduce the amount of training data required, 
while retaining the key benefit of inferring a latent representation of dialog state.

## Usage

#### Config parameters:  
* `name` always equals to `"hcn_new"`
* `train_now` — with the flag set to `true` the model will be trained _(optional)_
* `num_epochs` — maximum number of epochs during training _(optional)_
* `val_patience` — stop training after `val_patience` epochs without improvement of turn accuracy on validation dialogs _(optional)_
* `template_path` — map from actions to text templates for response generation

#TODO: add vocabs description
* `vocabs` — vocabs used in model
    * `word_vocab` — 
        * `name` — "default_vocab",
        * `train` — true,
        *  `inputs` — [ "x" ],
        * `level` — "token",
        * `model_path` — "../data/hcn/token.dict"
* `tokenizer` — description of a tokenizer from from `deeppavlov.models.tokenizers` module
    * `name` — tokenizer name
    * `..` — other arguments specific to your tokenizer
* `bow_encoder` — description of a bag-of-words encoder from `deeppavlov.models.encoders.bow` module
    * `name` — encoder name
    * `..` — other arguments specific to your encoder
* `embedder` — description of an embedder from `deeppavlov.models.embedders` module
    * `name` — embedder name
    * `mean` — must be set to `true`
    * `..`  — other arguments specific to your embedder
* `tracker` — dialogue state tracker from `deeppavlov.models.trackers`
    * `name` — tracker name (`featurized_tracker` recommended)
    * `..` — other arguments specific to you tracker
* `network` — 
    * `name` — "custom_rnn",
    * `model_path` — name of the file that the model will be saved to and loaded from
    * `learning_rate` — learning rate while training
    * `hidden_dim` — hidden state dimension
    * `obs_size` — input observation size
    * `use_action_mask` — in case of `true`, action mask is applied to probability distribution
    * `action_size` — output action size
* `slot_filler` — 
* `intent_classifier` — 
* `debug` — whether to display debug output (defaults to `false`) _(optional)_

A working config could look like this:

```json
{
  "vocabs": {
    "word_vocab": {
      "name": "default_vocab",
      "train":  true,
      "inputs": [ "x" ],
      "level": "token",
      "model_path": "../data/hcn/token.dict"
    },
    "token_vocab": {
      "name": "default_vocab",
      "train": false,
      "inputs": [ "x" ],
      "level": "token",
      "model_path": "../data/ner/token.dict"
    },
    "tag_vocab": {
      "name": "default_vocab",
      "train": false,
      "inputs": [ "y" ],
      "level": "token",
      "model_path": "../data/ner/tag.dict"
    },
    "char_vocab": {
      "name": "default_vocab",
      "train": false,
      "inputs": [ "x" ],
      "level": "char",
      "model_path": "../data/ner/char.dict"
    }
  },
  "model": {
    "name": "hcn_new",
    "train_now": true,
    "num_epochs": 200,
    "val_patience": 3,
    "template_path": "../data/dstc2/dstc2-templates.txt",
    "tokenizer": {
      "name": "spacy_tokenizer"
    },
    "bow_encoder": {
      "name": "bow"
    },
    "embedder": {
      "name": "fasttext",
      "emb_module": "pyfasttext",
      "mean": true,
      "dim": 300,
      "model_path": "../data/embeddings/wiki.en.bin"
    },
    "tracker": {
      "name": "featurized_tracker",
      "slot_names": [
        "pricerange",
        "this",
        "area",
        "slot",
        "food",
        "name"
      ]
    },
    "network": {
      "name": "custom_rnn",
      "train_now": true,
      "model_path": "../data/hcn/model",
      "learning_rate": 0.05,
      "hidden_dim": 128,
      "obs_size": 831,
      "use_action_mask": false,
      "action_size": 45

    },
    "slot_filler": {
      "name": "dstc_slotfilling",
      "model_path": "../data/ner",
      "ner_network": {
          "name": "ner_tagging_network",
          "filter_width": 7,
          "embeddings_dropout": true,
          "n_filters": [
              64,
              64
          ],
          "token_embeddings_dim": 64,
          "char_embeddings_dim": 32,
          "use_batch_norm": true,
          "use_crf": true
        }
    },
    "intent_classifier": {
      "name": "intent_model",
      "model_path": "models/classifiers/intents/intent_cnn_best",
      "opt": {
          "classes_file": "models/classifiers/intents/intent_cnn_best/classes.txt",
          "lear_metrics": "binary_accuracy fmeasure",
          "confident_threshold": 0.5,
          "optimizer": "Adam",
          "lear_rate": 1.0,
          "lear_rate_decay": 0.1,
          "loss": "binary_crossentropy",
          "coef_reg_cnn": 1e-4,
          "coef_reg_den": 1e-4,
          "dropout_rate": 0.5,
          "epochs": 1,
          "model_name": "cnn_model",
          "batch_size": 64,
          "val_every_n_epochs": 5,
          "verbose": true,
          "val_patience": 5,
          "show_examples": false
      },
      "embedder": {
          "name": "fasttext",
          "model_path": "../data/embeddings/dstc2_fasttext_model_100.bin",
          "emb_module": "fasttext",
          "dim": 100
      }
    },
    "debug": false
  }
}
```
