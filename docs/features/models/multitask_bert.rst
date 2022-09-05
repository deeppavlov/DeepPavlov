Multi-task BERT in DeepPavlov
=============================

Multi-task BERT in DeepPavlov is an implementation of BERT training algorithm published in the paper "Multi-Task Deep
Neural Networks for Natural Language Understanding".

| Multi-task BERT paper: https://arxiv.org/abs/1901.11504

The idea is to share BERT body between several tasks. This is necessary if a model pipe has several
components using BERT and the amount of GPU memory is limited. Each task has its own 'head' part attached to the
output of the BERT encoder. If multi-task BERT has :math:`T` heads, one training iteration consists of

- composing :math:`T` mini-batches, one for each task,

- :math:`T` gradient steps, one gradient step for each task.

When one of BERT heads is being trained, other heads' parameters do not change. On each training step both BERT head
and body parameters are modified. You may specify different learning rates for a head and a body.

Currently there are heads for classification (``mt_bert_classification_task``) and sequence tagging
(``mt_bert_seq_tagging_task``).

At this page, multi-task BERT usage is explained on a toy configuration file of a model that detects
insults, analyzes sentiment, and recognises named entities. Multi-task BERT configuration files for training
:config:`mt_bert_train_tutorial.json <tutorials/mt_bert/mt_bert_train_tutorial.json>` and for inference
:config:`mt_bert_inference_tutorial.json <tutorials/mt_bert/mt_bert_inference_tutorial.json>` are based on configs
:config:`insults_kaggle_bert.json <classifiers/insults_kaggle_bert.json>`,
:config:`sentiment_sst_multi_bert.json <classifiers/sentiment_sst_multi_bert.json>`,
:config:`ner_conll2003_bert.json <ner/ner_conll2003_bert.json>`.

We start with the ``metadata`` field of the configuration file. Multi-task BERT model is saved in
``{"MT_BERT_PATH": "{MODELS_PATH}/mt_bert"}``. Classes and tag vocabularies are saved in
``{"INSULTS_PATH": "{MT_BERT_PATH}/insults"}``, ``{"SENTIMENT_PATH": "{MT_BERT_PATH}/sentiment"}``. ``downloads``
field of Multitask BERT configuration file is a union of ``downloads`` fields of original configs without pre-trained
models. The ``metadata`` field of our config is given below.

.. code:: json

  {
    "metadata": {
      "variables": {
        "ROOT_PATH": "~/.deeppavlov",
        "DOWNLOADS_PATH": "{ROOT_PATH}/downloads",
        "MODELS_PATH": "{ROOT_PATH}/models",
        "BERT_PATH": "{DOWNLOADS_PATH}/bert_models/cased_L-12_H-768_A-12",
        "MT_BERT_PATH": "{MODELS_PATH}/mt_bert_tutorial",
        "INSULTS_PATH": "{MT_BERT_PATH}/insults",
        "SENTIMENT_PATH": "{MT_BERT_PATH}/sentiment",
        "NER_PATH": "{MT_BERT_PATH}/ner"
      },
      "download": [
        {
          "url": "http://files.deeppavlov.ai/datasets/insults_data.tar.gz",
          "subdir": "{DOWNLOADS_PATH}"
        },
        {
          "url": "http://files.deeppavlov.ai/datasets/yelp_review_full_csv.tar.gz",
          "subdir": "{DOWNLOADS_PATH}"
        },
        {
          "url": "http://files.deeppavlov.ai/deeppavlov_data/bert/cased_L-12_H-768_A-12.zip",
          "subdir": "{DOWNLOADS_PATH}/bert_models"
        }
      ]
    }
  }

Train config
------------

When using ``multitask_bert`` component, you need **separate train and inference configuration files**.

Data reading and iteration is performed by ``multitask_reader`` and ``multitask_iterator``. These classes are composed
of task readers and iterators and generate batches that contain data from heterogeneous datasets.

A ``multitask_reader`` configuration has parameters ``class_name``, ``data_path``, and ``tasks``.
``data_path`` field may be any string because data paths are passed for tasks individually in ``tasks``
parameter. However, you can not drop a ``data_path`` parameter because it is obligatory for dataset reader
configuration. ``tasks`` parameter is a dictionary of task dataset readers configurations. In configurations of
task readers, ``reader_class_name`` parameter is used instead of ``class_name``. The dataset reader configuration is
provided:

.. code:: json

  {
    "dataset_reader": {
      "class_name": "multitask_reader",
      "data_path": "null",
      "tasks": {
        "insults": {
          "reader_class_name": "basic_classification_reader",
          "x": "Comment",
          "y": "Class",
          "data_path": "{DOWNLOADS_PATH}/insults_data"
        },
        "sentiment": {
          "reader_class_name": "basic_classification_reader",
          "x": "text",
          "y": "label",
          "data_path": "{DOWNLOADS_PATH}/yelp_review_full_csv",
          "train": "train.csv",
          "test": "test.csv",
          "header": null,
          "names": [
            "label",
            "text"
          ]
        },
        "ner": {
          "reader_class_name": "conll2003_reader",
          "data_path": "{DOWNLOADS_PATH}/conll2003/",
          "dataset_name": "conll2003",
          "provide_pos": false
        }
      }
    }
  }

A ``multitask_iterator`` configuration  has parameters ``class_name`` and ``tasks``. ``tasks`` is a dictionary of
configurations of task iterators. In configurations of task iterators, ``iterator_class_name`` is used instead of
``class_name``. The dataset iterator configuration is as follows:

.. code:: json

  {
    "dataset_iterator": {
      "class_name": "multitask_iterator",
      "tasks": {
        "insults": {
          "iterator_class_name": "basic_classification_iterator",
          "seed": 42
        },
        "sentiment": {
          "iterator_class_name": "basic_classification_iterator",
          "seed": 42,
          "split_seed": 23,
          "field_to_split": "train",
          "split_fields": [
            "train",
            "valid"
          ],
          "split_proportions": [
            0.9,
            0.1
          ]
        },
        "ner": {"iterator_class_name": "data_learning_iterator"}
      }
    }
  }

Batches generated by ``multitask_iterator`` are tuples of two elements: inputs of the model and labels. Both inputs
and labels are lists of tuples. The inputs have following format: ``[(first_task_inputs[0], second_task_inputs[0],
...), (first_task_inputs[1], second_task_inputs[1], ...), ...]`` where ``first_task_inputs``, ``second_task_inputs``,
and so on are x values of batches from task dataset iterators. The labels in the have the similar format.

If task datasets have different sizes, then smaller datasets are repeated until
their sizes are equal to the size of the largest dataset. For example, if the first task dataset inputs are
``[0, 1, 2, 3, 4, 5, 6]``, the second task dataset inputs are ``[7, 8, 9]``, and the batch size is ``2``, then
multi-task input mini-batches will be ``[(0, 7), (1, 8)]``, ``[(2, 9), (3, 7)]``, ``[(4, 8), (5, 9)]``, ``[(6, 7)]``.

In this tutorial, there are 3 datasets. Considering the batch structure, ``chainer`` inputs are:

.. code:: json

  {
    "in": ["x_insults", "x_sentiment", "x_ner"],
    "in_y": ["y_insults", "y_sentiment", "y_ner"]
  }

Sometimes a task dataset iterator returns inputs or labels consisting of more than one element. For example, in model
:config:`mt_bert_train_tutorial.json <kbqa/kbqa_mt_bert_train.json>` ``siamese_iterator`` input
element consists of 2 strings. If there is a necessity to split such a variable, ``InputSplitter`` component can
be used.

Data preparation steps in the pipe of tutorial config are similar to data preparation steps in the original
configs except for names of the variables.

A ``multitask_bert`` component has task-specific parameters and parameters that are common for all tasks. The first
are provided inside the ``tasks`` parameter. The ``tasks`` is a dictionary that keys are task names and values are 
task-specific parameters. **The names of tasks have to be the same in train and inference configs.**

If ``inference_task_names`` parameter of a ``multitask_bert`` component is provided, the component is created for
inference. Otherwise, it is created for training.

Task classes inherit ``MTBertTask`` class. Inputs and labels of a ``multitask_bert`` component are distributed between
the tasks according to the ``in_distribution`` and ``in_y_distribution`` parameters. You can drop these parameters if
only one task is called. In that case, all ``multitask_bert`` inputs are passed to the task. Another option is
to make a distribution parameter a dictionary whose keys are task names and values are numbers of arguments the tasks
take. If this option is used, the order of the ``multitask_bert`` component inputs in ``in`` and ``in_y`` parameters
must meet three conditions. First, ``in`` and ``in_y`` elements have to be grouped by tasks, e.g. arguments for the
first task, then arguments for the second task and so on. Secondly, the order of tasks in ``in`` and ``in_y`` has to
be the same as the order of tasks in the ``in_distribution`` and ``in_y_distribution`` parameters. Thirdly, in ``in``
and ``in_y`` parameters the arguments of a task have to be put in the same order as the order in which they are passed
to ``get_sess_run_infer_args`` and ``get_sess_run_train_args`` methods of the task. If ``in`` and ``in_y`` parameters
are dictionaries, you may make ``in_distribution`` and ``in_y_distribution`` parameter dictionaries which keys are
task names and values are lists of elements of ``in`` or ``in_y``.

.. code:: json

      {
        "id": "mt_bert",
        "class_name": "mt_bert",
        "save_path": "{MT_BERT_PATH}/model",
        "load_path": "{MT_BERT_PATH}/model",
        "bert_config_file": "{BERT_PATH}/bert_config.json",
        "pretrained_bert": "{BERT_PATH}/bert_model.ckpt",
        "attention_probs_keep_prob": 0.5,
        "body_learning_rate": 3e-5,
        "min_body_learning_rate": 2e-7,
        "learning_rate_drop_patience": 10,
        "learning_rate_drop_div": 1.5,
        "load_before_drop": true,
        "optimizer": "tf.train:AdamOptimizer",
        "clip_norm": 1.0,
        "tasks": {
          "insults": {
            "class_name": "mt_bert_classification_task",
            "n_classes": "#classes_vocab_insults.len",
            "keep_prob": 0.5,
            "return_probas": true,
            "learning_rate": 1e-3,
            "one_hot_labels": true
          },
          "sentiment": {
            "class_name": "mt_bert_classification_task",
            "n_classes": "#classes_vocab_sentiment.len",
            "return_probas": true,
            "one_hot_labels": true,
            "keep_prob": 0.5,
            "learning_rate": 1e-3
          },
          "ner": {
            "class_name": "mt_bert_seq_tagging_task",
            "n_tags": "#tag_vocab.len",
            "return_probas": false,
            "keep_prob": 0.5,
            "learning_rate": 1e-3,
            "use_crf": true,
            "encoder_layer_ids": [-1]
          }
        },
        "in_distribution": {"insults": 1, "sentiment": 1, "ner": 3},
        "in": [
          "bert_features_insults",
          "bert_features_sentiment",
          "x_ner_subword_tok_ids",
          "ner_attention_mask",
          "ner_startofword_markers"],
        "in_y_distribution": {"insults": 1, "sentiment": 1, "ner": 1},
        "in_y": ["y_insults_onehot", "y_sentiment_onehot", "y_ner_ind"],
        "out": ["y_insults_pred_probas", "y_sentiment_pred_probas", "y_ner_pred_ind"]
      }

You may need to design your own metric for early stopping. In this example, the target metric is an average of AUC ROC
for insults and sentiment tasks and F1 for NER task. In order to add a metric to config, you have to register the
metric. To register metric, add the decorator ``register_metric`` and run the command
``python -m utils.prepare.registry`` in DeepPavlov root directory. The code below should be placed in the file
``deeppavlov/metrics/fmeasure.py`` and registry is updated with command ``python -m utils.prepare.registry``.

.. code:: python

    @register_metric("average__roc_auc__roc_auc__ner_f1")
    def roc_auc__roc_auc__ner_f1(true_onehot1, pred_probas1, true_onehot2, pred_probas2, ner_true3, ner_pred3):
        from .roc_auc_score import roc_auc_score
        roc_auc1 = roc_auc_score(true_onehot1, pred_probas1)
        roc_auc2 = roc_auc_score(true_onehot2, pred_probas2)
        ner_f1_3 = ner_f1(ner_true3, ner_pred3) / 100
        return (roc_auc1 + roc_auc2 + ner_f1_3) / 3

Inference config
----------------

There is no need in dataset reader and dataset iterator in and inference config. A ``train`` field and components
preparing ``in_y`` are removed. In ``multitask_bert`` component configuration all training parameters (learning rate,
optimizer, etc.) are omitted.

For demonstration of DeepPavlov multi-task BERT functionality, in this example, the inference is made in 2 separate 
components: ``multitask_bert`` and ``mtbert_reuser``. The first component performs named entity recognition and the 
second performs insult detection and sentiment analysis.

To run NER using the ``multitask_bert`` component, ``inference_task_names`` parameter is added to
``multitask_bert`` component configuration. An ``inference_task_names`` parameter can be a string or a list containing
strings and lists of strings. If an ``inference_task_names`` parameter is a string, it is the name of the task called
separately (in individual ``tf.Session.run`` call). 

If an ``inference_task_names`` parameter is a list, then this list contains names of called tasks. You may group
several tasks to speed up inference if these tasks have common inputs. If an element of the ``inference_task_names``
is a list of task names, the tasks from the list are run simultaneously in one ``tf.Session.run`` call. Despite the
fact that tasks share inputs, you have to provide full sets of inputs for all tasks in ``in`` parameter of
``multitask_bert``. 

In the tutorial, NER task do not have common inputs with other tasks and have to be run
separately.

.. code:: json

      {
        "id": "mt_bert",
        "class_name": "mt_bert",
        "inference_task_names": "ner",
        "bert_config_file": "{BERT_PATH}/bert_config.json",
        "save_path": "{MT_BERT_PATH}/model",
        "load_path": "{MT_BERT_PATH}/model",
        "pretrained_bert": "{BERT_PATH}/bert_model.ckpt",
        "tasks": {
          "insults": {
            "class_name": "mt_bert_classification_task",
            "n_classes": "#classes_vocab_insults.len",
            "return_probas": true,
            "one_hot_labels": true
          },
          "sentiment": {
            "class_name": "mt_bert_classification_task",
            "n_classes": "#classes_vocab_sentiment.len",
            "return_probas": true,
            "one_hot_labels": true
          },
          "ner": {
            "class_name": "mt_bert_seq_tagging_task",
            "n_tags": "#tag_vocab.len",
            "return_probas": false,
            "use_crf": true,
            "encoder_layer_ids": [-1]
          }
        },
        "in": ["x_ner_subword_tok_ids", "ner_attention_mask", "ner_startofword_markers"],
        "out": ["y_ner_pred_ind"]
      }

``mtbert_reuser`` component is an interface to ``call`` method of ``MultiTaskBert`` class. ``mtbert_reuser``
component is provided with ``multitask_bert`` component, a list of task names for inference ``task_names`` (the format
is same as in ``inference_task_names`` parameter of ``multitask_bert``), and ``in_distribution`` parameter. Notice
that tasks "insults" and "sentiment" are grouped into a list of 2 elements. This syntax invokes inference of these
tasks in one call of ``tf.Session.run``. If ``task_names`` were equal to ``["insults", "sentiment"]``, the inference
of the tasks would be sequential and take approximately 2 times more time.

.. code:: json

      {
        "class_name": "mt_bert_reuser",
        "mt_bert": "#mt_bert",
        "task_names": [["insults", "sentiment"]],
        "in_distribution": {"insults": 1, "sentiment": 1},
        "in": ["bert_features", "bert_features"],
        "out": ["y_insults_pred_probas", "y_sentiment_pred_probas"]
      }

