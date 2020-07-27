Multitask BERT in DeepPavlov
============================

Multitask BERT in DeepPavlov is an implementation of BERT training algorithm published in the paper "Multi-Task Deep
Neural Networks for Natural Language Understanding".

| Multitask BERT paper: https://arxiv.org/abs/1901.11504

The idea is to share BERT body between several tasks. This is necessary if a model pipe has several
components using BERT and the amount of GPU memory is limited. Each task has its own 'head' part attached to the output
of the BERT encoder. If Multitask BERT has :math:`T` heads 1 training iteration consists of

- composing :math:`T` minibatches, one for each task,

- :math:`T` gradient steps, one gradient step for each task.

When one of BERT heads is being trained, other heads parameters do not change. On each training step both BERT head
and body parameters are modified. You may specify different learning rates for head and body.

Currently there are heads for classification (``mt_bert_classification_task``) and sequence tagging
(``mt_bert_seq_tagging_task``).

At this page Multitask BERT usage is explained on a toy configuration file of a model which detects
insults, analyzes sentiment and recognises named entities. Multitask Bert configuration files for train
:config:`mt_bert_train_tutorial.json <tutorials/mt_bert/mt_bert_train_tutorial.json>` and for inference
:config:`mt_bert_inference_tutorial.json <tutorials/mt_bert/mt_bert_inference_tutorial.json>` are based on configs
:config:`insults_kaggle_bert.json <classifiers/insults_kaggle_bert.json>`,
:config:`sentiment_sst_multi_bert.json <classifiers/sentiment_sst_multi_bert.json>`,
:config:`ner_conll2003_bert.json <ner/ner_conll2003_bert.json>`.

We start with ``metadata`` field  of configuration file. Multitask Bert model is saved in
``{"MT_BERT_PATH": "{MODELS_PATH}/mt_bert"}``. Classes and tag vocabularies are saved in
``{"INSULTS_PATH": "{MT_BERT_PATH}/insults"}``, ``{"SENTIMENT_PATH": "{MT_BERT_PATH}/sentiment"}``. ``requirements``
field of Multitask BERT configuration file is identical to ``requirements`` fields of original configs. ``downloads``
field of Multitask BERT configuration file is a union of ``downloads`` fields of original configs without pretrained
models. The ``metadata`` field of config is below

.. code: json

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
    "requirements": [                                                                                                     
      "{DEEPPAVLOV_PATH}/requirements/tf.txt",                                                                            
      "{DEEPPAVLOV_PATH}/requirements/bert_dp.txt",                                                                       
      "{DEEPPAVLOV_PATH}/requirements/fasttext.txt",                                                                      
      "{DEEPPAVLOV_PATH}/requirements/rapidfuzz.txt",                                                                     
      "{DEEPPAVLOV_PATH}/requirements/hdt.txt"                                                                            
    ],                                                                                                                    
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


Train config
------------

When using ``multitask_bert`` component you need **separate train and inference configuration files**.

Data reading and iteration is performed in ``multitask_reader`` and ``multitask_iterator``. These classes are composed
of task readers and iterators and generate batches which contain data from heterogeneous datasets.

``multitask_reader`` configuration has parameters ``class_name'``, ``data_path`` and ``tasks``.
``data_path`` field may have any string because data paths are passed for all tasks individually in ``tasks``
parameters. However, you can not drop ``data_path`` parameter because it is obligatory for dataset reader
configuration. ``tasks`` parameter is a dictionary of task dataset readers configurations. In configurations of
task readers ``reader_class_name`` parameter is used instead of ``class_name``. The dataset reader configuration is
below.

.. code:: json

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

``multitask_iterator`` configuration  has parameters ``class_name`` and ``tasks``. ``tasks`` is a dictionary of
configurations of task iterators. In configurations of task iterators ``iterator_class_name`` is used instead of
``class_name``. The dataset iterator configuration is below.

.. code:: json

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

Batches generated by ``multitask_iterator`` are tuples of two elements: inputs of the model and labels. Both inputs and
labels are lists of tuples. The inputs has following format: ``[(first_task_inputs[0], second_task_inputs[0], ...),
(first_task_inputs[1], second_task_inputs[1], ...), ...]`` where ``first_task_inputs``, ``second_task_inputs`` and so
on are x values of batches from task dataset iterators. Labels have the same format.

If task datasets have different sizes then smaller datasets are repeated until
their sizes are equal to the size of the largest dataset. For example, if the first task dataset inputs are
``[0, 1, 2, 3, 4, 5, 6]``, the second task dataset inputs are ``[7, 8, 9]`` and batch size is ``2`` then multitask
input mini-batches are ``[(0, 7), (1, 8)]``, ``[(2, 9), (3, 7)]``, ``[(4, 8), (5, 9)]``, ``[(6, 7)]``.

In this example there are 3 datasets. Considering the batch structure ``chainer`` inputs are

.. code:: json

    "in": ["x_insults", "x_sentiment", "x_ner"],
    "in_y": ["y_insults", "y_sentiment", "y_ner"],

Data preparation steps in pipe are similar to original configs except for names of the variables.

``multitask_bert`` component has parameters that are common for all tasks and task specific parameters. The latter
is provided inside ``tasks`` parameter. ``tasks`` is a dictionary which keys are task names and values are task
specific parameters. **The names of tasks have to be similar in train and inference configs.**

If ``inference_task_names`` parameter of ``multitask_bert`` component is provided the component is created for
inference. Otherwise it is created for training.

Task classes inherit ``MTBertTask`` class. Inputs and labels of ``multitask_bert`` component are distributed between
tasks according to ``in_distribution`` and ``in_y_distribution`` parameters. You can drop this parameters if only one
task is called. In that case all ``multitask_bert`` inputs are passed to the called task. Another option is to make
distribution parameter a dictionary which keys are task names and values are numbers of arguments called tasks take.
If this option is used, the order of component inputs in 'in' or 'in_y' has to match the order of tasks in
corresponding distribution parameter and the order of every task arguments has to match order of arguments of
``get_sess_run_infer_args`` and ``get_sess_run_train_args`` methods of the task. If 'in' and 'in_y' are dictionaries
you may make distribution parameter a dictionary which keys are task names and values are lists of key of 'in' or
'in_y'.

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

For early you may need to design your own metric. Here are target metric is an average of AUC ROC for insults and
sentiment tasks and F1 NER task. In order to add a metric to config the metric has to be registered. To register
metric add decorator ``register_metric`` and run command ``python -m utils.prepare.registry``. The code below should
work if put in file ``deeppavlov/metrics/fmeasure.py``.

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

Let's compare train and inference configs. In inference config there is no dataset reader and dataset iterator in test
config and there is no 'in_y' preparation components in pipe. ``train`` field can be removed. In ``multitask_bert``
configuration all training parameters (learning rate, optimizer, etc.) are omitted.

For demonstration of ``multitask_bert`` component functionality inference is made in 2 components: ``multitask_bert``
and ``mtbert_reuser``. The first component performs named entity recognition and the second performs insult detection
and sentiment analysis.

To run NER on ``multitask_bert`` component you need to add ``inference_task_names`` parameter to it. In our example
this parameter will be equal to "ner".

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

``mtbert_reuser`` component is an interface to ``multitask_bert`` ``call`` method. To use ``mtbert_reuser`` provide it
with ``multitask_bert`` object, a list of task for inference (the format is same as in ``inference_task_names``
parameter of ``multitask_bert``), and ``in_distribution`` parameter. Notice that tasks "insults" and "sentiment" are
are grouped into a list of 2 elements. Such s syntax invokes inference of these tasks in 1 call of ``tf.Session.run``.
If ``task_names`` were equal to ``["insults", "sentiment"]``, the inference of the tasks would sequential and took
approximately 2 times more time.

.. code:: json

      {
        "class_name": "mt_bert_reuser",
        "mt_bert": "#mt_bert",
        "task_names": [["insults", "sentiment"]],
        "in_distribution": {"insults": 1, "sentiment": 1},
        "in": ["bert_features", "bert_features"],
        "out": ["y_insults_pred_probas", "y_sentiment_pred_probas"]
      }

