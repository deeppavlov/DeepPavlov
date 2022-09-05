Multi-task BERT in DeepPavlov
=============================

Multi-task BERT in DeepPavlov is an implementation of BERT training algorithm published in the paper "Multi-Task Deep
Neural Networks for Natural Language Understanding".

| Multi-task BERT paper: https://arxiv.org/abs/1901.11504

The idea is to share BERT body between several tasks. This is necessary if a model pipe has several
components using BERT and the amount of GPU memory is limited. Each task has its own 'head' part attached to the
output of the BERT encoder. If multi-task BERT has :math:`T` heads, one training iteration consists of

- composing :math:`T` lists of examples, one for each task,

- :math:`T` gradient steps, one gradient step for each task.

By default, on every training steps lists of examples for all but one tasks are empty, as if in the original MT-DNN repository. 

When one of BERT heads is being trained, other heads' parameters do not change. On each training step both BERT head
and body parameters are modified.

Currently multitask bert heads support classification, regression, NER and multiple choice tasks. 

At this page, multi-task BERT usage is explained on a toy configuration file of a model that is trained for the single-sentence classification, sentence pair classification, regression, multiple choice and NER. The config for this nodel is :config:`multitask_example.json <configs/multitask/multitask_distilbert_example.json>`. This config is based on the single-task configs :config:`multitask_distilbert_rte.json <configs/multitask/multitask_distilbert_rte.json>` (sentence pair classification) , :config:`multitask_distilbert_sst.json <configs/multitask/multitask_distilbert_sst. json>`(single sentence classificarion), :config:`multitask_distilbert_copa.json <configs/multitask/multitask_distilbert_copa.json>`(multiple choice), :config:`multitask_distilbert_conll.json <configs/multitask/multitask_distilbert_conll.json>`(ner) , :config:`multitask_distilbert_stsb.json <configs/multitask/multitask_distilbert_stsb.json>` (regression). These single-task files show that the nodel can be used in single-task mode as well. 

Other examples of using multitask models can be found in :config:`config_glue.json <configs/multitask/config_glue.json>`.

We start with the ``metadata`` field of the configuration file.
Multi-task BERT model is saved in
``{"SAVE_LOAD_PATH": "{MODELS_PATH}/model_clean"}``. Number of train epochs is defined as ``NUM_TRAIN_EPOCHS``, number of gradient accumulation steps - as ``GRADIENT_ACC_STEPS``, backbone model to use - as ``BACKBONE`` . The metadata file is given below.
.. code:: json

  {
    "metadata": {
      "variables": {
        ROOT_PATH":"~/.deeppavlov",
         "BACKBONE":"bert-base-uncased",
         "MODELS_PATH":"{ROOT_PATH}/models_glue_clean",
         "SAVE_LOAD_PATH":"{MODELS_PATH}/model_clean",
         "NER_DATA_PATH":"~/GLUE/CONLL2003",
         "NUM_TRAIN_EPOCHS":5,
         "GRADIENT_ACC_STEPS":1
    }, 
      "download": [
        {
          "url": { "http://files.deeppavlov.ai/deeppavlov_data/ner_conll2003_v5.tar.gz", "subdir": "{MODELS_PATH}" 
        }
} 


Train config
------------

When using ``multitask_bert`` component, you can use the same inference file as the train file. 

Data reading and iteration is performed by ``multitask_reader`` and ``multitask_iterator``. These classes are composed
of task readers and iterators and generate batches that contain data from heterogeneous datasets.

A ``datset_reader`` configuration has parameters ``class_name``, ``path``, ``reader_class_name``,``task_names``, ``tasks``, ``train``, ``validation`` and ``test``
``train``,``validation``,
 and ``tasks``. ``class_name`` for multitask setting is equal to 
``multitask_reader``.
 ``path`` is a path where data, by default, are stored. This parameter can be overwriten in ``tasks`` as ``data_path``. By default, for reading data, ``reader_class_name`` is used. This
 parameter also can be overwriten in ``tasks``as ``class_name``
``data_path`` field may be any string because data paths are passed for tasks individually in ``tasks``
parameter. ``train``,``validation`` and ``test`` fields denote the train, validation and test fields of dataset, respectively. ``task_names`` is a list of tasks for which, all of the default above mentioned parameters are applied. For tasks where we want to overwrite any of these parameters, we do it in the dictionary ``tasks``. All other parameters that are specific for the certain dataset reader also need to be written there as subfields. 

The dataset_reader code is given below. 

.. code:: json


{
   "dataset_reader":{
      "class_name":"multitask_reader",
      "path":"glue",
      "reader_class_name":"huggingface_dataset_reader",
      "train":"train",
      "validation":"validation",
      "test":"test",
      "task_names":[
         "cola",
         "rte",
         "stsb"
      ],
      "tasks":{
         "copa":{
            "reader_class_name":"huggingface_dataset_reader",
            "data_path":"super_glue",
            "path":"super_glue",
            "name":"copa",
            "train":"train",
            "valid":"validation",
            "test":"test"
         },
      "conll": {
        "reader_class_name": "conll2003_reader",
        "data_path": "{NER_DATA_PATH}/conll2003/",
        "dataset_name": "conll2003",
        "provide_pos": false
      }
      }

..... START FROM HERE!!!!..... 

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




{
   "dataset_reader":{
      "class_name":"multitask_reader",
      "path":"glue",
      "reader_class_name":"huggingface_dataset_reader",
      "train":"train",
      "validation":"validation",
      "test":"test",
      "task_names":[
         "cola",
         "rte",
         "stsb"
      ],
      "tasks":{
         "copa":{
            "reader_class_name":"huggingface_dataset_reader",
            "data_path":"super_glue",
            "path":"super_glue",
            "name":"copa",
            "train":"train",
            "valid":"validation",
            "test":"test"
         },
      "conll": {
        "reader_class_name": "conll2003_reader",
        "data_path": "{NER_DATA_PATH}/conll2003/",
        "dataset_name": "conll2003",
        "provide_pos": false
      }
      }
   },
   "dataset_iterator":{
      "class_name":"multitask_iterator",
      "num_train_epochs":"{NUM_TRAIN_EPOCHS}",
      "gradient_accumulation_steps":"{GRADIENT_ACC_STEPS}",
      "iterator_class_name":"huggingface_dataset_iterator",
      "label":"label",
      "use_label_name":false,
      "seed":42,
      "tasks":{
         "cola":{
            "features":[
               "sentence"
            ]
         },
         "rte":{
            "features":[
               "sentence1",
               "sentence2"
            ]
         },
         "stsb":{
            "features":[
               "sentence1",
               "sentence2"
            ]
         },
         "copa":{
            "features":[
               "contexts",
               "choices"
            ]
         },
         "conll":{
            "iterator_class_name":"basic_classification_iterator"

         }
      }
   },
   "chainer":{
      "in":[
         "x_cola",
         "x_rte",
         "x_stsb",
         "x_copa",
         "x_conll"
      ],
      "in_y":[
         "y_cola",
         "y_rte",
         "y_stsb",
         "y_copa",
         "y_conll"
      ],
      "pipe":[
         {
            "class_name":"multitask_pipeline_preprocessor",
            "possible_keys_to_extract":[
               0,
               1
            ],
            "preprocessors":[
               "TorchTransformersPreprocessor",
               "TorchTransformersPreprocessor",
               "TorchTransformersPreprocessor",
               "TorchTransformersMultiplechoicePreprocessor",
               "TorchTransformersNerPreprocessor"
            ],
            "do_lower_case":true,
            "n_task":5,
            "vocab_file":"{BACKBONE}",
            "max_seq_length":200,
            "max_subword_length":15,
            "token_masking_prob":0.0,
            "return_features":true,
            "in":[
               "x_cola",
               "x_rte",
               "x_stsb",
               "x_copa",
               "x_conll"
            ],
            "out":[
               "bert_features_cola",
               "bert_features_rte",
               "bert_features_stsb",
               "bert_features_copa",
               "bert_features_conll"
            ]
         },
         {
            "id":"vocab_conll",
            "class_name":"simple_vocab",
            "unk_token":[
               "O"
            ],
            "pad_with_zeros":true,
            "save_path":"{MODELS_PATH}/tag.dict",
            "load_path":"{MODELS_PATH}/tag.dict",
            "fit_on":[
               "y_conll"
            ],
            "in":[
               "y_conll"
            ],
            "out":[
               "y_ids_conll"
            ]
         },
         {
            "id":"multitask_bert",
            "class_name":"multitask_bert",
            "optimizer_parameters":{
               "lr":2e-5
            },
            "gradient_accumulation_steps":"{GRADIENT_ACC_STEPS}",
            "learning_rate_drop_patience":2,
            "learning_rate_drop_div":2.0,
            "return_probas":true,
            "backbone_model":"{BACKBONE}",
            "save_path":"{SAVE_LOAD_PATH}_5",
            "load_path":"{SAVE_LOAD_PATH}_5",
            "tasks":{
               "cola":{
                  "type":"classification",
                  "options":2
               },
               "rte":{
                  "type":"classification",
                  "options":2
               },
               "stsb":{
                  "type":"regression",
                  "options":1
               },
               "copa":{
                  "type":"multiple_choice",
                  "options":2
               },
               "conll":{
                  "type":"sequence_labeling",
                  "options":"#vocab_conll.len"
               }
            },
            "in":[
               "bert_features_cola",
               "bert_features_rte",
               "bert_features_stsb",
               "bert_features_copa",
               "bert_features_conll"
            ],
            "in_y":[
               "y_cola",
               "y_rte",
               "y_stsb",
               "y_copa",
               "y_ids_conll"
            ],
            "out":[
               "y_cola_pred_probas",
               "y_rte_pred_probas",
               "y_stsb_pred",
               "y_copa_pred_probas",
               "y_conll_pred_ids"
            ]
         },
         {
            "in":[
               "y_cola_pred_probas"
            ],
            "out":[
               "y_cola_pred_ids"
            ],
            "class_name":"proba2labels",
            "max_proba":true
         },
         {
            "in":[
               "y_rte_pred_probas"
            ],
            "out":[
               "y_rte_pred_ids"
            ],
            "class_name":"proba2labels",
            "max_proba":true
         },
         {
            "in":[
               "y_copa_pred_probas"
            ],
            "out":[
               "y_copa_pred_ids"
            ],
            "class_name":"proba2labels",
            "max_proba":true
         },
         {
            "in":[
               "y_conll_pred_ids"
            ],
            "out":[
               "y_conll_pred_labels"
            ],
            "ref":"vocab_conll"
         }
      ],
      "out":[
         "y_cola_pred_ids",
         "y_rte_pred_ids",
         "y_stsb_pred",
         "y_copa_pred_ids",
         "y_conll_pred_labels"
      ]
   },
   "train":{
      "epochs":"{NUM_TRAIN_EPOCHS}",
      "batch_size":32,
      "metrics":[
         {
            "name":"multitask_accuracy",
            "inputs":[
               "y_rte",
               "y_cola",
               "y_copa",
               "y_rte_pred_ids",
               "y_cola_pred_ids",
               "y_copa_pred_ids"
            ]
         },
         {
            "name":"ner_f1",
            "inputs":[
               "y_conll",
               "y_conll_pred_labels"
            ]
         },
         {
            "name":"ner_token_f1",
            "inputs":[
               "y_conll",
               "y_conll_pred_labels"
            ]
         },
         {
            "name":"accuracy",
            "alias":"accuracy_cola",
            "inputs":[
               "y_cola",
               "y_cola_pred_ids"
            ]
         },
         {
            "name":"accuracy",
            "alias":"accuracy_rte",
            "inputs":[
               "y_rte",
               "y_rte_pred_ids"
            ]
         },
         {
            "name":"accuracy",
            "alias":"accuracy_copa",
            "inputs":[
               "y_copa",
               "y_copa_pred_ids"
            ]
         },
         {
            "name":"pearson_correlation",
            "alias":"pearson_stsb",
            "inputs":[
               "y_stsb",
               "y_stsb_pred"
            ]
         },
         {
            "name":"spearman_correlation",
            "alias":"spearman_stsb",
            "inputs":[
               "y_stsb",
               "y_stsb_pred"
            ]
         }
      ],
      "validation_patience":3,
      "val_every_n_epochs":1,
      "log_every_n_epochs":1,
      "show_examples":false,
      "evaluation_targets":[
         "valid"
      ],
      "class_name":"torch_trainer"
   },
   "metadata":{
      "variables":{
          "
      }
   }
}
