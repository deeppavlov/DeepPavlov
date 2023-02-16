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
``{"SAVE_LOAD_PATH": "{MODELS_PATH}/model_clean"}``. Number of train epochs is defined as ``NUM_TRAIN_EPOCHS``, number of gradient accumulation steps - as ``GRADIENT_ACC_STEPS``, backbone model to use - as ``BACKBONE`` . The metadata file for config:`multitask_example.json <configs/multitask/multitask_distilbert_example.json>` is given below.

.. code:: json

		{
	"metadata": {
		"variables": {
		"ROOT_PATH":"~/.deeppavlov",
		"BACKBONE":"bert-base-uncased",
		"MODELS_PATH":"{ROOT_PATH}/models_glue_clean",
		"SAVE_LOAD_PATH":"{MODELS_PATH}/model_clean",
		"NER_DATA_PATH":"~/GLUE/CONLL2003",
		"NUM_TRAIN_EPOCHS":5,
		"GRADIENT_ACC_STEPS":1
		}, 
	"download": [{
		"url": "http://files.deeppavlov.ai/deeppavlov_data/ner_conll2003_v5.tar.gz",
		"subdir": "{MODELS_PATH}" }]
		}
		}


Train config
------------

When using ``multitask_bert`` component, you can use the same inference file as the train file. 

Data reading and iteration is performed by ``multitask_reader`` and ``multitask_iterator``. These classes are composed
of task readers and iterators and generate batches that contain data from heterogeneous datasets.

The ``datset_reader`` configuration has parameters ``class_name``, ``path``, ``reader_class_name``,``task_names``, ``tasks``, ``train``, ``validation`` and ``test``.
``train``,``validation``, and ``tasks``. ``class_name`` for multitask setting is equal to ``multitask_reader``. Also ``path`` is a path where data, by default, are stored. This parameter can be overwriten in ``tasks`` as ``data_path``. By default, for reading data, ``reader_class_name`` is used. This parameter also can be overwriten in ``tasks``as ``class_name``
The ``data_path`` field may be any string because data paths are passed for tasks individually in ``tasks``
parameter. ``train``,``validation`` and ``test`` fields denote the train, validation and test fields of dataset, respectively. ``task_names`` is a list of tasks for which, all of the default above mentioned parameters are applied. For tasks where we want to overwrite any of these parameters, we do it in the dictionary ``tasks``. All other parameters that are specific for the certain dataset reader also need to be written there as subfields. 

The dataset_reader code for config:`multitask_example.json <configs/multitask/multitask_distilbert_example.json>` is given below. 

.. code:: json

{
   "dataset_reader":{
      "class_name":"multitask_reader",
      "path":"glue",
      "reader_class_name":"huggingface_dataset_reader",
      "train":"train",
      "validation":"validation",
      "test":"test",
      "task_names":["cola", "rte", "stsb"],
      "tasks":
      {"copa":
      {"reader_class_name":"huggingface_dataset_reader",
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


A ``multitask_iterator`` configuration  has the main parameters ``class_name`` and ``tasks``. ``tasks`` is a dictionary of
configurations of task iterators. In configurations of task iterators, ``iterator_class_name`` is used instead of
``class_name``. This parameter denotes the name of iterator for every task. Auhillary parameters are ``num_train_epochs``, ``gradient_accumulation_steps``(which denote train epoch number and number of gradient accumulation steps respectively). Name of label is denoted by ``label`` parameter. Parameters for ``iteraror_class_name``, such as ``use_label_name`` and ``seed``, can also be handed down there. If some parameters need to be defined or redefined for every task (e.g features), these parameters will be redefined as values in ``tasks`` dictionary.  Instead of using this dictionary, you can put all task names into the task_names field if all parameters for these tasks are the same.
NOTE THAT ORDER OF TASKS HANDED IN ALL NEXT COMPONENTS OF TRAINER MATTERS(IF THE COMPONENT GETS VARIABLES FOR MORE THAN 1 TASK)!!!!
The dataset iterator configuration for config:`multitask_example.json <configs/multitask/multitask_distilbert_example.json>` is as follows:

.. code:: json

{
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
	      }

    
Batches generated by ``multitask_iterator`` are tuples of two elements: inputs of the model and labels. 
Both inputsand labels are lists of tuples. The inputs have following format: ``[(first_task_inputs[0], second_task_inputs[0],...), (first_task_inputs[1], second_task_inputs[1], ...), ...]`` where ``first_task_inputs``, ``second_task_inputs``, and so on are x values of batches from task dataset iterators. The labels in the second element have the similar format.

If task datasets have different sizes, then for smaller datasets the lists are padded with ``None``s. For example, if the first task dataset inputs are
``[0, 1, 2, 3, 4, 5, 6]``, the second task dataset inputs are ``[7, 8, 9]``, and the batch size is ``2``, then
multi-task input mini-batches will be ``[(0, 7), (1, 8)]``, ``[(2, 9), (3, None)]``, ``[(4, None), (5, None)]``, ``[(6, None)]``.

In this tutorial, there are 5 datasets. Considering the batch structure, ``chainer`` inputs in config:`multitask_example.json <configs/multitask/multitask_distilbert_example.json>` are:

.. code:: json

{
	"in":["x_cola",  "x_rte", "x_stsb", "x_copa",  "x_conll"],
	"in_y":["y_cola", "y_rte", "y_stsb", "y_copa",  "y_conll"]
        }

Sometimes a task dataset iterator returns inputs or labels consisting of more than one element. For example, in model
:config:`mt_bert_train_tutorial.json <kbqa/kbqa_mt_bert_train.json>` ``siamese_iterator`` input
element consists of 2 strings. If there is a necessity to split such a variable, ``InputSplitter`` component can
be used.
Data preparation in the multitask setting can be similar to the preparation in singletask setting except for the names of the variables. 

For streamlining the code, however, ``input_splitter``and ``tokenizer`` can be unified into the ``multitask_pipeline_preprocessor``. This preprocessor gets as a parameter ``preprocessor`` the one preprocessor class name for all tasks, or gets the preprocessor name list as a parameter ``preprocessors``. After splitting input by ``possible_keys_to_extract``, every preprocessor (being initialized by the input beforehand) processes the input. Note, that if ``strict`` parameter(default:False) is set to True, we always try to split data. Here is the definition of ``multitask_pipeline_preprocessor`` from the config:`multitask_example.json <configs/multitask/multitask_distilbert_example.json>`.

..code:: json
         {
            "class_name":"multitask_pipeline_preprocessor",
            "possible_keys_to_extract":[0, 1],
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
            "in":["x_cola", "x_rte", "x_stsb", "x_copa", "x_conll"],
            "out":[
            "bert_features_cola",
            "bert_features_rte",
            "bert_features_stsb",
            "bert_features_copa",
            "bert_features_conll"
            ]
            }


The multitask_bert component has common and task_specific parameters. Shared parameters are provided inside the tasks parameter.
The tasks is a dictionary that keys are task names and values are task-specific parameters(type, options).
Common parameters, are backbone_model(same parameter as in the tokenizer) and all parameters from torch_bert. 
**The order of tasks MATTERS.**


Here is the definition of ``multitask_bert`` from the config:`multitask_example.json <configs/multitask/multitask_distilbert_example.json>`.
.. code:: json

        {
            "id":"multitask_bert",
            "class_name":"multitask_bert",
            "optimizer_parameters":{"lr":2e-5},
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
            }
         
Note that ``proba2labels`` can now take several arguments. 

.. code:: json

        {
            "in":[
            "y_cola_pred_probas", 
            "y_rte_pred_probas", 
            "y_copa_pred_probas"             
            ],
            "out":[
            "y_cola_pred_ids", 
            "y_rte_pred_ids", 
            "y_copa_pred_ids" 
            ],
            "class_name":"proba2labels",
            "max_proba":true
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


You can make an inference-only config. In this config, there is no need in dataset reader and dataset iterator. A ``train`` field and components
preparing ``in_y`` are removed. In ``multitask_bert`` component configuration all training parameters (learning rate,
optimizer, etc.) are omitted.

Here are the results of ``deeppavlov/configs/multitask/glue.json`` compared to the analogous singletask configs, according to the test server.
+-------------------+-------------+----------------+----------+---------------+-----------------------+---------------+------------+----------+----------+----------------+
| Task              | Score       | CoLA           | SST-2    | MRPC          | STS-B                 | QQP           | MNLI(m/mm) | QNLI     | RTE      | AX             |
+===================+=============+================+==========+===============+=======================+===============+============+==========+==========+================+
| Metric            | from server | Matthew's Corr | Accuracy | F1 / Accuracy | Pearson/Spearman Corr | F1 / Accuracy | Accuracy   | Accuracy | Accuracy | Matthew's Corr |
| Multitask config  | 77.8        | 43.6           | 93.2     | 88.6/84.2     | 84.3/84.0             | 70.1/87.9     | 83.0/82.6  | 90.6     | 75.4     | 35.4           |
| Singletask config | 77.6        | 53.6           | 92.7     | 87.7/83.6     | 84.4/83.1             | 70.5/88.9     | 84.4/83.2  | 90.3     | 63.4     | 36.3           |
+-------------------+-------------+----------------+----------+---------------+-----------------------+---------------+------------+----------+----------+----------------+


Here are the same results for SuperGLUE tasks.

+-------------------+--------------+-------------------+-----------+------------+-----------+-----------+-----------+-----------+-----------------+
| Task              | Score        | CB                | COPA      | MultiRC    | RTE       | WiC       | WSC       | BoolQ     | AX              |
+-------------------+--------------+-------------------+-----------+------------+-----------+-----------+-----------+-----------+-----------------+
| Metric            | from server  | Avg.F1/ Accuracy  | Accuracy  | F1a/EM     | Accuracy  | Accuracy  | Accuracy  | Accuracy  | Matthew's Corr  |
+===================+==============+===================+===========+============+===========+===========+===========+===========+=================+
| Multitask config  | 53.0         | 82.2/85.6         | 61.8      | 60.4/13.2  | 57.9      | 63.0      | 62.3      | 66.7      | 9.4             |
+-------------------+--------------+-------------------+-----------+------------+-----------+-----------+-----------+-----------+-----------------+
| Singletask config | 58.8         | 82.2/85.6         | 68.6      | 59.4/14.7  | 67.5      | 68.1      | 58.2      | 74.2      | 20.2            |
+-------------------+--------------+-------------------+-----------+------------+-----------+-----------+-----------+-----------+-----------------+
