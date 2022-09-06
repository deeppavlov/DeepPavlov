Russian SuperGLUE Submission
==========================================
The DeepPavlov library provides a way to train your Russian SuperGLUE models and submit the results to the leaderboard in a couple of easy steps.

Task definition
---------------
`Russian SuperGLUE <https://russiansuperglue.com/>`__ is a benchmark that contains a set of tasks in Russian developed for evaluating general language understanding.

There are 9 tasks in the Russian SuperGLUE set:

**DaNetQA (Yes/no Question Answering Dataset for Russian)** is a binary classification task of question answering, in which the model is asked to answer a yes/no question based on a given context fragment.

**PARus (Choice of Plausible Alternatives for Russian language)** is a causal reasoning task. The model is asked to choose the most plausible alternative that has causal relation with the given premise.

**RCB (Russian Commitment Bank)** is a classification task in which the model is asked to define the type of textual entailment (Entailment, Contradiction, Neutral) between two sentences.

In the **MuSeRC (Russian Multi-Sentence Reading Comprehension)** task the model needs to process information from multiple sentences at once and identify the correct answers for the
question from the given list.

In the **RuCoS (Russian reading comprehension with Commonsense reasoning)** task the model has to choose the answer to each query from a list of text spans from a fragment.

**RUSSE (Russian Word-in-Context)** is a reading comprehension task in which the model has to identify whether a given word is used in the same
meaning in two different sentences.

In **RWSD (The Russian Winograd Schema Challenge)** the data is a set of sentences that differ by one or two words
in which syntactic ambiguity is resolved differently. The model is trained to predict whether it is resolved correctly.

**LiDiRus** is a diagnostic task in which the model has to identify whether there is entailment between two sentences.

**TERRa (Textual Entailment Recognition for Russian)** is a binary classification task of identifying whether there is entailment between two sentences.


For more detailed description of each task see `this <https://russiansuperglue.com/tasks/>`__.

Train your model
----------------
Modify the configuration file you need and train your own model for the task (see :doc:`here </intro/quick_start>` 
for more detailed instructions). The full list of models designed for each task can be found in the table below.

Create your submission files
----------------------------
To do that, use the ``submit`` command with the name of the configuration file that defines the path to your model.
Note that the name of the Russian SuperGLUE task should be defined in the ``["metadata"]["variables"]["TASK"]`` variable in the config file.

.. code:: bash

    python -m deeppavlov.utils.benchmarks.superglue <config_name> [-d] [-o <output_file_name.jsonl>]

* ``-d``: downloads model specific data before starting submission generation.
* ``-o <output_file_name.jsonl>``: set output file name. By default for Russian SuperGLUE models output filenames are
  comply with benchmark requirements.

For example, ``russian_superglue_danetqa_rubert`` solves **Yes/no Question Answering Dataset for the Russian** task.
Following command will generate ``DaNetQA.jsonl`` ready for submission:

.. code:: bash

    python -m deeppavlov.utils.benchmarks.superglue russian_superglue_danetqa_rubert -d

The prediction results will be saved in the correct format and the file will be automatically named with the name required by the system and saved to the current directory. All you have to do next 
is to zip the files you want into one archive and `submit them to leaderboard <https://russiansuperglue.com/guide/>`__.

Scores
------
The scores for DeepPavlov's pretrained models on the tasks are presented in the table.
    
+--------------------------------------------------------------------------------------------------------+----------------+-----------------+
| Model                                                                                                  |     Metric     |      Score      |
+========================================================================================================+================+=================+
|  :config:`russian_superglue_danetqa_rubert <russian_super_glue/russian_superglue_danetqa_rubert.json>` |    Accuracy    |      0.647      |
+--------------------------------------------------------------------------------------------------------+----------------+-----------------+
|  :config:`russian_superglue_parus_rubert <russian_super_glue/russian_superglue_parus_rubert.json>`     |    Accuracy    |      0.588      |
+--------------------------------------------------------------------------------------------------------+----------------+-----------------+
|  :config:`russian_superglue_russe_rubert <russian_super_glue/russian_superglue_russe_rubert.json>`     |    Accuracy    |      0.641      |
+--------------------------------------------------------------------------------------------------------+----------------+-----------------+
|  :config:`russian_superglue_lidirus_rubert <russian_super_glue/russian_superglue_lidirus_rubert.json>` | Matthew's Corr |      0.251      |
+--------------------------------------------------------------------------------------------------------+----------------+-----------------+
|  :config:`russian_superglue_rcb_rubert <russian_super_glue/russian_superglue_rcb_rubert.json>`         |     F1/Acc     |  0.336 / 0.486  |
+--------------------------------------------------------------------------------------------------------+----------------+-----------------+
|  :config:`russian_superglue_rwsd_rubert <russian_super_glue/russian_superglue_rwsd_rubert.json>`       |    Accuracy    |      0.669      |
+--------------------------------------------------------------------------------------------------------+----------------+-----------------+
|  :config:`russian_superglue_muserc_rubert <russian_super_glue/russian_superglue_muserc_rubert.json>`   |     F1a/Em     |  0.689 / 0.298  |
+--------------------------------------------------------------------------------------------------------+----------------+-----------------+
|  :config:`russian_superglue_rucos_rubert <russian_super_glue/russian_superglue_rucos_rubert.json>`     |      F1/EM     |   0.77 / 0.768  |
+--------------------------------------------------------------------------------------------------------+----------------+-----------------+
|  :config:`russian_superglue_terra_rubert <russian_super_glue/russian_superglue_terra_rubert.json>`     |    Accuracy    |      0.65       |
+--------------------------------------------------------------------------------------------------------+----------------+-----------------+
