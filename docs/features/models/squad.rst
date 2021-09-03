Question Answering Model for SQuAD dataset
==========================================

Task definition
---------------

Question Answering on SQuAD dataset is a task to find an answer on
question in a given context (e.g, paragraph from Wikipedia), where the
answer to each
question is a segment of the context:

Context:

    In meteorology, precipitation is any product of the condensation of
    atmospheric water vapor that falls under gravity. The main forms of
    precipitation include drizzle, rain, sleet, snow, graupel and
    hail... Precipitation forms as smaller droplets coalesce via
    collision with other rain drops or ice crystals **within a cloud**.
    Short, intense periods of rain in scattered locations are called
    “showers”.

Question:

    Where do water droplets collide with ice crystals to form
    precipitation?

Answer:

    within a cloud

Datasets, which follow this task format:

-  Stanford Question Answering Dataset
   (`SQuAD <https://rajpurkar.github.io/SQuAD-explorer/>`__) (EN)
-  `SDSJ Task B <https://sdsj.sberbank.ai/2017/ru/contest.html>`__ (RU)

Models
------

There are two models for this task in DeepPavlov: BERT-based and R-Net. Both models predict answer start and end
position in a given context.
Their performance is compared in :ref:`pretrained models <pretrained_models>` section of this documentation.

BERT
~~~~
Pretrained BERT can be used for Question Answering on SQuAD dataset just by applying two linear transformations to
BERT outputs for each subtoken. First/second linear transformation is used for prediction of probability that current
subtoken is start/end position of an answer.

BERT for SQuAD model documentation on TensorFlow :class:`~deeppavlov.models.bert.bert_squad.BertSQuADModel`
and on PyTorch :class:`~deeppavlov.models.torch_bert.torch_transformers_squad:TorchTransformersSquad`.

R-Net
~~~~~

Question Answering Model is based on R-Net, proposed by Microsoft
Research Asia (`"R-NET: Machine Reading Comprehension with Self-matching
Networks" <https://www.microsoft.com/en-us/research/publication/mcr/>`__)
and its `implementation <https://github.com/HKUST-KnowComp/R-Net>`__ by
Wenxuan Zhou.

R-Net for SQuAD model documentation: :class:`~deeppavlov.models.squad.squad.SquadModel`

Configuration
-------------

Default configs could be found in :config:`deeppavlov/configs/squad/ <squad/>` folder.

Prerequisites
-------------

Before using the model make sure that all required packages are installed running the command for TensorFlow:

.. code:: bash

    python -m deeppavlov install squad_bert

and for PyTorch

.. code:: bash

    python -m deeppavlov install squad_torch_bert


By running this command we will install requirements for
:config:`deeppavlov/configs/squad/squad_bert.json <squad/squad_bert.json>` or for
:config:`deeppavlov/configs/squad/squad_torch_bert.json <squad/squad_torch_bert.json>`

Model usage from Python
-----------------------

.. code:: python

    from deeppavlov import build_model, configs

    model = build_model(configs.squad.squad, download=True)
    model(['DeepPavlov is library for NLP and dialog systems.'], ['What is DeepPavlov?'])


Model usage from CLI
--------------------

.. _reader_training:

Training
~~~~~~~~

**Warning**: training with default config requires about 10Gb on GPU. Run
following command to train the model:

.. code:: bash

    python -m deeppavlov train deeppavlov/configs/squad/squad_bert.json

Interact mode
~~~~~~~~~~~~~

Interact mode provides command line interface to already trained model.

To run model in interact mode run the following command:

.. code:: bash

    python -m deeppavlov interact deeppavlov/configs/squad/squad_bert.json

Model will ask you to type in context and question.

.. _pretrained_models:

Pretrained models:
------------------

SQuAD
~~~~~

We have all pretrained model available to download:

.. code:: bash

    python -m deeppavlov download deeppavlov/configs/squad/squad_bert.json

It achieves ~88 F-1 score and ~80 EM on `SQuAD-v1.1`_ dev set.

In the following table you can find comparison with published results. Results of the most recent competitive solutions could be found on `SQuAD
Leadearboad <https://rajpurkar.github.io/SQuAD-explorer/>`__.

+---------------------------------------------------------+----------------+-----------------+
| Model (single model)                                    |    EM (dev)    |    F-1 (dev)    |
+=========================================================+================+=================+
| :config:`DeepPavlov BERT <squad/squad_bert.json>`       |     80.88      |     88.49       |
+---------------------------------------------------------+----------------+-----------------+
| :config:`BERT on PyTorch <squad/squad_torch_bert.json>` |     78.8       |     86.7        |
+---------------------------------------------------------+----------------+-----------------+
| :config:`DeepPavlov R-Net <squad/squad.json>`           |     71.49      |     80.34       |
+---------------------------------------------------------+----------------+-----------------+
| `BiDAF + Self Attention + ELMo`_                        |       --       |     85.6        |
+---------------------------------------------------------+----------------+-----------------+
| `QANet`_                                                |     75.1       |     83.8        |
+---------------------------------------------------------+----------------+-----------------+
| `FusionNet`_                                            |     75.3       |     83.6        |
+---------------------------------------------------------+----------------+-----------------+
| `R-Net`_                                                |     71.1       |     79.5        |
+---------------------------------------------------------+----------------+-----------------+
| `BiDAF`_                                                |     67.7       |     77.3        |
+---------------------------------------------------------+----------------+-----------------+

.. _`SQuAD-v1.1`: https://arxiv.org/abs/1606.05250
.. _`BiDAF`: https://arxiv.org/abs/1611.01603
.. _`R-Net`: https://www.microsoft.com/en-us/research/publication/mcr/
.. _`FusionNet`: https://arxiv.org/abs/1711.07341
.. _`QANet`: https://arxiv.org/abs/1804.09541
.. _`BiDAF + Self Attention + ELMo`: https://arxiv.org/abs/1802.05365

SQuAD with contexts without correct answers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the case when answer is not necessary present in given context we have :config:`squad_noans <squad/multi_squad_noans.json>`
config with pretrained model. This model outputs empty string in case if there is no answer in context.
This model was trained not on SQuAD dataset. For each question-context pair from SQuAD we extracted contexts from the same
Wikipedia article and ranked them according to tf-idf score between question and context. In this manner we built dataset
with contexts without an answer.

Special trainable `no_answer` token is added to output of self-attention layer and it makes model able to select
`no_answer` token in cases, when answer is not present in given context.

We got 57.88 EM and 65.91 F-1 on ground truth Wikipedia article (we used the same Wiki dump as `DrQA`_):

+---------------+-----------------------------------------------+----------------+-----------------+
| Model config                                                  |    EM (dev)    |    F-1 (dev)    |
+===============================================================+================+=================+
| :config:`DeepPavlov <squad/multi_squad_noans.json>`           |     57.88      |     65.91       |
+---------------------------------------------------------------+----------------+-----------------+
| `Simple and Effective Multi-Paragraph Reading Comprehension`_ |     59.14      |     67.34       |
+---------------------------------------------------------------+----------------+-----------------+
| `DrQA`_                                                       |     49.7       |     --          |
+---------------------------------------------------------------+----------------+-----------------+

Pretrained model is available and can be downloaded (~2.5Gb):

.. code:: bash

    python -m deeppavlov download deeppavlov/configs/squad/multi_squad_noans.json


.. _`DrQA`: https://arxiv.org/abs/1704.00051
.. _`Simple and Effective Multi-Paragraph Reading Comprehension`: https://arxiv.org/abs/1710.10723

SDSJ Task B
~~~~~~~~~~~

Pretrained models are available and can be downloaded:

.. code:: bash

    python -m deeppavlov download deeppavlov/configs/squad/squad_ru.json

    python -m deeppavlov download deeppavlov/configs/squad/squad_ru_rubert_infer.json

    python -m deeppavlov download deeppavlov/configs/squad/squad_ru_bert_infer.json

Link to SDSJ Task B dataset: http://files.deeppavlov.ai/datasets/sber_squad-v1.1.tar.gz

+------------------------------------------------------------------------+----------------+-----------------+
| Model config                                                           |    EM (dev)    |    F-1 (dev)    |
+========================================================================+================+=================+
| :config:`DeepPavlov RuBERT <squad/squad_ru_rubert_infer.json>`         |   66.30+-0.24  |    84.60+-0.11  |
+------------------------------------------------------------------------+----------------+-----------------+
| :config:`DeepPavlov multilingual BERT <squad/squad_ru_bert_infer.json>`|   64.35+-0.39  |    83.39+-0.08  |
+------------------------------------------------------------------------+----------------+-----------------+
| :config:`DeepPavlov R-Net <squad/squad_ru.json>`                       |     60.62      |     80.04       |
+------------------------------------------------------------------------+----------------+-----------------+


DRCD
~~~~~~~~~~~

Pretrained models are available and can be downloaded:

.. code:: bash

    python -m deeppavlov download deeppavlov/configs/squad/squad_zh_bert.json
    python -m deeppavlov download deeppavlov/configs/squad/squad_zh_zh_bert.json
	
Link to DRCD dataset: http://files.deeppavlov.ai/datasets/DRCD.tar.gz
Link to DRCD paper: https://arxiv.org/abs/1806.00920

+------------------------------------------------------------------------+----------------+-----------------+
| Model config                                                           |    EM (dev)    |    F-1 (dev)    |
+========================================================================+================+=================+
| :config:`DeepPavlov ChineseBERT <squad/squad_zh_bert_zh.json>`         |   84.19        |    89.23        |
+------------------------------------------------------------------------+----------------+-----------------+
| :config:`DeepPavlov multilingual BERT <squad/squad_zh_bert_mult.json>` |   84.86        |    89.03        |
+------------------------------------------------------------------------+----------------+-----------------+
