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
-  `SDSJ Task B <https://www.sdsj.ru/ru/contest.html>`__ (RU)

Model
-----

Question Answering Model is based on R-Net, proposed by Microsoft
Research Asia (`"R-NET: Machine Reading Comprehension with Self-matching
Networks" <https://www.microsoft.com/en-us/research/publication/mrc/>`__)
and its `implementation <https://github.com/HKUST-KnowComp/R-Net>`__ by
Wenxuan Zhou.

Model documentation: :class:`~deeppavlov.models.squad.squad.SquadModel`

Configuration
-------------

Default config could be found at ``deeppavlov/configs/squad/squad.json``

Model usage
-------------

.. _reader_training:

Training
~~~~~~~~

**Warning**: training with default config requires about 9Gb on GPU. Run
following command to train the model:

.. code:: bash

    python -m deeppavlov train deeppavlov/configs/squad/squad.json

Interact mode
~~~~~~~~~~~~~

Interact mode provides command line interface to already trained model.

To run model in interact mode run the following command:

.. code:: bash

    python -m deeppavlov interact deeppavlov/configs/squad/squad.json

Model will ask you to type in context and question.

Pretrained models:
------------------

SQuAD
~~~~~

Pretrained model is available and can be downloaded:

.. code:: bash

    python -m deeppavlov download deeppavlov/configs/squad/squad.json

It achieves ~80 F-1 score and ~71 EM on `SQuAD-v1.1`_ dev set.

In the following table you can find comparison with published results. Results of the most recent competitive solutions could be found on `SQuAD
Leadearboad <https://rajpurkar.github.io/SQuAD-explorer/>`__.

+----------------------------------------------+----------------+-----------------+
| Model (single model)                         |    EM (dev)    |    F-1 (dev)    |
+----------------------------------------------+----------------+-----------------+
| :config:`DeepPavlov <squad/squad.json>`      |     71.49      |     80.34       |
+----------------------------------------------+----------------+-----------------+
| `BiDAF + Self Attention + ELMo`_             |       --       |     85.6        |
+----------------------------------------------+----------------+-----------------+
| `QANet`_                                     |     75.1       |     83.8        |
+----------------------------------------------+----------------+-----------------+
| `FusionNet`_                                 |     75.3       |     83.6        |
+----------------------------------------------+----------------+-----------------+
| `R-Net`_                                     |     71.1       |     79.5        |
+----------------------------------------------+----------------+-----------------+
| `BiDAF`_                                     |     67.7       |     77.3        |
+----------------------------------------------+----------------+-----------------+

.. _`SQuAD-v1.1`: https://arxiv.org/abs/1606.05250
.. _`BiDAF`: https://arxiv.org/abs/1611.01603
.. _`R-Net`: https://www.microsoft.com/en-us/research/publication/mrc/
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
+---------------------------------------------------------------+----------------+-----------------+
| :config:`DeepPavlov <squad/multi_squad_noans.json>`           |     57.88      |     65.91       |
+---------------------------------------------------------------+----------------+-----------------+
| `Simple and Effective Multi-Paragraph Reading Comprehension`_ |     59.14      |     67.34       |
+---------------------------------------------------------------+----------------+-----------------+
| `DrQA`_                                                       |     49.7       |     --          |
+---------------------------------------------------------------+----------------+-----------------+


.. _`DrQA`: https://arxiv.org/abs/1704.00051
.. _`Simple and Effective Multi-Paragraph Reading Comprehension`: https://arxiv.org/abs/1710.10723

SDSJ Task B
~~~~~~~~~~~

Pretrained model is available and can be downloaded:

.. code:: bash

    python -m deeppavlov download deeppavlov/configs/squad/squad_ru.json

+---------------+---------------------------------+----------------+-----------------+
| Model config                                    |    EM (dev)    |    F-1 (dev)    |
+-------------------------------------------------+----------------+-----------------+
| :config:`DeepPavlov <squad/squad_ru.json>`      |     60.62      |     80.04       |
+-------------------------------------------------+----------------+-----------------+
