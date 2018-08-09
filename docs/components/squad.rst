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

It achieves ~80 F-1 score and ~71 EM on dev set. Results of the most
recent solutions could be found on `SQuAD
Leadearboad <https://rajpurkar.github.io/SQuAD-explorer/>`__.

SDSJ Task B
~~~~~~~~~~~

Pretrained model is available and can be downloaded:

.. code:: bash

    python -m deeppavlov download deeppavlov/configs/squad/squad_ru.json

It achieves ~80 F-1 score and ~60 EM on dev set.
