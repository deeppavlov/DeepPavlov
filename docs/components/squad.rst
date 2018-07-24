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

Configuration
-------------

Default config could be found at ``deeppavlov/configs/squad/squad.json``

Config components
~~~~~~~~~~~~~~~~~

-  **squad\_dataset\_reader** - downloads and reads SQuAD dataset
-  data\_path - path to save dataset
-  **squad\_iterator** - create batches from SQuAD dataset
-  **squad\_preprocessor** - preprocesses context, question by cleaning
   data and tokenizing
-  in: **context\_raw**, **question\_raw** - not processed contexts and
   questions
-  out:

   -  **context** - processed context (cleaned unicode symbols and
      quoting)
   -  **context\_tokens** - tokenized context
   -  **context\_chars** - tokenized context split on chars
   -  **c\_r2p** - mapping from raw context to processed context
   -  **c\_p2r** - mapping from processed context to raw context
   -  **question** - processed question
   -  **question\_tokens** - tokenized question
   -  **question\_chars** - tokenized question split on chars
   -  **spans** - mapping from word indices to position in text

-  context\_limit - maximum length of context in words
-  question\_limit - maximum length of question in words
-  char\_limit - maximum number of chars in token
-  **squad\_ans\_preprocessor** - preprocesses answer
-  in:

   -  **ans\_raw** - not processed answer
   -  **ans\_raw\_start** - start position of not processed answer in
      context
   -  **c\_r2p**
   -  **spans**

-  out:

   -  **ans** - processed answer
   -  **ans\_start** - start position of processed answer
   -  **ans\_end** - end position of processed answer

-  **squad\_vocab\_embedder** - builds vocabulary and embedding matrix
-  in:

   -  **context\_tokens**
   -  **question\_tokens**

-  out

   -  **context\_tokens\_idxs**
   -  **question\_tokens\_idxs**

-  fit\_on: **context\_tokens** and **question\_tokens**
-  level - token or char
-  emb\_folder - path to store pretrained embeddings
-  emb\_url - url to donwload embeddings
-  save\_path - path to save vocabulary and embedding matrix
-  load\_path - path to load vocabulary and embedding matrix
-  context\_limit - maximum length of context in words
-  question\_limit - maximum length of question in words
-  char\_limit - maximum number of chars in token
-  squad\_model - model to find answer on question in context
-  in: **context\_tokens\_idxs**, **context\_chars\_idxs**,
   **question\_tokens\_idxs**, **question\_chars\_idxs**
-  in\_y: **ans\_start**, **ans\_end**
-  out:

   -  **ans\_start\_predicted** - start position of predicted answer
   -  **ans\_end\_predicted** - end position of predicted answer

-  word\_emb - pretrained word embeddings
-  char\_emb - pretrained char embeddings
-  context\_limit - maximum length of context in words
-  question\_limit - maximum length of question in words
-  char\_limit - maximum number of chars in token
-  train\_char\_emb - update char\_emb during training or not
-  char\_hidden\_size - size of word embedding built on characters
-  encoder\_hidden\_size - hidden size of encoder cells
-  attention\_hidden\_size - hidden size to use to compute attention
-  learning\_rate
-  min\_learning\_rate - minimal lr for lr decay
-  learning\_rate\_patience - patience for lr decay
-  keep\_prob - dropout keep probability
-  grad\_clip - gradient clipping value
-  weight\_decay - weight\_decay rate for exponential moving average
-  save\_path
-  load\_path
-  squad\_ans\_postprocessor - extracts predicted answer from context
-  in: **ans\_start\_predicted**, **ans\_end\_predicted**,
   **context\_raw**, **c\_p2r**, **spans**
-  out:
-  **ans\_predicted** - text of predicted answer in raw context
-  **ans\_start\_predicted** - start position of predicted answer in raw
   context
-  **ans\_end\_predicted** - end position of predicted answer in raw
   context

Running model
-------------

**Tensorflow-1.8 with GPU support is required** to run this model.

Training
--------

**Warning**: training with default config requires about 9Gb on GPU. Run
following command to train the model:

.. code:: bash

    python -m deeppavlov train deeppavlov/configs/squad/squad.json

Interact mode
-------------

Interact mode provides command line interface to already trained model.

To run model in interact mode run the following command:

``bash python -m deeppavlov interact deeppavlov/configs/squad/squad.json``
Model will ask you to type in context and question.

Pretrained models:
------------------

SQuAD
~~~~~

Pretrained model is available and can be downloaded:
http://lnsigo.mipt.ru/export/deeppavlov_data/squad_model_1.2.tar.gz

It achieves ~80 F-1 score and ~71 EM on dev set. Results of the most
recent solutions could be found on `SQuAD
Leadearboad <https://rajpurkar.github.io/SQuAD-explorer/>`__.

SDSJ Task B
~~~~~~~~~~~

Pretrained model is available and can be downloaded:
http://lnsigo.mipt.ru/export/deeppavlov_data/squad_model_ru_1.2.tar.gz

It achieves ~80 F-1 score and ~60 EM on dev set.
