================================
Frequently Asked Questions (FAQ)
================================

This is implementation of FAQ component which helps to classify incoming questions.

::

    :: What is your open hours?
    >> 8am - 8pm


Config
======

As usual, config consists of main parts:

-  **dataset_reader**
-  **dataset_iterator**
-  **chainer**

You can use you own dataset_reader, dataset_iterator for you speficic data.
Let's consider chainer in more details.

Config Structure
----------------

-  **chainer** - pipeline manager

   -  **in** - pipeline input data: question
   -  **out** - pipeline output data: answer + score[0,1]

-  **preprocessing** - it can be tokenization, lemmatization, stemming and etc. In example tfidf_logreg_autofaq.json there are tokenization and lemmatization.

-  **vectorizer** - vectorizer of incoming sentences. It can be word embeddings vectorizer, bag of words vectorizer, tf-idf vectorizer and etc. Th output is vectorized sentences (numeric vectors).

-  **faq_model** - This is faq model that classify incoming question. Model receive vectorized train sentences and vectorized question for inference. Output is classified answer from train dataset.


Vectorizers
-----------

Vectorizers produce numeric vectors of input sentences

-  **tfidf_vectorizer** - TF-IDF vectorizer
   -  **in** - input data: question
   -  **fit_on** - train data: token lemmas of question
   -  **save_path** - path where to save model
   -  **load_path** - path where to load model
   -  **out** - output data: vectorized sentence

-  **sentence2vector_v2w_tfidf** - Weighted sum of word embeddings from sentence
   -  **in** - input data: question
   -  **fit_on** - train data: [token lemmas of question, word embeddings]
   -  **save_path** - path where to save model
   -  **load_path** - path where to load model
   -  **out** - output data: vectorized sentence

-  **sentence2vector_v2w_avg** - Average sum of word embeddings from sentence
   -  **in** - input data: question
   -  **out** - output data: vectorized sentence



Faq models
----------

This is models that classify incoming question and find corresponding answer

-  **faq_cos_model** - ranking model that output answer that has maximum cosine similarity with input vectorized question

   -  **in** - input data: question
   -  **fit_on** - train data: [vectorized sentences, answers]
   -  **save_path** - path where to save model
   -  **load_path** - path where to load model
   -  **out** - output data: [answer, score]


-  **faq_logreg_model** - Logistic Regression, that output most probable answer

   -  **in** - input data: question
   -  **fit_on** - train data: [vectorized sentences, answers]
   -  **C** - regularization parameter for logistic regression model
   -  **penalty** - regularization type: 'l1' or 'l2'
   -  **save_path** - path where to save model
   -  **load_path** - path where to load model
   -  **out** - output data: [answer, score]



Running FAQ
===========


Training
--------

To train your own model by running command `train`, for example:

.. code:: bash

    cd deeppavlov/
    python deep.py train deeppavlov/configs/faq/tfidf_autofaq.json


Interacting
-----------

After model has trained, you can use it for inference: model will return answers from FAQ data that used for train.

.. code:: bash

    cd deeppavlov/
    python deep.py interact deeppavlov/configs/faq/tfidf_autofaq.json -d


Inference example:

::

    :: What is your open hours?
    >> 8am - 8pm


Available Data and Pretrained Models
====================================

You can use pretrained model by links:

.. code::

    faq_cos_model                - http://lnsigo.mipt.ru/export/faq/pretrained/faq_model.pkl
    faq_logreg_model             - http://lnsigo.mipt.ru/export/faq/pretrained/faq_logreg_model.pkl
    sentence2vector_v2w_tfidf    - http://lnsigo.mipt.ru/export/faq/pretrained/faq_sentence2vector_v2w_tfidf.pkl
    tfidf_vectorizer             - http://lnsigo.mipt.ru/export/faq/pretrained/faq_vectorizer_tfidf.pkl



-  **faq_vectorizer_tfidf.pkl** - faq_vectorizer_tfidf.pklPre trained model for TF-IDF vectorizer of sentences
-  **faq_sentence2vector_v2w_tfidf.pkl** - pre-trained model for sentence vectorizer based on weighed average of word embeddings
-  **faq_model.pkl** - pre-trained cosine similarity model for classiffing input question
-  **faq_logreg_model.pkl** - pre-trained logistic regression model for classiffing input question






