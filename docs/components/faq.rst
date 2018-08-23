================================
Frequently Asked Questions (FAQ)
================================

This is implementation of FAQ component which helps to classify incoming questions.

::

    :: What is your open hours?
    >> 8am - 8pm


Config
======

As usual, config consists of:

-  **dataset_reader**
-  **dataset_iterator**
-  **chainer**

You can use you own dataset_reader, dataset_iterator for speficic data.
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
   -  **fit_on** - train data:   token lemmas of question
   -  **is_pretrained** - do you use pretrained model? If so there is no need to train vectorizer
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
   -  **c** - regularization parameter for logistic regression model
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
    python deep.py train configs/faq/tfidf_autofaq.json


Interacting
-----------

After model has trained, you can use it for inference: model will return answers from FAQ data that used for train.

.. code:: bash

    cd deeppavlov/
    python deep.py interact configs/faq/tfidf_autofaq.json -d


Inference example:

::

    :: What is your open hours?
    >> 8am - 8pm


Available Data and Pretrained Models
====================================

You can use pretrained model on FAQ dataset from school-site: http://www.ftl.name/page/989

.. code::

    faq_vectorizer_tfidf            - http://files.deeppavlov.ai/faq/faq_vectorizer_tfidf.pkl
    faq_tfidf_cos_model             - http://files.deeppavlov.ai/faq/faq_tfidf_cos_model.pkl
    faq_tfidf_logreg_model          - http://files.deeppavlov.ai/faq/faq_tfidf_logreg_model.pkl
    faq_fasttext_cos_model          - http://files.deeppavlov.ai/faq/faq_fasttext_cos_model.pkl
    faq_sentence2vector_v2w_tfidf   - http://files.deeppavlov.ai/faq/faq_sentence2vector_v2w_tfidf.pkl


-  **faq_vectorizer_tfidf.pkl** - pre-trained model for TF-IDF vectorizer of sentences
-  **faq_tfidf_cos_model.pkl** - pre-trained cosine similarity model for classifying input question(vectorized by tfidf)
-  **faq_tfidf_logreg_model.pkl** - pre-trained logistic regression model for classifying input question(vectorized by tfidf)
-  **faq_fasttext_cos_model.pkl** - pre-trained cosine similarity model for classifying input question(vectorized by word embeddings)
-  **faq_sentence2vector_v2w_tfidf.pkl** - pre-trained model for sentence vectorizer based on weighed average of word embeddings



