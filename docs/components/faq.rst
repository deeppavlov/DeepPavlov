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

-  **classifier** - This is faq model that classify incoming question. Model receive vectorized train sentences and vectorized question for inference. Output is classified answer from train dataset.


Vectorizers
-----------

Vectorizers produce numeric vectors of input sentences

-  **tfidf_vectorizer** - TF-IDF vectorizer

   -  **in** - input data: question
   -  **fit_on** - train data:   token lemmas of question
   -  **save_path** - path where to save model
   -  **load_path** - path where to load model
   -  **out** - output data: vectorized sentence

-  **sentence2vector_v2w_tfidf** - Sentence vectorizer: weighted sum of word embeddings from sentence

   -  **in** - input data: question
   -  **fit_on** - train data: [token lemmas of question, word embeddings]
   -  **save_path** - path where to save model
   -  **load_path** - path where to load model
   -  **out** - output data: vectorized sentence

-  **sentence2vector_v2w_avg** - Sentence vectorizer: average sum of word embeddings from sentence
   -  **in** - input data: question
   -  **out** - output data: vectorized sentence



Classifiers for FAQ
-------------------

This is models that classify incoming question and find corresponding answer

-  **cos_sim_classifier** - Classifier based on cosine similarity

   -  **in** - input data: question
   -  **fit_on** - train data: [vectorized sentences, answers]
   -  **save_path** - path where to save model
   -  **load_path** - path where to load model
   -  **out** - output data: [answer, score]


-  **logreg_classifier** - Logistic Regression classifier, that output most probable answer with score

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

As an example you can try pretrained models on FAQ dataset in English: MIPT FAQ for entrants - https://mipt.ru/english/edu/faqs/


   ::

    tfidf_logreg_classifier_en_mipt_faq  - http://files.deeppavlov.ai/faq/mipt/tfidf_logreg_classifier_en_mipt_faq.pkl
    tfidf_vectorizer_en_mipt_faq         - http://files.deeppavlov.ai/faq/mipt/tfidf_vectorizer_en_mipt_faq.pkl


-  **tfidf_logreg_classifier_en_mipt_faq.pkl**  - pre-trained logistic regression classifier for classifying input question (vectorized by tfidf)
-  **tfidf_vectorizer_en_mipt_faq.pkl**         - pre-trained model for TF-IDF vectorizer based on MIPT FAQ

Example config - :download:`deeppavlov/configs/faq/tfidf_logreg_en_faq.json <../../deeppavlov/configs/faq/tfidf_logreg_en_faq.json>`


Also you can use pretrained model on Russan FAQ dataset from school-site: http://www.ftl.name/page/989

   ::

    tfidf_cos_sim_classifier    - http://files.deeppavlov.ai/faq/school/faq_tfidf_cos_model.pkl
    tfidf_logreg_classifier     - http://files.deeppavlov.ai/faq/school/faq_tfidf_logreg_model.pkl
    fasttext_cos_classifier     - http://files.deeppavlov.ai/faq/school/faq_fasttext_cos_model.pkl
    tfidf_vectorizer_ruwiki     - http://files.deeppavlov.ai/vectorizer/tfidf_vectorizer_ruwiki.pkl


-  **tfidf_cos_sim_classifier.pkl** - pre-trained cosine similarity classifier for classifying input question (vectorized by tfidf)
-  **tfidf_logreg_classifier.pkl**  - pre-trained logistic regression classifier for classifying input question (vectorized by tfidf)
-  **fasttext_cos_classifier.pkl**  - pre-trained cosine similarity classifier for classifying input question (vectorized by word embeddings)
-  **tfidf_vectorizer_ruwiki.pkl**  - pre-trained model for TF-IDF vectorizer based on Russian Wikipedia



