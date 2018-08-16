Features
========

Components
----------

- :doc:`NER component </components/ner>`

Based on neural Named Entity Recognition network. The NER component reproduces architecture from the paper `Application
of a Hybrid Bi-LSTM-CRF model to the task of Russian Named Entity Recognition <https://arxiv.org/pdf/1709.09686.pdf>`__
which is inspired by Bi-LSTM+CRF architecture from https://arxiv.org/pdf/1603.01360.pdf.


- :doc:`Slot filling components </components/slot_filling>`

Based on fuzzy Levenshtein search to extract normalized slot values from text. The components either rely on NER results
or perform needle in haystack search.


- :doc:`Classification component </components/classifiers>`

Component for classification tasks (intents, sentiment, etc) on word-level. Shallow-and-wide CNN, Deep CNN, BiLSTM,
BiLSTM with self-attention and other models are presented. The model also allows multilabel classification of texts.
Several pre-trained models are available and presented in Table below.

+--------------------------------------------------------------------------------------------------+------------------------+-------------------------+
| Dataset                                                                                          |    Valid accuracy      |      Test accuracy      |
+--------------------------------------------------------------------------------------------------+------------------------+-------------------------+
| :config:`DSTC 2 on DSTC 2 embeddings <classifiers/intents_dstc2.json>`                           |       0.8554           |        0.8658           |
+--------------------------------------------------------------------------------------------------+------------------------+-------------------------+
| :config:`DSTC 2 on Wiki embeddings <classifiers/intents_dstc2_big.json>`                         |       0.9659           |        0.9659           |
+--------------------------------------------------------------------------------------------------+------------------------+-------------------------+
| :config:`SNIPS on DSTC 2 embeddings  <classifiers/intents_snips.json>`                           |       0.8829           |        --               |
+--------------------------------------------------------------------------------------------------+------------------------+-------------------------+
| :config:`InsultsKaggle on Reddit embeddings <classifiers/insults_kaggle.json>`                   |       0.8757           |        0.7503           |
+--------------------------------------------------------------------------------------------------+------------------------+-------------------------+
| :config:`AG News on Wiki embeddings <classifiers/topic_ag_news.json>`                            |       0.8735           |        0.8859           |
+--------------------------------------------------------------------------------------------------+------------------------+-------------------------+
| :config:`Twitter.mokoron on Russian Wiki+Lenta embeddings <classifiers/sentiment_twitter.json>`  |  0.8021 (with smiles)  |   0.7949 (with smiles)  |
+--------------------------------------------------------------------------------------------------+------------------------+-------------------------+
| :config:`Twitter.mokoron on Russian Wiki+Lenta embeddings <classifiers/sentiment_twitter.json>`  |  0.8008 (no\_smiles)   |   0.7943 (no\_smiles)   |
+--------------------------------------------------------------------------------------------------+------------------------+-------------------------+


- :doc:`Goal-oriented bot </components/go_bot>`

Based on Hybrid Code Networks (HCNs) architecture from `Jason D. Williams, Kavosh Asadi, Geoffrey Zweig, Hybrid Code
Networks: practical and efficient end-to-end dialog control with supervised and reinforcement learning – 2017
<https://arxiv.org/abs/1702.03274>`__. It allows to predict responses in goal-oriented dialog. The model is
customizable: embeddings, slot filler and intent classifier can switched on and off on demand.


- :doc:`Seq2seq goal-oriented bot </skills/seq2seq_go_bot>`

Dialogue agent predicts responses in a goal-oriented dialog and is able to handle multiple domains (pretrained bot
allows calendar scheduling, weather information retrieval, and point-of-interest navigation). The model is end-to-end
differentiable and does not need to explicitly model dialogue state or belief trackers.


- :doc:`Automatic spelling correction component </components/spelling_correction>`

Pipelines that use candidates search in a static dictionary and an ARPA language model to correct spelling errors.


- :doc:`Ranking component </components/neural_ranking>`

Based on `LSTM-based deep learning models for non-factoid answer selection <https://arxiv.org/abs/1511.04108>`__. The
model performs ranking of responses or contexts from some database by their relevance for the given context.


- :doc:`Question Answering component </components/squad>`

Based on `R-NET: Machine Reading Comprehension with Self-matching Networks
<https://www.microsoft.com/en-us/research/publication/mrc/>`__. The model solves the task of looking for an answer on a
question in a given context (`SQuAD <https://rajpurkar.github.io/SQuAD-explorer/>`__ task format).


- :doc:`Morphological tagging component </components/morphotagger>`

Based on character-based approach to morphological tagging `Heigold et al., 2017. An extensive empirical evaluation of
character-based morphological tagging for 14 languages <http://www.aclweb.org/anthology/E17-1048>`__. A state-of-the-art
model for Russian and several other languages. Model assigns morphological tags in UD format to sequences of words.


Skills
------

- :doc:`ODQA </skills/odqa>`

An open domain question answering skill. The skill accepts free-form questions about the world and outputs an answer
based on its Wikipedia knowledge.


Parameters evolution
--------------------

- :doc:`Parameters evolution for models </intro/parameters_evolution>`

Implementation of parameters evolution for DeepPavlov models that requires only some small changes in a config file.


Embeddings
----------

- :doc:`Pre-trained embeddings for the Russian language </intro/pretrained_vectors>`

Word vectors for the Russian language trained on joint `Russian Wikipedia <https://ru.wikipedia.org/>`__ and `Lenta.ru
<https://lenta.ru/>`__ corpora.


Examples of some components
---------------------------

-  Run goal-oriented bot with Telegram interface:

   ``python -m deeppavlov interactbot deeppavlov/configs/go_bot/gobot_dstc2.json -d -t <TELEGRAM_TOKEN>``
-  Run goal-oriented bot with console interface:

   ``python -m deeppavlov interact deeppavlov/configs/go_bot/gobot_dstc2.json -d``
-  Run goal-oriented bot with REST API:

   ``python -m deeppavlov riseapi deeppavlov/configs/go_bot/gobot_dstc2.json -d``
-  Run slot-filling model with Telegram interface:

   ``python -m deeppavlov interactbot deeppavlov/configs/ner/slotfill_dstc2.json -d -t <TELEGRAM_TOKEN>``
-  Run slot-filling model with console interface:

   ``python -m deeppavlov interact deeppavlov/configs/ner/slotfill_dstc2.json -d``
-  Run slot-filling model with REST API:

   ``python -m deeppavlov riseapi deeppavlov/configs/ner/slotfill_dstc2.json -d``
-  Predict intents on every line in a file:

   ``python -m deeppavlov predict deeppavlov/configs/classifiers/intents_snips.json -d --batch-size 15 < /data/in.txt > /data/out.txt``


View `video demo <https://youtu.be/yzoiCa_sMuY>`__ of deployment of a
goal-oriented bot and a slot-filling model with Telegram UI.
