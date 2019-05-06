BERT in DeepPavlov
==================
BERT (Bidirectional Encoder Representations from Transformers) is a Transformer pre-trained on masked language model
and next sentence prediction tasks. This approach showed state-of-the-art results on a wide range of NLP tasks in
English.

| BERT paper: https://arxiv.org/abs/1810.04805
| Google Research BERT repository: https://github.com/google-research/bert

There are several pre-trained BERT models released by Google Research, more detail about these pretrained models could be found here https://github.com/google-research/bert#pre-trained-models:

-  BERT-base, English, cased, 12-layer, 768-hidden, 12-heads, 110M parameters: download from `[google] <https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip>`__, `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/cased_L-12_H-768_A-12.zip>`__
-  BERT-base, English, uncased, 12-layer, 768-hidden, 12-heads, 110M parameters: download from `[google] <https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip>`__, `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/uncased_L-12_H-768_A-12.zip>`__
-  BERT-large, English, cased, 24-layer, 1024-hidden, 16-heads, 340M parameters: download from `[google] <https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip>`__
-  BERT-base, multilingual, cased, 12-layer, 768-hidden, 12-heads, 110M parameters: download from `[google] <https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip>`__, `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/multi_cased_L-12_H-768_A-12.zip>`__
-  BERT-base, Chinese, cased, 12-layer, 768-hidden, 12-heads, 110M parameters: download from `[google] <https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip>`__

We have trained BERT-base for Russian Language:

-  RuBERT, Russian, cased, 12-layer, 768-hidden, 12-heads, 110M parameters: `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_v1.tar.gz>`__

RuBERT was trained on the Russian part of Wikipedia and news data. We used this training data to build vocabulary of Russian subtokens and took
multilingual version of BERT-base as initialization for RuBERT.

Here, in DeepPavlov, we made it easy to use pre-trained BERT for downstream tasks like classification, tagging, question answering and
ranking. We provide pre-trained models and examples on how to use BERT with DeepPavlov.

BERT for Classification
-----------------------

**deeppavlov.models.bert.BertClassifierModel** (see :doc:`here </apiref/models/bert>`) provides easy to use
solution for classification problem using pre-trained BERT.
One can use several pre-trained English, multi-lingual and Russian BERT models that are listed above.

Two main components of BERT classifier pipeline in DeepPavlov are
``deeppavlov.models.preprocessors.BertPreprocessor`` (see :doc:`here </apiref/models/bert>`)
and ``deeppavlov.models.bert.BertClassifierModel`` (see :doc:`here </apiref/models/bert>`).
Non-processed texts should be given to ``bert_preprocessor`` for tokenization on subtokens,
encoding subtokens with their indices and creating tokens and segment masks.
If one processed classes to one-hot labels in pipeline, ``one_hot_labels`` should be set to ``true``.

``bert_classifier`` has a dense layer of number of classes size upon pooled outputs of Transformer encoder,
it is followed by ``softmax`` activation (``sigmoid`` if ``multilabel`` parameter is set to ``true`` in config).


BERT for Named Entity Recognition (Sequence Tagging)
----------------------------------------------------

Pre-trained BERT model can be used for sequence tagging. Examples of usage of BERT for sequence tagging can be
found :doc:`here </components/ner>`. The module used for tagging is :class:`~deeppavlov.models.bert.bert_ner.BertNerModel`.
To tag each word representations of the first sub-word elements are extracted. So for each word there is only one vector produced.
These representations are passed to a dense layer or Bi-RNN layer to produce distribution over tags. There is
also an optional CRF layer on the top.

Multilingual BERT models allows to perform zero-shot transfer across languages. To use our 19 tags NER for over a
hundred languages see :ref:`ner_multi_bert`.


BERT for Context Question Answering (SQuAD)
-------------------------------------------
Context Question Answering on `SQuAD <https://rajpurkar.github.io/SQuAD-explorer/>`__ dataset is a task
of looking for an answer on a question in a given context. This task could be formalized as predicting answer start
and end position in a given context. :class:`~deeppavlov.models.bert.bert_squad.BertSQuADModel` uses two linear
transformations to predict probability that currents subtoken is start/end position of an answer. For details check
:doc:`Context Question Answering documentation page </components/squad>`.

BERT for Ranking
----------------
There are two main approaches in text ranking. The first one is interaction-based which is relatively accurate but
works slow and the second one is representation-based which is less accurate but faster [1]_.
The interaction-based ranking based on BERT is represented in the DeepPavlov with two main components
:class:`~deeppavlov.models.preprocessors.bert_preprocessor.BertRankerPreprocessor`
and :class:`~deeppavlov.models.bert.bert_ranker.BertRankerModel`
and the representation-based ranking with components
:class:`~deeppavlov.models.preprocessors.bert_preprocessor.BertSepRankerPreprocessor`
and :class:`~deeppavlov.models.bert.bert_ranker.BertSepRankerModel`.
Additional components
:class:`~deeppavlov.models.preprocessors.bert_preprocessor.BertSepRankerPredictorPreprocessor`
and :class:`~deeppavlov.models.bert.bert_ranker.BertSepRankerPredictor` are for usage in the ``interact`` mode
where the task for ranking is to retrieve the best possible response from some provided response base with the help of
the trained model. Working examples with the trained models are given :doc:`here </components/neural_ranking>`.
Statistics are available :doc:`here </intro/features>`.

.. [1] McDonald, R., Brokos, G. I., & Androutsopoulos, I. (2018). Deep relevance ranking using enhanced document-query interactions. arXiv preprint arXiv:1809.01682.