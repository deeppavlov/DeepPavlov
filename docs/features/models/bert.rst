BERT in DeepPavlov
==================
BERT (Bidirectional Encoder Representations from Transformers) is a Transformer pre-trained on masked language model
and next sentence prediction tasks. This approach showed state-of-the-art results on a wide range of NLP tasks in
English.

| BERT paper: https://arxiv.org/abs/1810.04805
| Google Research BERT repository: https://github.com/google-research/bert

There are several pre-trained BERT models released by Google Research, more details about these pre-trained models could be found here: https://github.com/google-research/bert#pre-trained-models

-  BERT-base, English, cased, 12-layer, 768-hidden, 12-heads, 110M parameters: download from `[google] <https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip>`__,
   `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/cased_L-12_H-768_A-12.zip>`__
-  BERT-base, English, uncased, 12-layer, 768-hidden, 12-heads, 110M parameters: download from `[google] <https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip>`__,
   `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/uncased_L-12_H-768_A-12.zip>`__
-  BERT-large, English, cased, 24-layer, 1024-hidden, 16-heads, 340M parameters: download from `[google] <https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip>`__
-  BERT-base, multilingual, cased, 12-layer, 768-hidden, 12-heads, 180M parameters: download from `[google] <https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip>`__,
   `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/multi_cased_L-12_H-768_A-12.zip>`__, `[deeppavlov_pytorch] <http://files.deeppavlov.ai/deeppavlov_data/bert/multi_cased_L-12_H-768_A-12_pt.tar.gz>`__
-  BERT-base, Chinese, cased, 12-layer, 768-hidden, 12-heads, 110M parameters: download from `[google] <https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip>`__,
   `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/chinese_L-12_H-768_A-12.zip>`__, `[deeppavlov_pytorch] <http://files.deeppavlov.ai/deeppavlov_data/bert/chinese_L-12_H-768_A-12_pt.tar.gz>`__

We have trained BERT-base model for other languages and domains:

-  RuBERT, Russian, cased, 12-layer, 768-hidden, 12-heads, 180M parameters: `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_v2.tar.gz>`__,
   `[deeppavlov_pytorch] <http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_pt_v1.tar.gz>`__
-  SlavicBERT, Slavic (bg, cs, pl, ru), cased, 12-layer, 768-hidden, 12-heads, 180M parameters: `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/bg_cs_pl_ru_cased_L-12_H-768_A-12_v1.tar.gz>`__,
   `[deeppavlov_pytorch] <http://files.deeppavlov.ai/deeppavlov_data/bert/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt_v1.tar.gz>`__
-  Conversational BERT, English, cased, 12-layer, 768-hidden, 12-heads, 110M parameters: `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/conversational_cased_L-12_H-768_A-12_v1.tar.gz>`__,
   `[deeppavlov_pytorch] <http://files.deeppavlov.ai/deeppavlov_data/bert/conversational_cased_L-12_H-768_A-12_pt_v1.tar.gz>`__
-  Conversational RuBERT, Russian, cased, 12-layer, 768-hidden, 12-heads, 180M parameters: `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/ru_conversational_cased_L-12_H-768_A-12.tar.gz>`__,
   `[deeppavlov_pytorch] <http://files.deeppavlov.ai/deeppavlov_data/bert/ru_conversational_cased_L-12_H-768_A-12_pt_v1.tar.gz>`__
-  Conversational DistilRuBERT, Russian, cased, 6-layer, 768-hidden, 12-heads, 135.4M parameters: `[deeppavlov_pytorch] <http://files.deeppavlov.ai/deeppavlov_data/bert/distil_ru_conversational_cased_L-6_H-768_A-12_pt.tar.gz>`__
-  Conversational DistilRuBERT-tiny, Russian, cased, 2-layer, 768-hidden, 12-heads, 107M parameters: `[deeppavlov_pytorch] <http://files.deeppavlov.ai/deeppavlov_data/bert/distil_ru_conversational_cased_L-2_H-768_A-12_pt.tar.gz>`__
-  Sentence Multilingual BERT, 101 languages, cased, 12-layer, 768-hidden, 12-heads, 180M parameters: `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/sentence_multi_cased_L-12_H-768_A-12.tar.gz>`__,
   `[deeppavlov_pytorch] <http://files.deeppavlov.ai/deeppavlov_data/bert/sentence_multi_cased_L-12_H-768_A-12_pt_v1.tar.gz>`__
-  Sentence RuBERT, Russian, cased, 12-layer, 768-hidden, 12-heads, 180M parameters: `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/sentence_ru_cased_L-12_H-768_A-12.tar.gz>`__,
   `[deeppavlov_pytorch] <http://files.deeppavlov.ai/deeppavlov_data/bert/sentence_ru_cased_L-12_H-768_A-12_pt_v1.tar.gz>`__

The ``deeppavlov_pytorch`` models are designed to be run with the `HuggingFace's Transformers <https://huggingface.co/transformers/>`__ library.

RuBERT was trained on the Russian part of Wikipedia and news data. We used this training data to build vocabulary of Russian subtokens and took
multilingual version of BERT-base as initialization for RuBERT [1]_.

SlavicBERT was trained on Russian News and four Wikipedias: Bulgarian, Czech, Polish, and Russian.
Subtoken vocabulary was built using this data. Multilingual BERT was used as an initialization for SlavicBERT.
The model is described in our ACL paper [2]_.

Conversational BERT was trained on the English part of Twitter, Reddit, DailyDialogues [4]_, OpenSubtitles [5]_, Debates [6]_, Blogs [7]_, Facebook News Comments.
We used this training data to build the vocabulary of English subtokens and took
English cased version of BERT-base as initialization for English Conversational BERT.

Conversational RuBERT was trained on OpenSubtitles [5]_, Dirty, Pikabu, and Social Media segment of Taiga corpus [8]_.
We assembled new vocabulary for Conversational RuBERT model on this data and initialized model with RuBERT.

Conversational DistilRuBERT (6 transformer layers) and DistilRuBERT-tiny (2 transformer layers) were trained on the same data as Conversational RuBERT and highly inspired by DistilBERT [3]_. Namely, Distil* models (students) used pretrained Conversational RuBERT as teacher and linear combination of the following losses:

1. Masked language modeling loss (between student output logits for tokens and its true labels)
2. Kullback-Leibler divergence (between student and teacher output logits)
3. Cosine embedding loss (between averaged hidden states of the teacher and hidden states of the student)
4. Mean squared error loss (between averaged attention maps of the teacher and attention maps of the student)

Sentence Multilingual BERT is a representation-based sentence encoder for 101 languages of Multilingual BERT.
It is initialized with Multilingual BERT and then fine-tuned on english MultiNLI [9]_ and on dev set of multilingual XNLI [10]_.
Sentence representations are mean pooled token embeddings in the same manner as in Sentence-BERT [12]_.

Sentence RuBERT is a representation-based sentence encoder for Russian.
It is initialized with RuBERT and fine-tuned on SNLI [11]_ google-translated to russian and on russian part of XNLI dev set [10]_.
Sentence representations are mean pooled token embeddings in the same manner as in Sentence-BERT [12]_.

Here, in DeepPavlov, we made it easy to use pre-trained BERT for downstream tasks like classification, tagging, question answering and
ranking. We also provide pre-trained models and examples on how to use BERT with DeepPavlov.

BERT as Embedder
----------------

:class:`~deeppavlov.models.embedders.transformers_embedder.TransformersBertEmbedder` allows for using BERT
model outputs as token, subtoken and sentence level embeddings.

Additionaly the embeddings can be easily used in DeepPavlov. To get text level, token level and subtoken level representations,
you can use or modify a :config:`BERT embedder configuration <embedder/bert_embedder.json>`:

.. code:: python
    
    from deeppavlov.core.common.file import read_json
    from deeppavlov import build_model, configs
    
    bert_config = read_json(configs.embedder.bert_embedder)
    bert_config['metadata']['variables']['BERT_PATH'] = 'path/to/bert/directory'

    m = build_model(bert_config)

    texts = ['Hi, i want my embedding.', 'And mine too, please!']
    tokens, token_embs, subtokens, subtoken_embs, sent_max_embs, sent_mean_embs, bert_pooler_outputs = m(texts)


BERT for Classification
-----------------------

:class:`~deeppavlov.models.torch_bert.torch_transformers_classifier.TorchTransformersClassifierModel`
provides solution for classification problem using pre-trained BERT on PyTorch.
One can use several pre-trained English, multi-lingual and Russian BERT models that are
listed above. :class:`~deeppavlov.models.torch_bert.torch_transformers_classifier.TorchTransformersClassifierModel`
also supports any Transformer-based model of `Transformers <https://github.com/huggingface/transformers>`.

Two main components of BERT classifier pipeline in DeepPavlov are
:class:`~deeppavlov.models.preprocessors.torch_transformers_preprocessor.TorchTransformersPreprocessor` and
:class:`~deeppavlov.models.torch_bert.torch_transformers_classifier.TorchTransformersClassifierModel`.
Non-processed texts should be given to ``torch_transformers_preprocessor`` for tokenization on subtokens,
encoding subtokens with their indices and creating tokens and segment masks.

``torch_transformers_classifier`` has a dense layer of number of classes size upon pooled outputs of Transformer encoder,
it is followed by ``softmax`` activation (``sigmoid`` if ``multilabel`` parameter is set to ``true`` in config).


BERT for Named Entity Recognition (Sequence Tagging)
----------------------------------------------------

Pre-trained BERT model can be used for sequence tagging. Examples of BERT application to sequence tagging
can be found :doc:`here </features/models/NER>`. The module used for tagging
is :class:`~deeppavlov.models.torch_bert.torch_transformers_sequence_tagger:TorchTransformersSequenceTagger`.
The tags are obtained by applying a dense layer to the representation of
the first subtoken of each word. There is also an optional CRF layer on the top.
You can choose among different Transformers architectures by modifying the TRANSFORMER variable in the corresponding configuration files.
The possible choices are DistilBert, Albert, Camembert, XLMRoberta, Bart, Roberta, Bert, XLNet, Flaubert, XLM.

..
    TODO: fix Zero-Shot NER reference

Multilingual BERT model allows to perform zero-shot transfer across languages. To use our 19 tags NER for over a
hundred languages see ner_multi_bert.


BERT for Context Question Answering (SQuAD)
-------------------------------------------
Context Question Answering on `SQuAD <https://rajpurkar.github.io/SQuAD-explorer/>`__ dataset is a task
of looking for an answer on a question in a given context. This task could be formalized as predicting answer start
and end position in a given context. :class:`~deeppavlov.models.torch_bert.torch_transformers_squad:TorchTransformersSquad` on PyTorch uses two linear
transformations to predict probability that current subtoken is start/end position of an answer. For details check
:doc:`Context Question Answering documentation page </features/models/SQuAD>`.

Using custom BERT in DeepPavlov
-------------------------------

The previous sections describe the BERT based models implemented in DeepPavlov.
To change the BERT model used for initialization in any downstream task mentioned above the following parameters of
the :doc:`config </intro/configuration>` file must be changed to match new BERT path:

* download URL in the ``metadata.download.url`` part of the config
* ``bert_config_file``, ``pretrained_bert`` in the BERT based Component. In case of PyTorch BERT, ``pretrained_bert`` can be assigned to
    string name of any Transformer-based model (e.g. ``"bert-base-uncased"``, ``"distilbert-base-uncased"``) and then ``bert_config_file`` is set to ``None``.
* ``vocab_file`` in the ``torch_transformers_preprocessor``. ``vocab_file`` can be assigned to
    string name of used pre-trained BERT (e.g. ``"bert-base-uncased"``).

.. [1] Kuratov, Y., Arkhipov, M. (2019). Adaptation of Deep Bidirectional Multilingual Transformers for Russian Language. arXiv preprint arXiv:1905.07213.
.. [2] Arkhipov M., Trofimova M., Kuratov Y., Sorokin A. (2019). `Tuning Multilingual Transformers for Language-Specific Named Entity Recognition <https://www.aclweb.org/anthology/W19-3712/>`__ . ACL anthology W19-3712.
.. [3] Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.
.. [4] Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset. IJCNLP 2017.
.. [5] P. Lison and J. Tiedemann, 2016, OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles. In Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016)
.. [6] Justine Zhang, Ravi Kumar, Sujith Ravi, Cristian Danescu-Niculescu-Mizil. Proceedings of NAACL, 2016.
.. [7] J. Schler, M. Koppel, S. Argamon and J. Pennebaker (2006). Effects of Age and Gender on Blogging in Proceedings of 2006 AAAI Spring Symposium on Computational Approaches for Analyzing Weblogs.
.. [8] Shavrina T., Shapovalova O. (2017) TO THE METHODOLOGY OF CORPUS CONSTRUCTION FOR MACHINE LEARNING: «TAIGA» SYNTAX TREE CORPUS AND PARSER. in proc. of “CORPORA2017”, international conference , Saint-Petersbourg, 2017.
.. [9] Williams A., Nangia N. & Bowman S. (2017) A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference. arXiv preprint arXiv:1704.05426
.. [10] Williams A., Bowman S. (2018) XNLI: Evaluating Cross-lingual Sentence Representations. arXiv preprint arXiv:1809.05053
.. [11] S. R. Bowman, G. Angeli, C. Potts, and C. D. Manning. (2015) A large annotated corpus for learning natural language inference. arXiv preprint arXiv:1508.05326
.. [12] N. Reimers, I. Gurevych (2019) Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. arXiv preprint arXiv:1908.10084
