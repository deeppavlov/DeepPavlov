Features
========

.. contents:: :local:

Models
------

NER model :doc:`[docs] </features/models/ner>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two models for Named Entity Recognition task in DeepPavlov:
BERT-based and Bi-LSTM+CRF. The models predict tags (in BIO format) for tokens
in input.

BERT-based model is described in  `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
<https://arxiv.org/abs/1810.04805>`__.

The second model reproduces architecture from the paper `Application
of a Hybrid Bi-LSTM-CRF model to the task of Russian Named Entity Recognition <https://arxiv.org/pdf/1709.09686.pdf>`__
which is inspired by Bi-LSTM+CRF architecture from https://arxiv.org/pdf/1603.01360.pdf.

+---------------------------------------------------------+-------+--------------------------------------------------------------------------------------------+-------------+
| Dataset                                                 | Lang  | Model                                                                                      |   Test F1   |
+=========================================================+=======+============================================================================================+=============+
| Persons-1000 dataset with additional LOC and ORG markup | Ru    | :config:`ner_rus_bert.json <ner/ner_rus_bert.json>`                                        |    98.1     |
+                                                         +       +--------------------------------------------------------------------------------------------+-------------+
| (Collection 3)                                          |       | :config:`ner_rus.json <ner/ner_rus.json>`                                                  |    95.1     |
+                                                         +       +--------------------------------------------------------------------------------------------+-------------+
|                                                         |       | :config:`ner_rus_convers_distilrubert_2L.json  <ner/ner_rus_convers_distilrubert_2L.json>` |  88.4 ± 0.5 |
+                                                         +       +--------------------------------------------------------------------------------------------+-------------+
|                                                         |       | :config:`ner_rus_convers_distilrubert_6L.json  <ner/ner_rus_convers_distilrubert_6L.json>` |  93.3 ± 0.3 |
+---------------------------------------------------------+-------+--------------------------------------------------------------------------------------------+-------------+
| Ontonotes                                               | Multi | :config:`ner_ontonotes_bert_mult.json <ner/ner_ontonotes_bert_mult.json>`                  |    88.8     |
+                                                         +-------+--------------------------------------------------------------------------------------------+-------------+
|                                                         | En    | :config:`ner_ontonotes_bert.json <ner/ner_ontonotes_bert.json>`                            |    88.6     |
+                                                         +       +--------------------------------------------------------------------------------------------+-------------+
|                                                         |       | :config:`ner_ontonotes.json <ner/ner_ontonotes.json>`                                      |    87.1     |
+---------------------------------------------------------+       +--------------------------------------------------------------------------------------------+-------------+
| ConLL-2003                                              |       | :config:`ner_conll2003_bert.json <ner/ner_conll2003_bert.json>`                            |    91.7     |
+                                                         +       +--------------------------------------------------------------------------------------------+-------------+
|                                                         |       | :config:`ner_conll2003_torch_bert.json <ner/ner_conll2003_torch_bert.json>`                |    88.6     |
+                                                         +       +--------------------------------------------------------------------------------------------+-------------+
|                                                         |       | :config:`ner_conll2003.json <ner/ner_conll2003.json>`                                      |    89.9     |
+---------------------------------------------------------+       +--------------------------------------------------------------------------------------------+-------------+
| DSTC2                                                   |       | :config:`ner_dstc2.json <ner/ner_dstc2.json>`                                              |    97.1     |
+---------------------------------------------------------+-------+--------------------------------------------------------------------------------------------+-------------+

Slot filling models :doc:`[docs] </features/models/slot_filling>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on fuzzy Levenshtein search to extract normalized slot values from text. The models either rely on NER results
or perform needle in haystack search.

+---------------------------------------------------------------------------------------------------------------------------+------------------+
| Dataset                                                                                                                   |  Slots Accuracy  |
+===========================================================================================================================+==================+
| :config:`DSTC 2 <ner/slotfill_dstc2.json>`                                                                                |       98.85      |
+---------------------------------------------------------------------------------------------------------------------------+------------------+


Classification model :doc:`[docs] </features/models/classifiers>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model for classification tasks (intents, sentiment, etc) on word-level. Shallow-and-wide CNN, Deep CNN, BiLSTM,
BiLSTM with self-attention and other models are presented. The model also allows multilabel classification of texts.
Several pre-trained models are available and presented in Table below.


+------------------+---------------------+------+----------------------------------------------------------------------------------------------------+-------------+------------------+-----------------+-----------+
| Task             | Dataset             | Lang | Model                                                                                              | Metric      | Valid            | Test            | Downloads |
+==================+=====================+======+====================================================================================================+=============+==================+=================+===========+
| 28 intents       | `DSTC 2`_           | En   | :config:`DSTC 2 emb <classifiers/intents_dstc2.json>`                                              | Accuracy    | 0.7613           | 0.7733          |  800 Mb   |
+                  +                     +      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  |                     |      | :config:`Wiki emb <classifiers/intents_dstc2_big.json>`                                            |             | 0.9629           | 0.9617          |  8.5 Gb   |
+                  +                     +      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  |                     |      | :config:`BERT <classifiers/intents_dstc2_bert.json>`                                               |             | 0.9673           | 0.9636          |  800 Mb   |
+------------------+---------------------+      +----------------------------------------------------------------------------------------------------+-------------+------------------+-----------------+-----------+
| 7 intents        | `SNIPS-2017`_ [1]_  |      | :config:`DSTC 2 emb <classifiers/intents_snips.json>`                                              | F1-macro    | 0.8591           |    --           |  800 Mb   |
+                  +                     +      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  |                     |      | :config:`Wiki emb <classifiers/intents_snips_big.json>`                                            |             | 0.9820           |    --           |  8.5 Gb   |
+                  +                     +      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  |                     |      | :config:`Tfidf + SelectKBest + PCA + Wiki emb <classifiers/intents_snips_sklearn.json>`            |             | 0.9673           |    --           |  8.6 Gb   |
+                  +                     +      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  |                     |      | :config:`Wiki emb weighted by Tfidf <classifiers/intents_snips_tfidf_weighted.json>`               |             | 0.9786           |    --           |  8.5 Gb   |
+------------------+---------------------+      +----------------------------------------------------------------------------------------------------+-------------+------------------+-----------------+-----------+
| Insult detection | `Insults`_          |      | :config:`Reddit emb <classifiers/insults_kaggle.json>`                                             | ROC-AUC     | 0.9263           | 0.8556          |  6.2 Gb   |
+                  +                     +      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  |                     |      | :config:`English BERT <classifiers/insults_kaggle_bert.json>`                                      |             | 0.9255           | 0.8612          |  1200 Mb  |
+                  +                     +      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  |                     |      | :config:`English Conversational BERT <classifiers/insults_kaggle_conv_bert.json>`                  |             | 0.9389           | 0.8941          |  1200 Mb  |
+                  +                     +      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  |                     |      | :config:`English BERT on PyTorch <classifiers/insults_kaggle_bert_torch.json>`                     |             | 0.9329           | 0.877           |  1.1 Gb   |
+------------------+---------------------+      +----------------------------------------------------------------------------------------------------+-------------+------------------+-----------------+-----------+
| 5 topics         | `AG News`_          |      | :config:`Wiki emb <classifiers/topic_ag_news.json>`                                                | Accuracy    | 0.8922           | 0.9059          |  8.5 Gb   |
+------------------+---------------------+      +----------------------------------------------------------------------------------------------------+-------------+------------------+-----------------+-----------+
| Intent           | `Yahoo-L31`_        |      | :config:`Yahoo-L31 on conversational BERT <classifiers/yahoo_convers_vs_info_bert.json>`           | ROC-AUC     | 0.9436           |   --            |  1200 Mb  |
+------------------+---------------------+      +----------------------------------------------------------------------------------------------------+-------------+------------------+-----------------+-----------+
| Sentiment        | `SST`_              |      | :config:`5-classes SST on conversational BERT <classifiers/sentiment_sst_conv_bert.json>`          | Accuracy    | 0.6456           | 0.6715          |  400 Mb   |
+                  +                     +      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  |                     |      | :config:`5-classes SST on multilingual BERT <classifiers/sentiment_sst_multi_bert.json>`           |             | 0.5738           | 0.6024          |  660 Mb   |
+                  +                     +      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  |                     |      | :config:`3-classes SST SWCNN on PyTorch <classifiers/sst_torch_swcnn.json>`                        |             | 0.7379           | 0.6312          |  4.3 Mb   |
+                  +---------------------+      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  | `Yelp`_             |      | :config:`5-classes Yelp on conversational BERT <classifiers/sentiment_yelp_conv_bert.json>`        |             | 0.6925           | 0.6842          |  400 Mb   |
+                  +                     +      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  |                     |      | :config:`5-classes Yelp on multilingual BERT <classifiers/sentiment_yelp_multi_bert.json>`         |             | 0.5896           | 0.5874          |  660 Mb   |
+------------------+---------------------+------+----------------------------------------------------------------------------------------------------+-------------+------------------+-----------------+-----------+
| Sentiment        | `Twitter mokoron`_  | Ru   | :config:`RuWiki+Lenta emb w/o preprocessing <classifiers/sentiment_twitter.json>`                  |             | 0.9965           | 0.9961          |  6.2 Gb   |
+                  +                     +      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  |                     |      | :config:`RuWiki+Lenta emb with preprocessing <classifiers/sentiment_twitter_preproc.json>`         |             | 0.7823           | 0.7759          |  6.2 Gb   |
+                  +---------------------+      +----------------------------------------------------------------------------------------------------+-------------+------------------+-----------------+-----------+
|                  | `RuSentiment`_      |      | :config:`RuWiki+Lenta emb <classifiers/rusentiment_cnn.json>`                                      | F1-weighted | 0.6541           | 0.7016          |  6.2 Gb   |
+                  +                     +      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  |                     |      | :config:`Twitter emb super-convergence <classifiers/rusentiment_bigru_superconv.json>` [2]_        |             | 0.7301           | 0.7576          |  3.4 Gb   |
+                  +                     +      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  |                     |      | :config:`ELMo <classifiers/rusentiment_elmo_twitter_cnn.json>`                                     |             | 0.7519           | 0.7875          |  700 Mb   |
+                  +                     +      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  |                     |      | :config:`Multi-language BERT <classifiers/rusentiment_bert.json>`                                  |             | 0.6809           | 0.7193          |  1900 Mb  |
+                  +                     +      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  |                     |      | :config:`Conversational RuBERT <classifiers/rusentiment_convers_bert.json>`                        |             | 0.7548           | 0.7742          |  657 Mb   |
+                  +                     +      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  |                     |      | :config:`Conversational DistilRuBERT-tiny <classifiers/rusentiment_convers_distilrubert_2L.json>`  |             |  0.703 ± 0.0031  | 0.7348 ± 0.0028 |  690 Mb   |
+                  +                     +      +----------------------------------------------------------------------------------------------------+             +------------------+-----------------+-----------+
|                  |                     |      | :config:`Conversational DistilRuBERT-base <classifiers/rusentiment_convers_distilrubert_6L.json>`  |             |  0.7376 ± 0.0045 | 0.7645 ± 0.035  |  1.0 Gb   |
+------------------+---------------------+      +----------------------------------------------------------------------------------------------------+-------------+------------------+-----------------+-----------+
| Intent           | Ru like`Yahoo-L31`_ |      | :config:`Conversational vs Informational on ELMo <classifiers/yahoo_convers_vs_info.json>`         | ROC-AUC     | 0.9412           |   --            |  700 Mb   |
+------------------+---------------------+------+----------------------------------------------------------------------------------------------------+-------------+------------------+-----------------+-----------+

.. [1] Coucke A. et al. Snips voice platform: an embedded spoken language understanding system for private-by-design voice interfaces //arXiv preprint arXiv:1805.10190. – 2018.
.. [2] Smith L. N., Topin N. Super-convergence: Very fast training of residual networks using large learning rates. – 2018.

.. _`DSTC 2`: http://camdial.org/~mh521/dstc/
.. _`SNIPS-2017`: https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines
.. _`Insults`: https://www.kaggle.com/c/detecting-insults-in-social-commentary
.. _`AG News`: https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html
.. _`Twitter mokoron`: http://study.mokoron.com/
.. _`RuSentiment`: http://text-machine.cs.uml.edu/projects/rusentiment/
.. _`Yahoo-L31`: https://webscope.sandbox.yahoo.com/catalog.php?datatype=l
.. _`Yahoo-L6`: https://webscope.sandbox.yahoo.com/catalog.php?datatype=l
.. _`SST`: https://nlp.stanford.edu/sentiment/index.html
.. _`Yelp`: https://www.yelp.com/dataset

As no one had published intent recognition for DSTC-2 data, the
comparison of the presented model is given on **SNIPS** dataset. The
evaluation of model scores was conducted in the same way as in [3]_ to
compare with the results from the report of the authors of the dataset.
The results were achieved with tuning of parameters and embeddings
trained on Reddit dataset.

+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+
| Model                  | AddToPlaylist   | BookRestaurant   | GetWheather   | PlayMusic    | RateBook     | SearchCreativeWork   | SearchScreeningEvent   |
+========================+=================+==================+===============+==============+==============+======================+========================+
| api.ai                 | 0.9931          | 0.9949           | 0.9935        | 0.9811       | 0.9992       | 0.9659               | 0.9801                 |
+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+
| ibm.watson             | 0.9931          | 0.9950           | 0.9950        | 0.9822       | 0.9996       | 0.9643               | 0.9750                 |
+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+
| microsoft.luis         | 0.9943          | 0.9935           | 0.9925        | 0.9815       | 0.9988       | 0.9620               | 0.9749                 |
+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+
| wit.ai                 | 0.9877          | 0.9913           | 0.9921        | 0.9766       | 0.9977       | 0.9458               | 0.9673                 |
+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+
| snips.ai               | 0.9873          | 0.9921           | 0.9939        | 0.9729       | 0.9985       | 0.9455               | 0.9613                 |
+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+
| recast.ai              | 0.9894          | 0.9943           | 0.9910        | 0.9660       | 0.9981       | 0.9424               | 0.9539                 |
+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+
| amazon.lex             | 0.9930          | 0.9862           | 0.9825        | 0.9709       | 0.9981       | 0.9427               | 0.9581                 |
+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+
+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+
| Shallow-and-wide CNN   | **0.9956**      | **0.9973**       | **0.9968**    | **0.9871**   | **0.9998**   | **0.9752**           | **0.9854**             |
+------------------------+-----------------+------------------+---------------+--------------+--------------+----------------------+------------------------+

.. [3] https://www.slideshare.net/KonstantinSavenkov/nlu-intent-detection-benchmark-by-intento-august-2017


Automatic spelling correction model :doc:`[docs] </features/models/spelling_correction>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pipelines that use candidates search in a static dictionary and an ARPA language model to correct spelling errors.

.. note::

    About 4.4 GB on disc required for the Russian language model and about 7 GB for the English one.

Comparison on the `test set <http://www.dialog-21.ru/media/3838/test_sample_testset.txt>`__ for the `SpellRuEval
competition <http://www.dialog-21.ru/en/evaluation/2016/spelling_correction/>`__
on Automatic Spelling Correction for Russian:

+-----------------------------------------------------------------------------------------+-----------+--------+-----------+---------------------+
| Correction method                                                                       | Precision | Recall | F-measure | Speed (sentences/s) |
+=========================================================================================+===========+========+===========+=====================+
| Yandex.Speller                                                                          | 83.09     | 59.86  | 69.59     | 5.                  |
+-----------------------------------------------------------------------------------------+-----------+--------+-----------+---------------------+
| :config:`Damerau Levenshtein 1 + lm<spelling_correction/levenshtein_corrector_ru.json>` | 53.26     | 53.74  | 53.50     | 29.3                |
+-----------------------------------------------------------------------------------------+-----------+--------+-----------+---------------------+
| :config:`Brill Moore top 4 + lm<spelling_correction/brillmoore_kartaslov_ru.json>`      | 51.92     | 53.94  | 52.91     | 0.6                 |
+-----------------------------------------------------------------------------------------+-----------+--------+-----------+---------------------+
| Hunspell + lm                                                                           | 41.03     | 48.89  | 44.61     | 2.1                 |
+-----------------------------------------------------------------------------------------+-----------+--------+-----------+---------------------+
| JamSpell                                                                                | 44.57     | 35.69  | 39.64     | 136.2               |
+-----------------------------------------------------------------------------------------+-----------+--------+-----------+---------------------+
| :config:`Brill Moore top 1 <spelling_correction/brillmoore_kartaslov_ru_nolm.json>`     | 41.29     | 37.26  | 39.17     | 2.4                 |
+-----------------------------------------------------------------------------------------+-----------+--------+-----------+---------------------+
| Hunspell                                                                                | 30.30     | 34.02  | 32.06     | 20.3                |
+-----------------------------------------------------------------------------------------+-----------+--------+-----------+---------------------+



Ranking model :doc:`[docs] </features/models/neural_ranking>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main neural ranking model based on `LSTM-based deep learning models for non-factoid answer selection
<https://arxiv.org/abs/1511.04108>`__. The model performs ranking of responses or contexts from some database by their
relevance for the given context.

There are 3 alternative neural architectures available as well:

Sequential Matching Network (SMN)
   Based on the work `Wu, Yu, et al. "Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-based Chatbots". ACL. 2017. <https://arxiv.org/abs/1612.01627>`__

Deep Attention Matching Network (DAM)
   Based on the work `Xiangyang Zhou, et al. "Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network". Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018 <http://aclweb.org/anthology/P18-1103>`__

Deep Attention Matching Network + Universal Sentence Encoder v3 (DAM-USE-T)
   Our new proposed architecture based on the works: `Xiangyang Zhou, et al. "Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network". Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018 <http://aclweb.org/anthology/P18-1103>`__
   and `Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant, Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, Brian Strope, Ray Kurzweil. 2018a. Universal Sentence Encoder for English. <https://arxiv.org/abs/1803.11175>`__


Available pre-trained models for ranking:

.. table::
   :widths: auto

   +-------------------+----------------------------------------------------------------------------------------------------------------------+-----------+-----------------------------------+
   |    Dataset        |   Model config                                                                                                       |    Val    |   Test                            |
   |                   |                                                                                                                      +-----------+-------+-------+-------+-----------+
   |                   |                                                                                                                      |   R10@1   | R10@1 | R10@2 | R10@5 | Downloads |
   +===================+======================================================================================================================+===========+=======+=======+=======+===========+
   | `Ubuntu V2`_      | :config:`ranking_ubuntu_v2_mt_word2vec_dam_transformer <ranking/ranking_ubuntu_v2_mt_word2vec_dam_transformer.json>` |   74.32   | 74.46 | 86.77 | 97.38 |  2457 MB  |
   +-------------------+----------------------------------------------------------------------------------------------------------------------+-----------+-------+-------+-------+-----------+
   | `Ubuntu V2`_      | :config:`ranking_ubuntu_v2_mt_word2vec_smn <ranking/ranking_ubuntu_v2_mt_word2vec_smn.json>`                         |   68.56   | 67.91 | 81.49 | 95.63 |  1609 MB  |
   +-------------------+----------------------------------------------------------------------------------------------------------------------+-----------+-------+-------+-------+-----------+
   | `Ubuntu V2`_      | :config:`ranking_ubuntu_v2_bert_uncased <ranking/ranking_ubuntu_v2_bert_uncased.json>`                               |   66.5    | 66.6  | --    | --    |  396 MB   |
   +-------------------+----------------------------------------------------------------------------------------------------------------------+-----------+-------+-------+-------+-----------+
   | `Ubuntu V2`_      | :config:`ranking_ubuntu_v2_bert_uncased on PyTorch <ranking/ranking_ubuntu_v2_torch_bert_uncased.json>`              |   65.73   | 65.74 | --    | --    |  1.1 Gb   |
   +-------------------+----------------------------------------------------------------------------------------------------------------------+-----------+-------+-------+-------+-----------+
   | `Ubuntu V2`_      | :config:`ranking_ubuntu_v2_bert_sep <ranking/ranking_ubuntu_v2_bert_sep.json>`                                       |   66.5    | 66.5  | --    | --    |  396 MB   |
   +-------------------+----------------------------------------------------------------------------------------------------------------------+-----------+-------+-------+-------+-----------+
   | `Ubuntu V2`_      | :config:`ranking_ubuntu_v2_mt_interact <ranking/ranking_ubuntu_v2_mt_interact.json>`                                 |   59.2    | 58.7  | --    | --    |  8906 MB  |
   +-------------------+----------------------------------------------------------------------------------------------------------------------+-----------+-------+-------+-------+-----------+

.. _`Ubuntu V2`: https://github.com/rkadlec/ubuntu-ranking-dataset-creator

Available pre-trained models for paraphrase identification:

.. table::
   :widths: auto

   +------------------------+------------------------------------------------------------------------------------------------------+----------------+-----------------+------------+------------+----------------+-----------------+-----------+
   |    Dataset             | Model config                                                                                         | Val (accuracy) | Test (accuracy) | Val (F1)   | Test (F1)  | Val (log_loss) | Test (log_loss) | Downloads |
   +========================+======================================================================================================+================+=================+============+============+================+=================+===========+
   | `paraphraser.ru`_      | :config:`paraphrase_ident_paraphraser_ft <ranking/paraphrase_ident_paraphraser_interact.json>`       |   83.8         |   75.4          |   87.9     |  80.9      |   0.468        |   0.616         | 5938M     |
   +------------------------+------------------------------------------------------------------------------------------------------+----------------+-----------------+------------+------------+----------------+-----------------+-----------+
   | `paraphraser.ru`_      | :config:`paraphrase_bert_multilingual <classifiers/paraphraser_bert.json>`                           |   87.4         |   79.3          |   90.2     |  83.4      |   --           |   --            | 1330M     |
   +------------------------+------------------------------------------------------------------------------------------------------+----------------+-----------------+------------+------------+----------------+-----------------+-----------+
   | `paraphraser.ru`_      | :config:`paraphrase_rubert <classifiers/paraphraser_rubert.json>`                                    |   90.2         |   84.9          |   92.3     |  87.9      |   --           |   --            | 1325M     |
   +------------------------+------------------------------------------------------------------------------------------------------+----------------+-----------------+------------+------------+----------------+-----------------+-----------+
   | `paraphraser.ru`_      | :config:`paraphraser_convers_distilrubert_2L <classifiers/paraphraser_convers_distilrubert_2L.json>` |  76.1 ± 0.2    |  64.5 ± 0.5     | 81.8 ± 0.2 | 73.9 ± 0.8 |   --           |   --            | 618M      |
   +------------------------+------------------------------------------------------------------------------------------------------+----------------+-----------------+------------+------------+----------------+-----------------+-----------+
   | `paraphraser.ru`_      | :config:`paraphraser_convers_distilrubert_6L <classifiers/paraphraser_convers_distilrubert_6L.json>` |  86.5 ± 0.5    |  78.9 ± 0.4     | 89.6 ± 0.3 | 83.2 ± 0.5 |   --           |   --            | 930M      |
   +------------------------+------------------------------------------------------------------------------------------------------+----------------+-----------------+------------+------------+----------------+-----------------+-----------+

.. _`paraphraser.ru`: https://paraphraser.ru/


Comparison with other models on the `Ubuntu Dialogue Corpus v2 <http://www.cs.toronto.edu/~lcharlin/papers/ubuntu_dialogue_dd17.pdf>`__ (test):

+---------------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------+-----------+
| Model                                                                                                                                       | R@1       | R@2       | R@5       |
+=============================================================================================================================================+===========+===========+===========+
| SMN last [`Wu et al., 2017 <https://www.aclweb.org/anthology/P17-1046>`_]                                                                   | --        | --        | --        |
+---------------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------+-----------+
| SMN last [DeepPavlov :config:`ranking_ubuntu_v2_mt_word2vec_smn <ranking/ranking_ubuntu_v2_mt_word2vec_smn.json>`]                          | 0.6791    | 0.8149    | 0.9563    |
+---------------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------+-----------+
| DAM [`Zhou et al., 2018 <http://aclweb.org/anthology/P18-1103>`_]                                                                           | --        | --        | --        |
+---------------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------+-----------+
| MRFN-FLS [`Tao et al., 2019 <https://dl.acm.org/citation.cfm?id=3290985>`_]                                                                 | --        | --        | --        |
+---------------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------+-----------+
| IMN [`Gu et al., 2019 <https://arxiv.org/abs/1901.01824>`_]                                                                                 | 0.771     | 0.886     | 0.979     |
+---------------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------+-----------+
| IMN Ensemble [`Gu et al., 2019 <https://arxiv.org/abs/1901.01824>`_]                                                                        | **0.791** | **0.899** | **0.982** |
+---------------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------+-----------+
| DAM-USE-T [DeepPavlov :config:`ranking_ubuntu_v2_mt_word2vec_dam_transformer <ranking/ranking_ubuntu_v2_mt_word2vec_dam_transformer.json>`] | 0.7446    | 0.8677    | 0.9738    |
+---------------------------------------------------------------------------------------------------------------------------------------------+-----------+-----------+-----------+


References:

* Yu Wu, Wei Wu, Ming Zhou, and Zhoujun Li. 2017. Sequential match network: A new architecture for multi-turn response selection in retrieval-based chatbots. In ACL, pages 372–381. https://www.aclweb.org/anthology/P17-1046

* Xiangyang Zhou, Lu Li, Daxiang Dong, Yi Liu, Ying Chen, Wayne Xin Zhao, Dianhai Yu and Hua Wu. 2018. Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1118-1127, ACL. http://aclweb.org/anthology/P18-1103

* Chongyang Tao, Wei Wu, Can Xu, Wenpeng Hu, Dongyan Zhao, and Rui Yan. Multi-Representation Fusion Network for Multi-turn Response Selection in Retrieval-based Chatbots. In WSDM'19. https://dl.acm.org/citation.cfm?id=3290985

* Gu, Jia-Chen & Ling, Zhen-Hua & Liu, Quan. (2019). Interactive Matching Network for Multi-Turn Response Selection in Retrieval-Based Chatbots. https://arxiv.org/abs/1901.01824



TF-IDF Ranker model :doc:`[docs] </features/models/tfidf_ranking>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on `Reading Wikipedia to Answer Open-Domain Questions <https://github.com/facebookresearch/DrQA/>`__. The model solves the task of document retrieval for a given query.

+---------------+-------------------------------------------------------------------+----------------------+-----------------+-----------+
| Dataset       | Model                                                             |  Wiki dump           |  Recall@5       | Downloads |
+===============+========================================================+==========+======================+=================+===========+
| `SQuAD-v1.1`_ | :config:`doc_retrieval <doc_retrieval/en_ranker_tfidf_wiki.json>` |  enwiki (2018-02-11) |   75.6          | 33 GB     |
+---------------+-------------------------------------------------+-----------------+----------------------+-----------------+-----------+


Question Answering model :doc:`[docs] </features/models/squad>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models in this section solve the task of looking for an answer on a
question in a given context (`SQuAD <https://rajpurkar.github.io/SQuAD-explorer/>`__ task format).
There are two models for this task in DeepPavlov: BERT-based and R-Net. Both models predict answer start and end
position in a given context.

BERT-based model is described in  `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
<https://arxiv.org/abs/1810.04805>`__.

R-Net model is based on `R-NET: Machine Reading Comprehension with Self-matching Networks
<https://www.microsoft.com/en-us/research/publication/mcr/>`__.

+----------------+---------------------------------------------------------------------------------------------+-------+----------------+-----------------+-----------------+
|    Dataset     | Model config                                                                                | lang  |    EM (dev)    |    F-1 (dev)    |    Downloads    |
+================+=============================================================================================+=======+================+=================+=================+
| `SQuAD-v1.1`_  | :config:`DeepPavlov BERT <squad/squad_bert.json>`                                           |  en   |     80.88      |     88.49       |     806Mb       |
+----------------+---------------------------------------------------------------------------------------------+-------+----------------+-----------------+-----------------+
| `SQuAD-v1.1`_  | :config:`DeepPavlov BERT on PyTorch <squad/squad_torch_bert.json>`                          |  en   |    80.79       |     88.30       |     1.1 Gb      |
+----------------+---------------------------------------------------------------------------------------------+-------+----------------+-----------------+-----------------+
| `SQuAD-v1.1`_  | :config:`DeepPavlov R-Net <squad/squad.json>`                                               |  en   |     71.49      |     80.34       |     ~2.5Gb      |
+----------------+---------------------------------------------------------------------------------------------+-------+----------------+-----------------+-----------------+
| `SDSJ Task B`_ | :config:`DeepPavlov RuBERT <squad/squad_ru_bert_infer.json>`                                |  ru   |  66.30 ± 0.24  |   84.60 ± 0.11  |     1325Mb      |
+----------------+---------------------------------------------------------------------------------------------+-------+----------------+-----------------+-----------------+
| `SDSJ Task B`_ | :config:`DeepPavlov multilingual BERT <squad/squad_ru_bert_infer.json>`                     |  ru   |  64.35 ± 0.39  |   83.39 ± 0.08  |     1323Mb      |
+----------------+---------------------------------------------------------------------------------------------+-------+----------------+-----------------+-----------------+
| `SDSJ Task B`_ | :config:`DeepPavlov R-Net <squad/squad_ru.json>`                                            |  ru   |     60.62      |     80.04       |     ~5Gb        |
+----------------+---------------------------------------------------------------------------------------------+-------+----------------+-----------------+-----------------+
| `SDSJ Task B`_ | :config:`DeepPavlov DistilRuBERT-tiny <squad/squad_ru_convers_distilrubert_2L_infer.json>`  |  ru   |  44.2 ± 0.46   |  65.1 ± 0.36    |     867Mb       |
+----------------+---------------------------------------------------------------------------------------------+-------+----------------+-----------------+-----------------+
| `SDSJ Task B`_ | :config:`DeepPavlov DistilRuBERT-base <squad/squad_ru_convers_distilrubert_6L_infer.json>`  |  ru   |  61.23 ± 0.42  |  80.36 ± 0.28   |     1.18Gb      |
+----------------+---------------------------------------------------------------------------------------------+-------+----------------+-----------------+-----------------+
|    `DRCD`_     | :config:`DeepPavlov multilingual BERT <squad/squad_zh_bert_mult.json>`                      |  ch   |     84.86      |     89.03       |     630Mb       |
+----------------+---------------------------------------------------------------------------------------------+-------+----------------+-----------------+-----------------+
|    `DRCD`_     | :config:`DeepPavlov Chinese BERT <squad/squad_zh_bert_zh.json>`                             |  ch   |     84.19      |     89.23       |     362Mb       |
+----------------+---------------------------------------------------------------------------------------------+-------+----------------+-----------------+-----------------+

In the case when answer is not necessary present in given context we have :config:`squad_noans <squad/multi_squad_noans.json>`
model. This model outputs empty string in case if there is no answer in context.


Morphological tagging model :doc:`[docs] </features/models/morphotagger>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have a BERT-based model for Russian and character-based models for 11 languages.
The character model is based on `Heigold et al., 2017. An extensive empirical evaluation of
character-based morphological tagging for 14 languages <http://www.aclweb.org/anthology/E17-1048>`__.
It is a state-of-the-art model for Russian and near state of the art for several other languages.
Model takes as input tokenized sentences and outputs the corresponding
sequence of morphological labels in `UD format <http://universaldependencies.org/format.html>`__.
The table below contains word and sentence accuracy on UD2.0 datasets.
For more scores see :doc:`full table </features/models/morphotagger>`.

.. table::
    :widths: auto

    +----------------------+--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    |    Dataset           | Model                                                                                                        | Word accuracy | Sent. accuracy | Download size (MB) |
    +======================+==============================================================================================================+===============+================+====================+
    | `UD2.3`_ (Russian)   | `UD Pipe 2.3`_ (Straka et al., 2017)                                                                         |    93.5       |                |                    |
    |                      +--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    |                      | `UD Pipe Future`_ (Straka et al., 2018)                                                                      |    96.90      |                |                    |
    |                      +--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    |                      | :config:`BERT-based model <morpho_tagger/BERT/morpho_ru_syntagrus_bert.json>`                                |    97.83      |     72.02      |       661          |
    +----------------------+--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    |                      | `Pymorphy`_ + `russian_tagsets`_ (first tag)                                                                 |     60.93     |      0.00      |                    |
    +                      +--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    | `UD2.0`_ (Russian)   | `UD Pipe 1.2`_ (Straka et al., 2017)                                                                         |     93.57     |     43.04      |                    |
    +                      +--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    |                      | :config:`Basic model <morpho_tagger/UD2.0/ru_syntagrus/morpho_ru_syntagrus.json>`                            |     95.17     |     50.58      |        48.7        |
    +                      +--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    |                      | :config:`Pymorphy-enhanced model <morpho_tagger/UD2.0/ru_syntagrus/morpho_ru_syntagrus_pymorphy.json>`       |   **96.23**   |     58.00      |        48.7        |
    +----------------------+--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    | `UD2.0`_ (Czech)     | `UD Pipe 1.2`_ (Straka et al., 2017)                                                                         |     91.86     |     42.28      |                    |
    |                      +--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    |                      | :config:`Basic model <morpho_tagger/UD2.0/morpho_cs.json>`                                                   |   **94.35**   |     51.56      |        41.8        |
    +----------------------+--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    | `UD2.0`_ (English)   | `UD Pipe 1.2`_ (Straka et al., 2017)                                                                         |     92.89     |     55.75      |                    |
    |                      +--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    |                      | :config:`Basic model <morpho_tagger/UD2.0/morpho_en.json>`                                                   |   **93.00**   |     55.18      |        16.9        |
    +----------------------+--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    | `UD2.0`_ (German)    | `UD Pipe 1.2`_ (Straka et al., 2017)                                                                         |     76.65     |     10.24      |                    |
    |                      +--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    |                      | :config:`Basic model <morpho_tagger/UD2.0/morpho_de.json>`                                                   |   **83.83**   |     15.25      |        18.6        |
    +----------------------+--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+

.. _`Pymorphy`: https://pymorphy2.readthedocs.io/en/latest/
.. _`russian_tagsets`: https://github.com/kmike/russian-tagsets
.. _`UD2.0`: https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1983
.. _`UD2.3`: http://hdl.handle.net/11234/1-2895
.. _`UD Pipe 1.2`: http://ufal.mff.cuni.cz/udpipe
.. _`UD Pipe 2.3`: http://ufal.mff.cuni.cz/udpipe
.. _`UD Pipe Future`: https://github.com/CoNLL-UD-2018/UDPipe-Future

Syntactic parsing model :doc:`[docs] </features/models/syntaxparser>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have a biaffine model for syntactic parsing based on RuBERT.
It achieves the highest known labeled attachments score of 93.7%
on ``ru_syntagrus`` Russian corpus (version UD 2.3).

.. table::
    :widths: auto

    +-------------------------+-------------------------------------------------------------------------------------------+---------+----------+
    |   Dataset               |  Model                                                                                    | UAS     | LAS      |
    +=========================+===========================================================================================+=========+==========+
    | `UD2.3`_ (Russian)      | `UD Pipe 2.3`_ (Straka et al., 2017)                                                      | 90.3    | 89.0     |
    |                         +-------------------------------------------------------------------------------------------+---------+----------+
    |                         | `UD Pipe Future`_ (Straka, 2018)                                                          | 93.0    | 91.5     |
    |                         +-------------------------------------------------------------------------------------------+---------+----------+
    |                         | `UDify (multilingual BERT)`_ (Kondratyuk, 2018)                                           | 94.8    | 93.1     |
    |                         +-------------------------------------------------------------------------------------------+---------+----------+
    |                         | :config:`our BERT model <syntax/syntax_ru_syntagrus_bert.json>`                           | 95.2    | 93.7     |
    +-------------------------+-------------------------------------------------------------------------------------------+---------+----------+

.. _`UD2.3`: http://hdl.handle.net/11234/1-2895
.. _`UD Pipe 2.3`: http://ufal.mff.cuni.cz/udpipe
.. _`UD Pipe Future`: https://github.com/CoNLL-UD-2018/UDPipe-Future
.. _`UDify (multilingual BERT)`: https://github.com/hyperparticle/udify

Frequently Asked Questions (FAQ) model :doc:`[docs] </features/skills/faq>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set of pipelines for FAQ task: classifying incoming question into set of known questions and return prepared answer.
You can build different pipelines based on: tf-idf, weighted fasttext, cosine similarity, logistic regression.


Skills
------

Goal-oriented bot :doc:`[docs] </features/skills/go_bot>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on Hybrid Code Networks (HCNs) architecture from `Jason D. Williams, Kavosh Asadi, 
Geoffrey Zweig, Hybrid Code Networks: practical and efficient end-to-end dialog control 
with supervised and reinforcement learning – 2017 <https://arxiv.org/abs/1702.03274>`__.
It allows to predict responses in a goal-oriented dialog. The model is
customizable: embeddings, slot filler and intent classifier can be switched on and off on demand.

Available pre-trained models and their comparison with existing benchmarks:

+-----------------------------------+------+------------------------------------------------------------------------------------+---------------+-----------+---------------+
| Dataset                           | Lang | Model                                                                              | Metric        | Test      | Downloads     |
+===================================+======+====================================================================================+===============+===========+===============+
| `DSTC 2`_                         | En   | :config:`basic bot <go_bot/gobot_dstc2_minimal.json>`                              | Turn Accuracy | 0.380     | 10 Mb         |
+ (:ref:`modified <dstc2_dataset>`) +      +------------------------------------------------------------------------------------+               +-----------+---------------+
|                                   |      | :config:`bot with slot filler <go_bot/gobot_dstc2.json>`                           |               | 0.542     | 400 Mb        |
+                                   +      +------------------------------------------------------------------------------------+               +-----------+---------------+
|                                   |      | :config:`bot with slot filler, intents & attention <go_bot/gobot_dstc2_best.json>` |               | **0.553** | 8.5 Gb        |
+-----------------------------------+      +------------------------------------------------------------------------------------+               +-----------+---------------+
| `DSTC 2`_                         |      | Bordes and Weston (2016)                                                           |               | 0.411     | --            |
+                                   +      +------------------------------------------------------------------------------------+               +-----------+---------------+
|                                   |      | Eric and Manning (2017)                                                            |               | 0.480     | --            |
+                                   +      +------------------------------------------------------------------------------------+               +-----------+---------------+
|                                   |      | Perez and Liu (2016)                                                               |               | 0.487     | --            |
+                                   +      +------------------------------------------------------------------------------------+               +-----------+---------------+
|                                   |      | Williams et al. (2017)                                                             |               | **0.556** | --            |
+-----------------------------------+------+------------------------------------------------------------------------------------+---------------+-----------+---------------+


ODQA :doc:`[docs] </features/skills/odqa>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An open domain question answering skill. The skill accepts free-form questions about the world and outputs an answer
based on its Wikipedia knowledge.


+----------------+--------------------------------------------------------------------+-----------------------+--------+-----------+
| Dataset        | Model config                                                       |  Wiki dump            |   F1   | Downloads |
+================+====================================================================+=======================+========+===========+
| `SQuAD-v1.1`_  | :config:`ODQA <odqa/en_odqa_infer_wiki.json>`                      | enwiki (2018-02-11)   |  35.89 | 9.7Gb     |
+----------------+--------------------------------------------------------------------+-----------------------+--------+-----------+
| `SQuAD-v1.1`_  | :config:`ODQA <odqa/en_odqa_infer_enwiki20161221.json>`            | enwiki (2016-12-21)   |  37.83 | 9.3Gb     |
+----------------+--------------------------------------------------------------------+-----------------------+--------+-----------+
| `SDSJ Task B`_ | :config:`ODQA <odqa/ru_odqa_infer_wiki.json>`                      | ruwiki (2018-04-01)   |  28.56 | 7.7Gb     |
+----------------+--------------------------------------------------------------------+-----------------------+--------+-----------+
| `SDSJ Task B`_ | :config:`ODQA with RuBERT <odqa/ru_odqa_infer_wiki_rubert.json>`   | ruwiki (2018-04-01)   |  37.83 | 4.3Gb     |
+----------------+--------------------------------------------------------------------+-----------------------+--------+-----------+


AutoML
--------------------

Hyperparameters optimization :doc:`[docs] </features/hypersearch>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hyperparameters optimization by cross-validation for DeepPavlov models
that requires only some small changes in a config file.


Embeddings
----------

Pre-trained embeddings :doc:`[docs] </features/pretrained_vectors>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Word vectors for the Russian language trained on joint `Russian Wikipedia <https://ru.wikipedia.org/>`__ and `Lenta.ru
<https://lenta.ru/>`__ corpora.


Examples of some models
---------------------------

-  Run goal-oriented bot with Telegram interface:

   .. code-block:: bash

      python -m deeppavlov telegram gobot_dstc2 -d -t <TELEGRAM_TOKEN>

-  Run goal-oriented bot with console interface:

   .. code-block:: bash

      python -m deeppavlov interact gobot_dstc2 -d

-  Run goal-oriented bot with REST API:

   .. code-block:: bash

      python -m deeppavlov riseapi gobot_dstc2 -d

-  Run slot-filling model with Telegram interface:

   .. code-block:: bash

      python -m deeppavlov telegram slotfill_dstc2 -d -t <TELEGRAM_TOKEN>

-  Run slot-filling model with console interface:

   .. code-block:: bash

      python -m deeppavlov interact slotfill_dstc2 -d

-  Run slot-filling model with REST API:

   .. code-block:: bash

      python -m deeppavlov riseapi slotfill_dstc2 -d

-  Predict intents on every line in a file:

   .. code-block:: bash

      python -m deeppavlov predict intents_snips -d --batch-size 15 < /data/in.txt > /data/out.txt


View `video demo <https://youtu.be/yzoiCa_sMuY>`__ of deployment of a
goal-oriented bot and a slot-filling model with Telegram UI.


.. _`SQuAD-v1.1`: https://arxiv.org/abs/1606.05250
.. _`SDSJ Task B`: https://arxiv.org/abs/1912.09723
.. _`DRCD`: https://arxiv.org/abs/1806.00920
