Features
========

Components
----------

:doc:`NER component </components/ner>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on neural Named Entity Recognition network. The NER component reproduces architecture from the paper `Application
of a Hybrid Bi-LSTM-CRF model to the task of Russian Named Entity Recognition <https://arxiv.org/pdf/1709.09686.pdf>`__
which is inspired by Bi-LSTM+CRF architecture from https://arxiv.org/pdf/1603.01360.pdf.

+---------------------------------------------------------------------------------------------------------------------------+------------------+
| Dataset                                                                                                                   |     Test F1      |
+===========================================================================================================================+==================+
| :config:`Persons-1000 dataset with additional LOC and ORG markup <ner/ner_rus.json>`                                      |       95.25      |
+---------------------------------------------------------------------------------------------------------------------------+------------------+
| :config:`DSTC 2 <ner/ner_dstc2.json>`                                                                                     |       98.40      |
+---------------------------------------------------------------------------------------------------------------------------+------------------+
| :config:`OntoNotes  <ner/ner_ontonotes.json>`                                                                             |       87.07      |
+---------------------------------------------------------------------------------------------------------------------------+------------------+


:doc:`Slot filling components </components/slot_filling>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on fuzzy Levenshtein search to extract normalized slot values from text. The components either rely on NER results
or perform needle in haystack search.

+---------------------------------------------------------------------------------------------------------------------------+------------------+
| Dataset                                                                                                                   |  Slots Accuracy  |
+===========================================================================================================================+==================+
| :config:`DSTC 2 <ner/slotfill_dstc2.json>`                                                                                |       98.85      |
+---------------------------------------------------------------------------------------------------------------------------+------------------+


:doc:`Classification component </components/classifiers>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Component for classification tasks (intents, sentiment, etc) on word-level. Shallow-and-wide CNN, Deep CNN, BiLSTM,
BiLSTM with self-attention and other models are presented. The model also allows multilabel classification of texts.
Several pre-trained models are available and presented in Table below.

+------------------+-------------------+------+-------------------------------------------------------------------------------------------------+----------+--------+--------+-----------+
| Task             | Dataset           | Lang | Model                                                                                           | Metric   | Valid  | Test   | Downloads |
+==================+===================+======+=================================================================================================+==========+========+========+===========+
| 28 intents       | `DSTC 2`_         | En   | :config:`DSTC 2 emb <classifiers/intents_dstc2.json>`                                           | Accuracy | 0.7613 | 0.7733 |  800 Mb   |
+                  +                   +      +-------------------------------------------------------------------------------------------------+          +--------+--------+-----------+
|                  |                   |      | :config:`Wiki emb <classifiers/intents_dstc2_big.json>`                                         |          | 0.9629 | 0.9617 |  8.5 Gb   |
+------------------+-------------------+      +-------------------------------------------------------------------------------------------------+----------+--------+--------+-----------+
| 7 intents        | `SNIPS-2017`_     |      | :config:`DSTC 2 emb <classifiers/intents_snips.json>`                                           | F1       | 0.8591 |    --  |  800 Mb   |
+                  +                   +      +-------------------------------------------------------------------------------------------------+          +--------+--------+-----------+
|                  |                   |      | :config:`Wiki emb <classifiers/intents_snips_big.json>`                                         |          | 0.9820 |    --  |  8.5 Gb   |
+                  +                   +      +-------------------------------------------------------------------------------------------------+          +--------+--------+-----------+
|                  |                   |      | :config:`Tfidf + SelectKBest + PCA + Wiki emb <classifiers/intents_snips_sklearn.json>`         |          | 0.9673 |    --  |  8.6 Gb   |
+                  +                   +      +-------------------------------------------------------------------------------------------------+          +--------+--------+-----------+
|                  |                   |      | :config:`Wiki emb weighted by Tfidf <classifiers/intents_snips_tfidf_weighted.json>`            |          | 0.9786 |    --  |  8.5 Gb   |
+------------------+-------------------+      +-------------------------------------------------------------------------------------------------+----------+--------+--------+-----------+
| Insult detection | `Insults`_        |      | :config:`Reddit emb <classifiers/insults_kaggle.json>`                                          | ROC-AUC  | 0.9263 | 0.8556 |  6.2 Gb   |
+------------------+-------------------+      +-------------------------------------------------------------------------------------------------+----------+--------+--------+-----------+
| 5 topics         | `AG News`_        |      | :config:`Wiki emb <classifiers/topic_ag_news.json>`                                             | Accuracy | 0.8922 | 0.9059 |  8.5 Gb   |
+------------------+-------------------+------+-------------------------------------------------------------------------------------------------+          +--------+--------+-----------+
| Sentiment        |`Twitter mokoron`_ | Ru   | :config:`RuWiki+Lenta emb w/o preprocessing <classifiers/sentiment_twitter.json>`               |          | 0.9965 | 0.9961 |  6.2 Gb   |
+                  +                   +      +-------------------------------------------------------------------------------------------------+          +--------+--------+-----------+
|                  |                   |      | :config:`RuWiki+Lenta emb with preprocessing <classifiers/sentiment_twitter_preproc.json>`      |          | 0.7823 | 0.7759 |  6.2 Gb   |
+                  +-------------------+      +-------------------------------------------------------------------------------------------------+----------+--------+--------+-----------+
|                  |`RuSentiment`_     |      | :config:`RuWiki+Lenta emb <classifiers/rusentiment_cnn.json>`                                   | F1       | 0.6258 | 0.6472 |  6.2 Gb   |
+                  +                   +      +-------------------------------------------------------------------------------------------------+          +--------+--------+-----------+
|                  |                   |      | :config:`ELMo <classifiers/rusentiment_elmo.json>`                                              |          | 0.7074 | 0.7288 |  700 Mb   |
+------------------+-------------------+      +-------------------------------------------------------------------------------------------------+----------+--------+--------+-----------+
| Intent           |`Yahoo-L31`_       |      | :config:`Yahoo-L31 on ELMo <classifiers/yahoo_convers_vs_info.json>` pre-trained on `Yahoo-L6`_ | ROC-AUC  | 0.9412 |   --   |  700 Mb   |
+------------------+-------------------+------+-------------------------------------------------------------------------------------------------+----------+--------+--------+-----------+

.. _`DSTC 2`: http://camdial.org/~mh521/dstc/
.. _`SNIPS-2017`: https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines
.. _`Insults`: https://www.kaggle.com/c/detecting-insults-in-social-commentary
.. _`AG News`: https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html
.. _`Twitter mokoron`: http://study.mokoron.com/
.. _`RuSentiment`: http://text-machine.cs.uml.edu/projects/rusentiment/
.. _`Yahoo-L31`: https://webscope.sandbox.yahoo.com/catalog.php?datatype=l
.. _`Yahoo-L6`: https://webscope.sandbox.yahoo.com/catalog.php?datatype=l

As no one had published intent recognition for DSTC-2 data, the
comparison of the presented model is given on **SNIPS** dataset. The
evaluation of model scores was conducted in the same way as in [3] to
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



:doc:`Goal-oriented bot </skills/go_bot>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on Hybrid Code Networks (HCNs) architecture from `Jason D. Williams, Kavosh Asadi, 
Geoffrey Zweig, Hybrid Code Networks: practical and efficient end-to-end dialog control 
with supervised and reinforcement learning – 2017 <https://arxiv.org/abs/1702.03274>`__.
It allows to predict responses in a goal-oriented dialog. The model is
customizable: embeddings, slot filler and intent classifier can be switched on and off on demand.

Available pre-trained models and their comparison with existing benchmarks:

+----------------+------+-------------------------------------------------------------------------------------+---------------+---------+------------+------------------+
| Dataset        | Lang | Model                                                                               | Metric        | Valid   | Test       | Downloads        |
+================+======+=====================================================================================+===============+=========+============+==================+
| `DSTC 2`_ [*]_ | En   | :config:`bot with slot filler <go_bot/gobot_dstc2.json>`                            | Turn Accuracy | 0.521   | 0.529      | 400 Mb           |
+                +      +-------------------------------------------------------------------------------------+               +---------+------------+------------------+
|                |      | :config:`bot with slot filler & intents & attention <go_bot/gobot_dstc2_best.json>` |               | 0.555   | **0.561**  | 8.5 Gb           |
+----------------+      +-------------------------------------------------------------------------------------+               +---------+------------+------------------+
| `DSTC 2`_      |      | Bordes and Weston (2016)                                                            |               | --      | 0.411      | --               |
+                +      +-------------------------------------------------------------------------------------+               +---------+------------+------------------+
|                |      | Eric and Manning (2017)                                                             |               | --      | 0.480      | --               |
+                +      +-------------------------------------------------------------------------------------+               +---------+------------+------------------+
|                |      | Perez and Liu (2016)                                                                |               | --      | 0.487      | --               |
+                +      +-------------------------------------------------------------------------------------+               +---------+------------+------------------+
|                |      | Williams et al. (2017)                                                              |               | --      | **0.556**  | --               |
+----------------+------+-------------------------------------------------------------------------------------+---------------+---------+------------+------------------+

.. _`DSTC 2`: http://camdial.org/~mh521/dstc/

.. [*] There were a few :ref:`modifications <dstc2_dataset>` to the original dataset. 


:doc:`Seq2seq goal-oriented bot </skills/seq2seq_go_bot>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dialogue agent predicts responses in a goal-oriented dialog and is able to handle
multiple domains (pretrained bot allows calendar scheduling, weather information retrieval,
and point-of-interest navigation). The model is end-to-end differentiable and
does not need to explicitly model dialogue state or belief trackers.

Comparison of deeppavlov pretrained model with others:

+-------------------+------+----------------------------------------------------+------------------+-----------------+-----------+
| Dataset           | Lang | Model                                              | Valid BLEU       | Test BLEU       | Downloads |
+===================+======+====================================================+==================+=================+===========+
| `Stanford Kvret`_ | En   | :config:`KvretNet <seq2seq_go_bot/bot_kvret.json>` | 0.131            | **0.132**       |  10 Gb    |
+                   +      +----------------------------------------------------+------------------+-----------------+-----------+
|                   |      | KvretNet, Mihail Eric et al. (2017)                | --               | **0.132**       |    --     +
+                   +      +----------------------------------------------------+------------------+-----------------+-----------+
|                   |      | CopyNet, Mihail Eric et al. (2017)                 | --               | 0.110           | --        +
+                   +      +----------------------------------------------------+------------------+-----------------+-----------+
|                   |      | Attn Seq2Seq, Mihail Eric et al. (2017)            | --               | 0.102           | --        +
+                   +      +----------------------------------------------------+------------------+-----------------+-----------+
|                   |      | Rule-based, Mihail Eric et al. (2017)              | --               | 0.066           | --        +
+-------------------+------+----------------------------------------------------+------------------+-----------------+-----------+

.. _`Stanford Kvret`: https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/

:doc:`Automatic spelling correction component </components/spelling_correction>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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



:doc:`Ranking component </components/neural_ranking>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on `LSTM-based deep learning models for non-factoid answer selection <https://arxiv.org/abs/1511.04108>`__. The
model performs ranking of responses or contexts from some database by their relevance for the given context.

Available pre-trained models for ranking:

.. table::
   :widths: auto

   +-------------------+-------------------------------------------------------------------------------------+-----------------------+------------------+----------+
   |    Dataset        | Model config                                                                        | Validation (Recall@1) | Test1 (Recall@1) |Downloads |
   +===================+=====================================================================================+=======================+==================+==========+
   | `InsuranceQA V1`_ | :config:`ranking_insurance_interact <ranking/ranking_insurance_interact.json>`      |   72.0                |   72.2           |8374M     |
   +-------------------+-------------------------------------------------------------------------------------+-----------------------+------------------+----------+
   | `Ubuntu V2`_      |:config:`ranking_ubuntu_v2_interact <ranking/ranking_ubuntu_v2_interact.json>`       |   52.9                |   52.4           |8913M     |
   +-------------------+-------------------------------------------------------------------------------------+-----------------------+------------------+----------+
   | `Ubuntu V2`_      |:config:`ranking_ubuntu_v2_mt_interact <ranking/ranking_ubuntu_v2_mt_interact.json>` |   59.2                |   58.7           |8906M     |
   +-------------------+-------------------------------------------------------------------------------------+-----------------------+------------------+----------+

.. _`InsuranceQA V1`: https://github.com/shuzi/insuranceQA
.. _`Ubuntu V2`: https://github.com/rkadlec/ubuntu-ranking-dataset-creator

Available pre-trained models for paraphrase identification:

.. table::
   :widths: auto

   +------------------------+---------------------------------------------------------------------------------------------+---------------+----------------+---------+----------+---------------+----------------+----------+
   |    Dataset             |Model config                                                                                 | Val (accuracy)| Test (accuracy)| Val (F1)| Test (F1)| Val (log_loss)| Test (log_loss)|Downloads |
   +========================+=============================================================================================+===============+================+=========+==========+===============+================+==========+
   |`paraphraser.ru`_       |:config:`paraphrase_ident_paraphraser <ranking/paraphrase_ident_paraphraser_interact.json>`  |   83.8        |   75.4         |   87.9  |  80.9    |   0.468       |   0.616        |5938M     |
   +------------------------+---------------------------------------------------------------------------------------------+---------------+----------------+---------+----------+---------------+----------------+----------+
   |`paraphraser.ru`_       |:config:`paraphrase_ident_paraphraser <ranking/paraphrase_ident_elmo_interact.json>`         |   82.7        |   76.0         |   87.3  |  81.4    |   0.391       |   0.510        |5938M     |
   +------------------------+---------------------------------------------------------------------------------------------+---------------+----------------+---------+----------+---------------+----------------+----------+
   |`paraphraser.ru`_       |:config:`paraphrase_ident_paraphraser_tune <ranking/paraphrase_ident_tune_interact.json>`    |   82.9        |   76.7         |   87.3  |  82.0    |   0.392       |   0.479        |5938M     |
   +------------------------+---------------------------------------------------------------------------------------------+---------------+----------------+---------+----------+---------------+----------------+----------+
   |`Quora Question Pairs`_ |:config:`paraphrase_ident_qqp <ranking/paraphrase_ident_qqp_bilstm_interact.json>`           |   87.1        |   87.0         |   83.0  |  82.6    |   0.300       |   0.305        |8134M     |
   +------------------------+---------------------------------------------------------------------------------------------+---------------+----------------+---------+----------+---------------+----------------+----------+
   |`Quora Question Pairs`_ |:config:`paraphrase_ident_qqp <ranking/paraphrase_ident_qqp_interact.json>`                  |   87.7        |   87.5         |   84.0  |  83.8    |   0.287       |   0.298        |8136M     |
   +------------------------+---------------------------------------------------------------------------------------------+---------------+----------------+---------+----------+---------------+----------------+----------+

.. _`paraphraser.ru`: https://paraphraser.ru/
.. _`Quora Question Pairs`: https://www.kaggle.com/c/quora-question-pairs/data

Comparison with other models on the `InsuranceQA V1 <https://github.com/shuzi/insuranceQA>`__:

+------------------------------------------------------------------------+-------------------------+--------------------+
| Model                                                                  | Validation (Recall@1)   | Test1 (Recall@1)   |
+========================================================================+=========================+====================+
| `Architecture II (HLQA(200) CNNQA(4000) 1-MaxPooling Tanh)`_           | 61.8                    | 62.8               |
+------------------------------------------------------------------------+-------------------------+--------------------+
| `QA-LSTM basic-model(max pooling)`_                                    | 64.3                    | 63.1               |
+------------------------------------------------------------------------+-------------------------+--------------------+
| :config:`ranking_insurance <ranking/ranking_insurance_interact.json>`  | **72.0**                | **72.2**           |
+------------------------------------------------------------------------+-------------------------+--------------------+

.. _`Architecture II (HLQA(200) CNNQA(4000) 1-MaxPooling Tanh)`: https://arxiv.org/pdf/1508.01585.pdf
.. _`QA-LSTM basic-model(max pooling)`: https://arxiv.org/pdf/1511.04108.pdf

:doc:`TF-IDF Ranker component </components/tfidf_ranking>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on `Reading Wikipedia to Answer Open-Domain Questions <https://github.com/facebookresearch/DrQA/>`__. The model solves the task of document retrieval for a given query.

+---------------+-------------------------------------------------------------------+----------------------+-----------------+-----------+
| Dataset       | Model                                                             |  Wiki dump           |  Recall@5       | Downloads |
+===============+========================================================+==========+======================+=================+===========+
| `SQuAD-v1.1`_ | :config:`doc_retrieval <doc_retrieval/en_ranker_tfidf_wiki.json>` |  enwiki (2018-02-11) |   75.6          | 33 GB     |
+---------------+-------------------------------------------------+-----------------+----------------------+-----------------+-----------+


:doc:`Question Answering component </components/squad>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on `R-NET: Machine Reading Comprehension with Self-matching Networks
<https://www.microsoft.com/en-us/research/publication/mcr/>`__. The model solves the task of looking for an answer on a
question in a given context (`SQuAD <https://rajpurkar.github.io/SQuAD-explorer/>`__ task format).

.. note::

    All pre-trained models could be downloaded. Model for English language will download about 2.5 Gb and model for Russian about 5 Gb.

+---------------+---------------------------------------------+-------+----------------+-----------------+
|    Dataset    | Model config                                | lang  |    EM (dev)    |    F-1 (dev)    |
+===============+=============================================+=======+================+=================+
| `SQuAD-v1.1`_ | :config:`squad <squad/squad.json>`          |  en   |     71.49      |     80.34       |
+---------------+---------------------------------------------+-------+----------------+-----------------+
|  SDSJ Task B  | :config:`squad_ru <squad/squad_ru.json>`    |  ru   |     60.62      |     80.04       |
+---------------+---------------------------------------------+-------+----------------+-----------------+

In the case when answer is not necessary present in given context we have :config:`squad_noans <squad/multi_squad_noans.json>`
model. This model outputs empty string in case if there is no answer in context.


:doc:`Morphological tagging component </components/morphotagger>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on character-based approach to morphological tagging `Heigold et al., 2017. An extensive empirical evaluation of
character-based morphological tagging for 14 languages <http://www.aclweb.org/anthology/E17-1048>`__. A state-of-the-art
model for Russian and several other languages. Model takes as input tokenized sentences and outputs the corresponding
sequence of morphological labels in `UD format <http://universaldependencies.org/format.html>`__. The table below
contains word and sentence accuracy on UD2.0 datasets. For more scores see :doc:`full table </components/morphotagger>`.

.. table::
    :widths: auto

    +----------------------+--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    |    Dataset           | Model                                                                                                        | Word accuracy | Sent. accuracy | Download size (MB) |
    +======================+==============================================================================================================+===============+================+====================+
    |                      |`Pymorphy`_ + `russian_tagsets`_ (first tag)                                                                  |     60.93     |      0.00      |                    |
    +                      +--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    |`UD2.0`_ (Russian)    |`UD Pipe 1.2`_ (Straka et al., 2017)                                                                          |     93.57     |     43.04      |                    |
    +                      +--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    |                      |:config:`Basic model <morpho_tagger/UD2.0/ru_syntagrus/morpho_ru_syntagrus.json>`                             |     95.17     |     50.58      |        48.7        |
    +                      +--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    |                      |:config:`Pymorphy-enhanced model <morpho_tagger/UD2.0/ru_syntagrus/morpho_ru_syntagrus_pymorphy.json>`        |   **96.23**   |     58.00      |        48.7        |
    +----------------------+--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    | `UD2.0`_ (Czech)     |`UD Pipe 1.2`_ (Straka et al., 2017)                                                                          |     91.86     |     42.28      |                    |
    |                      +--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    |                      |:config:`Basic model <morpho_tagger/UD2.0/morpho_cs.json>`                                                    |   **94.35**   |     51.56      |        41.8        |
    +----------------------+--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    |`UD2.0`_ (English)    |`UD Pipe 1.2`_ (Straka et al., 2017)                                                                          |     92.89     |     55.75      |                    |
    |                      +--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    |                      |:config:`Basic model <morpho_tagger/UD2.0/morpho_en.json>`                                                    |   **93.00**   |     55.18      |        16.9        |
    +----------------------+--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    |`UD2.0`_ (German)     |`UD Pipe 1.2`_ (Straka et al., 2017)                                                                          |     76.65     |     10.24      |                    |
    |                      +--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+
    |                      |:config:`Basic model <morpho_tagger/UD2.0/morpho_de.json>`                                                    |   **83.83**   |     15.25      |        18.6        |
    +----------------------+--------------------------------------------------------------------------------------------------------------+---------------+----------------+--------------------+

.. _`Pymorphy`: https://pymorphy2.readthedocs.io/en/latest/
.. _`russian_tagsets`: https://github.com/kmike/russian-tagsets
.. _`UD2.0`: https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1983
.. _`UD Pipe 1.2`: http://ufal.mff.cuni.cz/udpipe

:doc:`Frequently Asked Questions (FAQ) component </skills/faq>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set of pipelines for FAQ task: classifying incoming question into set of known questions and return prepared answer.
You can build different pipelines based on: tf-idf, weighted fasttext, cosine similarity, logistic regression.


Skills
------

:doc:`eCommerce bot </skills/ecommerce>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The eCommerce bot intends to retrieve product items from catalog in sorted order. In addition, it asks an user to provide additional information to specify the search.

.. note::

    About **130 Mb** on disc required for eCommerce bot with TfIdf-based ranker and **500 Mb** for BLEU-based ranker.


:doc:`ODQA </skills/odqa>`
~~~~~~~~~~~~~~~~~~~~~~~~~~

An open domain question answering skill. The skill accepts free-form questions about the world and outputs an answer
based on its Wikipedia knowledge.


+----------------+----------------------------------------------+-----------------------+--------+-----------+
| Dataset        | Model config                                 |  Wiki dump            |   F1   | Downloads |
+================+==============================================+=======================+========+===========+
| `SQuAD-v1.1`_  |:config:`ODQA <odqa/en_odqa_infer_wiki.json>` | enwiki (2018-02-11)   |  28.0  | 42 GB     |
+----------------+----------------------------------------------+-----------------------+--------+-----------+


AutoML
--------------------

:doc:`Hyperparameters optimization </intro/hypersearch>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hyperparameters optimization (either by cross-validation or neural evolution) for DeepPavlov models
that requires only some small changes in a config file.


Embeddings
----------

:doc:`Pre-trained embeddings for the Russian language </intro/pretrained_vectors>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


.. _`SQuAD-v1.1`: https://arxiv.org/abs/1606.05250

