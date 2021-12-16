Ranking and paraphrase identification
=====================================

This library model solves the tasks of ranking and paraphrase identification based on semantic similarity
which is trained with siamese neural networks. The trained network can retrieve the response
closest semantically to a given context from some database or answer whether two sentences are paraphrases or not.
It is possible to build automatic semantic FAQ systems with such neural architectures.

Training and inference models on predifined datasets
----------------------------------------------------

BERT Ranking
~~~~~~~~~~~~

Before using models make sure that all required packages are installed running the command for TensorFlow:

.. code:: bash

    python -m deeppavlov install ranking_ubuntu_v2_bert_uncased
    python -m deeppavlov install ranking_ubuntu_v2_bert_sep
    python -m deeppavlov install ranking_ubuntu_v2_bert_sep_interact

or on PyTorch:

.. code:: bash

    python -m deeppavlov install ranking_ubuntu_v2_torch_bert_uncased


To train the interaction-based (accurate, slow) model on the `Ubuntu V2`_ from command line:

::

    python -m deeppavlov train ranking_ubuntu_v2_bert_uncased [-d]

To train the representation-based (accurate, fast) model on the `Ubuntu V2`_ from command line:

::

    python -m deeppavlov train ranking_ubuntu_v2_bert_sep [-d]

Further the trained representation-based model can be run for inference over the provided response base
(~500K in our case) from command line:

::

    python -m deeppavlov interact ranking_ubuntu_v2_bert_sep_interact [-d]

Statistics on the models quality are available :doc:`here </features/overview>`.

Building your own response base for bert ranking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the BERT-based models we have the following mechanism of building your own response base.
If you run ``python -m deeppavlov download ranking_ubuntu_v2_bert_sep_interact`` in console
the model with the existing base will be downloaded.
If you look in the folder where the model with the base is located you will find four files:
``contexts.csv``, ``responses.csv``, ``cont_vecs.npy``, ``resp_vecs.npy``.
These are possible responses with their corresponding contexts (``.csv`` files) and their vector representations (``.npy`` files)
indexed using the model. Contexts for responses are used as additional features in some modes of the model operation
(see the attribute ``interact_mode`` in the class :class:`~deeppavlov.models.preprocessors.bert_preprocessor.BertSepRankerPredictorPreprocessor`).
If you would like to use your own response base you should remove all four files indicated above
and place your own ``responses.csv`` file in the folder,
and probably ``contexts.csv`` file depending on the value of the ``interact_mode`` you are planning to use.
The format of these files is very simple, namely each line should represent single response (or context).
You can use existing files as an example. Numbers of lines in ``responses.csv`` and ``contexts.csv`` must match exactly.
Once you have provided these files, you can run the above command in console.
As the system will not find vector representations, it will build them first.
You will see the message ``Building BERT features for the response base...``
(and probably ``Building BERT features for the context base...``) and then
``Building BERT vector representations for the response base...``
(and probably ``Building BERT vector representations for the context base...``).
After this is done, you will be able to interact with the system.
Next time you will use the model, built vector representations will be loaded.

Ranking
~~~~~~~

To use Sequential Matching Network (SMN) or Deep Attention Matching Network (DAM) or
Deep Attention Matching Network with Universal Sentence Encoder (DAM-USE-T)
on the `Ubuntu V2`_ for inference, please run one of the following commands:

::

    python -m deeppavlov interact -d ranking_ubuntu_v2_mt_word2vec_smn
    python -m deeppavlov interact -d ranking_ubuntu_v2_mt_word2vec_dam_transformer

Now a user can enter a dialog consists of 10 context sentences and several (>=1) candidate response sentences separated by '&'
and then get the probability that the response is proper continuation of the dialog:

::

    :: & & & & & & & & bonhoeffer  whar drives do you want to mount what &  i have an ext3 usb drive  & look with fdisk -l & hello there & fdisk is all you need
    >> [0.9776373  0.05753616 0.9642599 ]

To train the models on the `Ubuntu V2`_ dataset please run one of the following commands:

::

    python -m deeppavlov train -d ranking_ubuntu_v2_mt_word2vec_smn
    python -m deeppavlov train -d ranking_ubuntu_v2_mt_word2vec_dam_transformer

As an example of configuration file see
:config:`ranking_ubuntu_v2_mt_word2vec_smn.json <ranking/ranking_ubuntu_v2_mt_word2vec_smn.json>`.

If the model with multi-turn context is used
(such as :class:`~deeppavlov.models.ranking.bilstm_gru_siamese_network.BiLSTMGRUSiameseNetwork`
with the parameter ``num_context_turns`` set to the value higher than 1 in the configuration JSON file)
then the ``context`` to evaluate should consist of ``num_context_turns`` strings connected by the ampersand.
Some of these strings can be empty, i.e. equal to ``''``.


Paraphrase identification
~~~~~~~~~~~~~~~~~~~~~~~~~

Paraphraser.ru dataset
~~~~~~~~~~~~~~~~~~~~~~

Before using the model make sure that all required packages are installed running the command:

.. code:: bash

    python -m deeppavlov install paraphrase_ident_paraphraser

To train the model on the `paraphraser.ru`_ dataset with fasttext embeddings one can use the following code in python:

.. code:: python

    from deeppavlov import configs, train_model

    para_model = train_model(configs.ranking.paraphrase_ident_paraphraser, download=True)

Training and inference on your own data
---------------------------------------

Ranking
~~~~~~~

To train the model for ranking on your own data you should write your own :class:`~deeppavlov.core.data.dataset_reader.DatasetReader` component
or you can use default :class:`~deeppavlov.dataset_readers.siamese_reader.SiameseReader`. In the latter case, you should provide
three separate files in the default data format described below:

**train.csv**: each line in the file contains ``context``, ``response`` and ``label`` separated by the tab key. ``label`` can be
binary, i.e. 1 or 0 corresponding to the correct or incorrect ``response`` for the given ``context``, or it can be multi-class label.
In the latter case, each unique ``context`` has the unique class ``label`` and the only correct ``response`` is indicated for each ``context``.
Currently, all ranking and paraphrase identification models support `cross-entropy loss` training with binary labels.
Some models, such as :class:`~deeppavlov.models.ranking.bilstm_siamese_network.BiLSTMSiameseNetwork`,
:class:`~deeppavlov.models.ranking.bilstm_gru_siamese_network.BiLSTMGRUSiameseNetwork`
and :class:`~deeppavlov.models.ranking.mpm_siamese_network.MPMSiameseNetwork` support also training with `triplet loss`
(the parameter ``triplet_loss`` should be set to ``true`` for the model in the configuration JSON file in this case)
which can give potentially few percent of performance over the `cross-entropy loss` training.

If the model with multi-turn context is used
(such as :class:`~deeppavlov.models.ranking.bilstm_gru_siamese_network.BiLSTMGRUSiameseNetwork`
with the parameter ``num_context_turns`` set to the value higher than 1 in the configuration JSON file)
then the ``context`` should be specified with ``num_context_turns`` strings separated by the tab key instead of a single string.
Some of these strings can be empty, i.e. equal to ``''``.

Classification metrics on the train dataset part (the parameter ``train_metrics`` in the JSON configuration file)
such as ``f1``, ``acc`` and ``log_loss``  can be calculated only in the ``cross-entropy loss`` training mode.
Both, `cross-entropy loss` and `triplet loss` training can output loss function value returned by
:meth:`~deeppavlov.models.ranking.siamese_model.SiameseModel.train_on_batch` if the ``log_every_n_batches`` parameter is set to the non-negative value.


**valid.csv**, **test.csv**: each line in these files contains ``context``, ``response_1``, ``response_2``, ..., ``response_n``
separated by the tab key, where ``response_1`` is the correct response for the given ``context`` and the rest ``response_2``, ..., ``response_n``
are incorrect response candidates. The number of responses `n` in these files should correspond to the
parameter ``num_ranking_samples`` in the JSON configuration file. As an example see

Such ranking metrics on the valid and test parts of the dataset (the parameter ``metrics`` in the JSON configuration file) as
``r@1``, ``r@2``, ..., ``r@n`` and ``rank_response`` can be evaluated.

As an example of data usage in the default format, please, see :config:`ranking_default.json <ranking/ranking_default.json>`.
To train the model with this configuration file in python:

.. code:: python

    from deeppavlov import configs, train_model

    rank_model = train_model(configs.ranking.ranking_default, download=True)

To train from command line:

::

    python -m deeppavlov train deeppavlov/configs/ranking/ranking_default.json [-d]

Paraphrase identification
~~~~~~~~~~~~~~~~~~~~~~~~~

**train.csv**: the same as for ranking.

**valid.csv**, **test.csv**: each line in the file contains ``context``, ``response`` and ``label`` separated by the tab key. ``label`` is
binary, i.e. 1 or 0 corresponding to the correct or incorrect ``response`` for the given ``context``.
Instead of ``response`` and ``context`` it can be simply two phrases which are paraphrases or non-paraphrases as indicated by the ``label``.

Classification metrics on the valid and test dataset parts (the parameter ``metrics`` in the JSON configuration file)
such as ``f1``, ``acc`` and ``log_loss``  can be calculated.

.. _`paraphraser.ru`: https://paraphraser.ru
.. _`Ubuntu V2`: https://github.com/rkadlec/ubuntu-ranking-dataset-creator
