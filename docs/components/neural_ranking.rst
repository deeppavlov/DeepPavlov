Ranking and paraphrase identification
=====================================

This library component solves the tasks of ranking and paraphrase identification based on semantic similarity
which is trained with siamese neural networks. The trained network can retrieve the response
closest semantically to a given context from some database or answer whether two sentences are paraphrases or not.
It is possible to build automatic semantic FAQ systems with such neural architectures.

Training and inference models on predifined datasets
----------------------------------------------------

Ranking
~~~~~~~

Before using the model make sure that all required packages are installed running the command:

.. code:: bash

    python -m deeppavlov install ranking_insurance

To train the model on the `InsuranceQA V1`_ dataset one can use the following code in python:

.. code:: python

    from deeppavlov import configs, train_model

    rank_model = train_model(configs.ranking.ranking_insurance, download=True)

To train from command line:

::

    python -m deeppavlov train deeppavlov/configs/ranking/ranking_insurance.json [-d]

As an example of configuration file see
:config:`ranking_insurance.json <ranking/ranking_insurance.json>`.


To use the model trained on the `InsuranceQA V1`_ dataset for
inference one can use the following code in python:

.. code:: python

    from deeppavlov import build_model, configs

    rank_model = build_model(configs.ranking.ranking_insurance_interact, download=True)
    rank_model(['how much to pay for auto insurance?'])

    >>> ['the cost of auto insurance be based on several factor include your driving record , claim history , type of vehicle , credit score where you live and how far you travel to and from work I will recommend work with an independent agent who can shop several company find the good policy for you', 'there be not any absolute answer to this question rate for auto insurance coverage can vary greatly from carrier to carrier and from area to area contact local agent in your area find out about coverage availablity and pricing within your area look for an agent that you be comfortable working with as they will be the first last point of contact in most instance', 'the cost of auto insurance coverage for any vehicle or driver can vary greatly thing that effect your auto insurance rate be geographical location , vehicle , age (s) of driver (s) , type of coverage desire , motor vehicle record of all driver , credit rating of all driver and more contact a local agent get a quote a quote cost nothing but will let you know where your rate will']


By default the model returns the ``interact_pred_num`` most relevant responses from all responses the model saw during training time.
To get predictions on your own list of responses use the following code:

.. code:: python

    from deeppavlov import build_model, configs

    rank_model = build_model(configs.ranking.ranking_insurance_interact, download=True)
    predictor = rank_model.pipe[-1][-1]
    candidates = ['auto insurance', 'life insurance', 'home insurance']
    predictor.rebuild_responses(candidates)
    rank_model(['how much to pay for auto insurance?'])

    >>> [['auto insurance']]

If the model with multi-turn context is used
(such as :class:`~deeppavlov.models.ranking.bilstm_gru_siamese_network.BiLSTMGRUSiameseNetwork`
with the parameter ``num_context_turns`` set to the value higher than 1 in the configuration JSON file)
then the ``context`` to evaluate should consist of ``num_context_turns`` strings connected by the ampersand.
Some of these strings can be empty, i.e. equal to ``''``.

To run the model for inference from command line:

::

    python -m deeppavlov interact deeppavlov/configs/ranking/ranking_insurance_interact.json [-d]

Then a user can enter a context and get responses:

::

    :: how much to pay for auto insurance?
    >> ['the cost of auto insurance be based on several factor include your driving record , claim history , type of vehicle , credit score where you live and how far you travel to and from work I will recommend work with an independent agent who can shop several company find the good policy for you', 'there be not any absolute answer to this question rate for auto insurance coverage can vary greatly from carrier to carrier and from area to area contact local agent in your area find out about coverage availablity and pricing within your area look for an agent that you be comfortable working with as they will be the first last point of contact in most instance', 'the cost of auto insurance coverage for any vehicle or driver can vary greatly thing that effect your auto insurance rate be geographical location , vehicle , age (s) of driver (s) , type of coverage desire , motor vehicle record of all driver , credit rating of all driver and more contact a local agent get a quote a quote cost nothing but will let you know where your rate will']


Paraphrase identification
~~~~~~~~~~~~~~~~~~~~~~~~~

Paraphraser.ru dataset
~~~~~~~~~~~~~~~~~~~~~~

Before using the model make sure that all required packages are installed running the command:

.. code:: bash

    python -m deeppavlov install paraphrase_ident_paraphraser
    python -m deeppavlov install elmo_paraphraser_fine_tuning
    python -m deeppavlov install paraphrase_ident_paraphraser_elmo
    python -m deeppavlov install paraphrase_ident_paraphraser_pretrain
    python -m deeppavlov install paraphrase_ident_paraphraser_tune

To train the model on the `paraphraser.ru`_ dataset with fasttext embeddings one can use the following code in python:

.. code:: python

    from deeppavlov import configs, train_model

    para_model = train_model(configs.ranking.paraphrase_ident_paraphraser, download=True)


To train the model on the `paraphraser.ru`_ dataset with fine-tuned ELMO embeddings one should first fine-tune ELMO embeddings:

.. code:: python

    from deeppavlov import configs, train_model

    para_model = train_model(configs.elmo.elmo_paraphraser_fine_tuning, download=True)

To train the model itself with fine-tuned embeddings:

.. code:: python

    from deeppavlov import configs, train_model

    para_model = train_model(configs.elmo.paraphrase_ident_paraphraser_elmo, download=True)

The fine-tuned ELMO embeddings obtained at the previous step can be downloaded directly
from the :config:`paraphrase_ident_paraphraser_elmo.json <ranking/paraphrase_ident_paraphraser_elmo.json>`.

To train the model on the `paraphraser.ru`_ dataset with pre-training one should first train the model
on the additionally collected dataset:

.. code:: python

    from deeppavlov import configs, train_model

    para_model = train_model(configs.elmo.paraphrase_ident_paraphraser_pretrain, download=True)

To fine-tune the model on the target dataset:

.. code:: python

    from deeppavlov import configs, train_model

    para_model = train_model(configs.elmo.paraphrase_ident_paraphraser_tune , download=True)

The pre-trained model obtained at the previous step can be downloaded directly
from the :config:`paraphrase_ident_paraphraser_tune.json <ranking/paraphrase_ident_paraphraser_tune.json>`.

To use the model trained on the `paraphraser.ru`_ dataset for
inference, one can use the following code in python:

.. code:: python

    from deeppavlov import build_model, configs

    para_model = build_model(configs.ranking.deeppavlov/configs/ranking/paraphrase_ident_tune_interact, download=True)
    para_model(['9 мая метрополитен Петербурга будет работать круглосуточно&Петербургское метро в ночь на 10 мая будет работать круглосуточно'])
    >>> 'This is a paraphrase.'

Quora question pairs dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before using the model make sure that all required packages are installed running the command:

.. code:: bash

    python -m deeppavlov install paraphrase_ident_qqp

To train the model on the `Quora Question Pairs`_ dataset one can use the following code in python:

.. code:: python

    from deeppavlov import configs, train_model

    para_model = train_model(configs.ranking.paraphrase_ident_qqp, download=True)

To train from command line:

::

    python -m deeppavlov train deeppavlov/configs/ranking/paraphrase_ident_qqp.json [-d]

As an example of configuration file see
:config:`paraphrase_ident_qqp.json <ranking/paraphrase_ident_qqp.json>`.


To use the model trained on the `Quora Question Pairs`_ dataset for
inference, one can use the following code in python:

.. code:: python

    from deeppavlov import build_model, configs

    para_model = build_model(configs.ranking.paraphrase_ident_qqp_interact, download=True)
    para_model(['How can I be a good geologist?&What should I do to be a great geologist?'])
    >>> 'This is a paraphrase.'

Note that two sentences to evaluate are connected by the ampersand.

To use the model for inference from command line:

::

    python -m deeppavlov interact deeppavlov/configs/ranking/paraphrase_ident_qqp_interact.json [-d]

Now a user can enter two sentences and the model will make a prediction whether these sentences are paraphrases or not.

::

    :: How can I be a good geologist?&What should I do to be a great geologist?
    >> This is a paraphrase.

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

.. _`InsuranceQA V1`: https://github.com/shuzi/insuranceQA
.. _`paraphraser.ru`: https://paraphraser.ru
.. _`Quora Question Pairs`: https://www.kaggle.com/c/quora-question-pairs/data
