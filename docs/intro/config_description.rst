Configuration files
===================

An NLP pipeline config is a JSON file that contains one required element
``chainer``:

::

    {
      "chainer": {
        "in": ["x"],
        "in_y": ["y"],
        "pipe": [
          ...
        ],
        "out": ["y_predicted"]
      }
    }

Chainer is a core concept of DeepPavlov library: chainer builds a
pipeline from heterogeneous components
(rule-based/ml/dl) and allows to train or infer from pipeline as a
whole. Each component in the pipeline specifies
its inputs and outputs as arrays of names, for example:
``"in": ["tokens", "features"]`` and
``"out": ["token_embeddings", "features_embeddings"]`` and you can chain
outputs of one components with inputs of other components:

.. code:: python

    {
      "class": "deeppavlov.models.preproccessors.str_lower:StrLower",
      "in": ["x"],
      "out": ["x_lower"]
    },
    {
      "name": "nltk_tokenizer",
      "in": ["x_lower"],
      "out": ["x_tokens"]
    },

Each :class:`~deeppavlov.core.models.component.Component`
in the
pipeline must implement method ``__call__`` and has ``name`` parameter,
which is its registered codename,
or ``class`` parameter in the form of ``module_name:ClassName``.
It can also have any other parameters which repeat its ``__init__()``
method arguments.
Default values of ``__init__()`` arguments will be overridden with the
config values during the initialization of a class instance.

You can reuse components in the pipeline to process different parts of
data with the help of ``id`` and ``ref`` parameters:

.. code:: python

    {
      "name": "nltk_tokenizer",
      "id": "tokenizer",
      "in": ["x_lower"],
      "out": ["x_tokens"]
    },
    {
      "ref": "tokenizer",
      "in": ["y"],
      "out": ["y_tokens"]
    },

Training
--------

There are two abstract classes for trainable components: **Estimator**
and **NNModel**.
`**Estimators** <deeppavlov/core/models/estimator.py>`__ are fit once
on any data with no batching or early stopping,
so it can be safely done at the time of pipeline initialization.
``fit`` method has to be implemented for each Estimator. An example of
Estimator is `Vocab <deeppavlov/core/data/vocab.py>`__.
`**NNModel** <deeppavlov/core/models/nn_model.py>`__ requires more
complex training. It can only be trained in a supervised mode (as
opposed to **Estimator** which can be trained in both supervised and
unsupervised settings). This process takes multiple epochs with periodic
validation and logging.
``train_on_batch`` method has to be implemented for each NNModel.

Training is triggered by
``deeppavlov.core.commands.train.train_model_from_config()`` function.

Train config
------------

Estimators that are trained should also have ``fit_on`` parameter
which contains a list of input parameter names.
An NNModel should have the ``in_y`` parameter which contains a list of
ground truth answer names. For example:

.. code:: json

    [
      {
        "id": "classes_vocab",
        "name": "default_vocab",
        "fit_on": ["y"],
        "level": "token",
        "save_path": "vocabs/classes.dict",
        "load_path": "vocabs/classes.dict"
      },
      {
        "in": ["x"],
        "in_y": ["y"],
        "out": ["y_predicted"],
        "name": "intent_model",
        "save_path": "intents/intent_cnn",
        "load_path": "intents/intent_cnn",
        "classes_vocab": {
          "ref": "classes_vocab"
        }
      }
    ]

The config for training the pipeline should have three additional
elements: ``dataset_reader``, ``dataset_iterator`` and ``train``:

::

    {
      "dataset_reader": {
        "name": ...,
        ...
      }
      "dataset_iterator": {
        "name": ...,
        ...
      },
      "chainer": {
        ...
      }
      "train": {
        ...
      }
    }

Simplified version of trainig pipeline contains two elemens:
``dataset`` and ``train``. The ``dataset`` element currently
can be used for train from classification data in ``csv`` and ``json``
formats. You can find complete examples of how to use simplified
training pipeline in
`intents\_sample\_csv.json <deeppavlov/configs/intents/intents_sample_csv.json>`__
and
`intents\_sample\_json.json <deeppavlov/configs/intents/intents_sample_json.json>`__
config files.

Train Parameters
----------------

-  ``epochs`` — maximum number of epochs to train NNModel, defaults to
   ``-1`` (infinite)
-  ``batch_size``,
-  ``metrics`` — list of names of `registered
   metrics <deeppavlov/metrics>`__ to evaluate the model. The first
   metric in the list
   is used for early stopping
-  ``metric_optimization`` — ``maximize`` or ``minimize`` a metric,
   defaults to ``maximize``
-  ``validation_patience`` — how many times in a row the validation
   metric has to not improve for early stopping, defaults to ``5``
-  ``val_every_n_epochs`` — how often to validate the pipe, defaults to
   ``-1`` (never)
-  ``log_every_n_batches``, ``log_every_n_epochs`` — how often to
   calculate metrics for train data, defaults to ``-1`` (never)
-  ``validate_best``, ``test_best`` flags to infer the best saved model
   on valid and test data, defaults to ``true``

DatasetReader
-------------

``DatasetReader`` class reads data and returns it in a specified
format.
A concrete ``DatasetReader`` class should be inherited from the base
``deeppavlov.data.dataset_reader.DatasetReader`` class and registered
with a codename:

.. code:: python

    from deeppavlov.core.common.registry import register
    from deeppavlov.core.data.dataset_reader import DatasetReader

    @register('dstc2_datasetreader')
    class DSTC2DatasetReader(DatasetReader):

DatasetIterator
---------------

``DatasetIterator`` forms the sets of data ('train', 'valid', 'test')
needed for training/inference and divides it into batches.
A concrete ``DatasetIterator`` class should be registered and can be
inherited from
``deeppavlov.data.dataset_iterator.BasicDatasetIterator`` class.
``deeppavlov.data.dataset_iterator.BasicDatasetIterator``
is not an abstract class and can be used as a ``DatasetIterator`` as
well.

Inference
---------

All components inherited from
``deeppavlov.core.models.component.Component`` abstract class can be
used for inference. The ``__call__()`` method should return standard
output of a component. For example, a *tokenizer* should return
*tokens*, a *NER recognizer* should return *recognized entities*, a
*bot* should return an *utterance*.
A particular format of returned data should be defined in
``__call__()``.

Inference is triggered by
``deeppavlov.core.commands.infer.interact_model()`` function. There is
no need in a separate JSON for inference.
