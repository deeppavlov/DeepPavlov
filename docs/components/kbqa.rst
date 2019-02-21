Knowledge Base Question Answering (KBQA)
========================================

Use the model
-------------

Any pre-trained model can be used for inference from both Command Line Interface (CLI) and Python. Before using the
model make sure that all required packages are installed using the command:

.. code:: bash

    python -m deeppavlov install kbqa_mix_lowercase

To use a pre-trained model from CLI use the following command:

.. code:: bash

    python deeppavlov/deep.py interact kbqa_mix_lowercase [-d]

where ``kbqa_mix_lowercase`` is the name of the config and ``-d`` is an optional download key. The key ``-d`` is used
to download the pre-trained model along with embeddings and all other files needed to run the model. Also command
``download`` is possible,


Available config:

.. table::
    :widths: auto

    +-----------------------------------------------+-------------------+-----------------+------------+
    | Model                                         | Dataset           | Embeddings Size | Model Size |
    +===============================================+===================+=================+============+
    | :config:`kbqa <kbqa/kbqa_mix_lowercase.json>` | Simple Questions  |     7.7 GB      |   8.9 MB   |
    |                                               | and Zero-Shot IE  |                 |            |
    +-----------------------------------------------+-------------------+-----------------+------------+


Models can be used from Python using the following code:

.. code:: python

    from deeppavlov import configs, build_model

    kbqa_model = build_model(configs.kbqa.kbqa_mix_lowercase, download=True)
    kbqa_model(['Когда родился Пушкин?'])
    >>> ["26 May 1799"]
