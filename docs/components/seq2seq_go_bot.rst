Sequence-To-Sequence Dialogue Bot For Goal-Oriented Task
========================================================

A goal-oriented bot based on a rnn that encodes user utterance and generates 
response in a sequence-to-sequence manner. For network architecture is similar
to `[2] <#references>`__.

The dialogue agent is able to handle multi-domain data. The model is
end-to-end differentiable and does not need to explicitly model dialogue 
state or belief trackers.

We also used a new Stanford NLP Group released dataset of 3,031 dialogues
`[1] <#references>`__ that are grounded through underlying knowledge bases
and span three distinct tasks in the in-car personal assistant space:

    - calendar scheduling,
    - weather information retrieval and
    - point-of-interest navigation.

For more info on the dataset see :class:`~deeppavlov.dataset_readers.kvret_reader.KvretDatasetReader`.

Here is a simple example of interaction with a trained dialogue bot (available for download):

::

    x_text:: Where's the nearest parking garage?
    >> the nearest is dish parking at dish_parking_address. is that okay?
    x_text:: Yes, please set directions via a route that avoids all heavy traffic if possible.
    >> it looks like there is a road block being reported on the route but i will still find the quickest route to dish_parking_address.
    x_text:: Thanks so much for your help.
    >> you're welcome. have a good day. end_of_dialogue

    x_text:: Will it be warm in Camarillo over the next 2 days?
    >> no, it's not gon na be warm in camarillo over the next 2 days
    x_text:: Thank you car!
    >> you're welcome! end_of_dialogue

    x_text:: What day is my lab appointment, and who all will be there?
    >> your lab appointment is on lab_appointment_date at lab_appointment_time with lab_appointment_party.
    x_text:: Thanks!
    >> you're welcome.

Configs
^^^^^^^

Config :config:`configs/seq2seq_go_bot/bot_kvret_infer.json <seq2seq_go_bot/bot_kvret_infer.json>` is recommended to be used for inference (interaction) of a pretrained model.

Config :config:`configs/seq2seq_go_bot/bot_kvret.json <seq2seq_go_bot/bot_kvret.json>` is recommended to be used for training a new model.

Usage
-----
 
To interact with a pretrained seq2seq\_go\_bot model using commandline run:

.. code:: bash 
 
    python -m deeppavlov interact <path_to_config> [-d] 
 
where ``<path_to_config>`` is one of the :config:`provided config files <seq2seq_go_bot>`. 

You can also train your own model by running:

.. code:: bash 
 
    python -m deeppavlov train <path_to_config> [-d] 

The ``-d`` parameter downloads:
    - data required to train your model (embeddings, etc.);
    - a pretrained model when you use :config:`configs/seq2seq_go_bot/bot_kvret_infer.json <seq2seq_go_bot/bot_kvret_infer.json>` or :config:`configs/seq2seq_go_bot/bot_kvret.json <seq2seq_go_bot/bot_kvret.json>`.

After downloading required files you can use the configs in your python code.

To infer from a pretrained model with config path equal to ``<path_to_config>``:

.. code:: python

    from deeppavlov.core.commands.infer import build_model_from_config
    from deeppavlov.core.common.file import read_json

    CONFIG_PATH = '<path_to_config>'
    model = build_model_from_config(read_json(CONFIG_PATH))

    utterance = ""
    while utterance != 'exit':
        print(">> " + model([utterance])[0])
        utterance = input(':: ')

Config parameters:
^^^^^^^^^^^^^^^^^^

To configure your own pipelines that contain a ``"seq2seq_go_bot"`` component, refer to documentation for :class:`~deeppavlov.models.seq2seq_go_bot.bot.Seq2SeqGoalOrientedBot` and :class:`~deeppavlov.models.seq2seq_go_bot.network.Seq2SeqGoalOrientedBotNetwork` classes.

References
----------

[1] `A New Multi-Turn, Multi-Domain, Task-Oriented Dialogue Dataset - 2017 <https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/>`_

[2] `Mihail Eric, Lakshmi Krishnan, Francois Charette, and Christopher D. Manning, "Key-Value Retrieval Networks for Task-Oriented Dialogue – 2017 <https://arxiv.org/abs/1705.05414>`_

