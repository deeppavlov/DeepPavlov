
Telegram integration
========================

Any model specified by a DeepPavlov config can be launched as a Telegram bot.
You can do it using command line interface or using python.

Command line interface
~~~~~~~~~~~~~~~~~~~~~~

To run a model specified by the ``<config_path>`` config file as a Telegram bot
with a ``<telegram_token>``:

.. code:: bash

    python -m deeppavlov telegram <config_path> [-t <telegram_token>] [-d]


* ``-t <telegram_token>``: specifies telegram token as ``<telegram_token>``. Overrides
  default value from ``deeppavlov/utils/settings/server_config.json``.
* ``-d``: downloads model specific data before starting the service.

The command will print info message ``Bot initiated`` when starts bot.

``/start`` and ``/help`` Telegram bot messages can be modified via changing
``telegram.start_message`` and ``telegram.help_message``
in `deeppavlov/utils/settings/server_config.json`.

Python
~~~~~~

To run a model specified by a DeepPavlov config ``<config_path>`` as
Telegram bot, you have to run following code:

.. code:: python

    from deeppavlov.utils.telegram import interact_model_by_telegram

    interact_model_by_telegram(model_config=<config_path>, token=<telegram_token>)
