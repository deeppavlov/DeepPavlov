
Telegram integration
========================

Any model specified by a DeepPavlov config can be launched as a Telegram bot.
You can do it using command line interface or using python.

Command line interface
~~~~~~~~~~~~~~~~~~~~~~

To run a model specified by the ``<config_path>`` config file as a Telegram bot
with a ``<telegram_token>``:

.. code:: bash

    python -m deeppavlov interactbot <config_path> -t <telegram_token> [-d]


* ``-t <telegram_token>``: specifies telegram token as ``<telegram_token>``.
* ``-d``: downloads model specific data before starting the service.

The command will print the used host and port. Default web service properties
(host, port, model endpoint, GET request arguments) can be modified via changing
``deeppavlov/utils/settings/server_config.json`` file. Advanced API
configuration is described in :doc:`REST API </integrations/rest_api>` section.

If you want to get custom ``/start`` and ``/help`` Telegram messages for the running model you should:

* Add section to ``models_info`` section of ``deeppavlov/utils/settings/connector_config.json`` with your custom
  Telegram messages

Python
~~~~~~

To run a model specified by a DeepPavlov config ``<config_path>`` as as
Telegram bot, you have to run following code:

.. code:: python

    from deeppavlov.utils.telegram import interact_model_by_telegram

    interact_model_by_telegram(model_config=<config_path>, token=<telegram_token>)
