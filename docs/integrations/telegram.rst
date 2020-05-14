
Telegram integration
========================

Any model specified by a DeepPavlov config can be launched as a Telegram bot.
You can do it using command line interface or using python.

Command line interface
~~~~~~~~~~~~~~~~~~~~~~

To run a model specified by the ``<config_path>`` config file as a Telegram bot
with a ``<telegram_token>``:

.. code:: bash

    python -m deeppavlov telegram <config_path> [-t <telegram_token>] [-ps <proxy_scheme>] [-pa <proxy_auth>] [-d]


* ``-t <telegram_token>``: specifies telegram token as ``<telegram_token>``. Overrides
  default value of ``telegram.token`` from ``deeppavlov/utils/settings/server_config.json``.
* ``-ps <proxy_scheme>``: specifies telegram proxy scheme as ``<proxy_scheme>``. Overrides default value of
  ``telegram.proxy_scheme`` from ``deeppavlov/utils/settings/server_config.json``.
* ``-pa <proxy_auth>``: specifies telegram proxy authentication as ``<proxy_auth>``. Overrides default value of
  ``telegram.proxy_auth`` from ``deeppavlov/utils/settings/server_config.json``.
* ``-d``: downloads model specific data before starting the service.

The command will print info message ``Bot initiated`` when bot starts. ``<proxy_scheme>`` and ``<proxy_auth>`` parameter
values are used as key-value pair for `proxies <https://requests.readthedocs.io/en/master/user/advanced/#proxies>`_
argument in ``requests`` (for example, ``{'https': 'socks5h://userproxy:password@proxy_address:port'}``).

``/start`` and ``/help`` Telegram bot messages can be modified via changing
``telegram.start_message`` and ``telegram.help_message``
in ``deeppavlov/utils/settings/server_config.json``.

Python
~~~~~~

To run a model specified by a DeepPavlov config ``<config_path>`` as
Telegram bot, you have to run following code:

.. code:: python

    from deeppavlov.utils.telegram import interact_model_by_telegram

    interact_model_by_telegram(model_config=<config_path>,
                               token=<telegram_token>,
                               proxy_scheme=<proxy_scheme>,
                               proxy_auth=<proxy_auth>)
