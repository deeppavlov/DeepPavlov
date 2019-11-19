Yandex Alice integration
========================

Any model specified by a DeepPavlov config can be launched as a skill for
Yandex.Alice. You can do it using command line interface or using python.

Command line interface
~~~~~~~~~~~~~~~~~~~~~~

To interact with Alice you will require your own HTTPS certificate. To generate
a new one -- run:

::

    openssl req -new -newkey rsa:4096 -days 365 -nodes -x509 -subj "/CN=MY_DOMAIN_OR_IP" -keyout my.key -out my.crt

To run a model specified by the ``<config_path>`` config file as an Alice
skill, run:

::

    python -m deeppavlov alice <config_path> --https --key my.key --cert my.crt  [-d] [-p <port>]

* ``-d``: download model specific data before starting the service.

The command will print the used host and port. Default web service properties
(host, port, model endpoint, GET request arguments, paths to ssl cert and key,
https mode) can be modified via changing
``deeppavlov/utils/settings/server_config.json`` file. ``--https``, ``--key``,
``--cert``, ``-p`` arguments override default values from ``server_config.json``.
Advanced API configuration is described in
:doc:`REST API </integrations/rest_api>` section.

Now set up and test your dialog (https://dialogs.yandex.ru/developer/).
Detailed documentation of the platform could be found on 
https://tech.yandex.ru/dialogs/alice/doc/about-docpage/. Advanced API
configuration is described in :doc:`REST API </integrations/rest_api>` section.


Python
~~~~~~

To run a model specified by a DeepPavlov config ``<config_path>`` as an Alice
skill using python, you have to run following code:

.. code:: python

    from deeppavlov.utils.alice import start_alice_server

    start_alice_server(<config_path>,
                       host=<host>,
                       port=<port>,
                       endpoint=<endpoint>,
                       https=True,
                       ssl_key='my.key',
                       ssl_cert='my.crt')

All arguments except ``<model_config_path>`` are optional. Optional arguments override
corresponding values from ``deeppavlov/utils/settings/server_config.json``.
