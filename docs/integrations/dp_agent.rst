DeepPavlov Agent RabbitMQ integration
=====================================

Any model specified by a DeepPavlov config can be launched as a service for
`DeepPavlov Agent <https://deeppavlov-agent.readthedocs.io/en/latest/>`_
communicating with agent through RabbitMQ message broker. You can launch it
using command line interface or using python.

Command line interface
~~~~~~~~~~~~~~~~~~~~~~

To run a model specified by the ``<config_path>`` config file as a DeepPavlov Agent service, run:

.. code:: bash

    python -m deeppavlov agent-rabbit <config_path> [-d] \
                                                    [-sn <service_name>] \
                                                    [-an <agent_name>] \
                                                    [-ans <agent_namespace>] \
                                                    [-b <batch_size>] \
                                                    [-ul <utterace_lifetime] \
                                                    [-rh <rabbit_host>] \
                                                    [-rp <rabbit_port>] \
                                                    [-rl <rabbit_login>] \
                                                    [-rpwd <rabbit_password>] \
                                                    [-rvh <rabbit_virtualhost>]

* ``-d``: download model specific data before starting the service.
* ``-sn <service_name>``: service name set in the connector section of the DeepPavlov Agent config file.
* ``-an <agent_namespace>``: namespace the service works in. Messages only from agents from this namespace is processed.
* ``-b <batch_size>``: inference batch size.
* ``-ul <utterace_lifetime``: RabbitMQ message expiration time in seconds.
* ``-rh <rabbit_host>``: RabbitMQ server host.
* ``-rp <rabbit_port>``: RabbitMQ server port.
* ``-rl <rabbit_login>``: RabbitMQ server login.
* ``-rpwd <rabbit_password>``: RabbitMQ server password.
* ``-rvh <rabbit_virtualhost>``: RabbitMQ server virtualhost.

Default values of optional arguments can be modified via changing ``agent-rabbit`` section of the file
``deeppavlov/utils/settings/server_config.json``.

Python interface
~~~~~~~~~~~~~~~~

To run a model specified by the ``<config_path>`` config file as a DeepPavlov Agent service using python,
run the following code:

.. code:: python

    from deeppavlov.utils.agent import start_rabbit_service

    start_rabbit_service(model_config=<config_path>,
                         service_name=<service_name>,
                         agent_namespace=<agent_namespace>,
                         batch_size=<batch_size>,
                         utterance_lifetime_sec=<utterance_lifetime>,
                         rabbit_host=<rabbit_host>,
                         rabbit_port=<rabbit_port>,
                         rabbit_login=<rabbit_login>,
                         rabbit_password=<rabbit_password>,
                         rabbit_virtualhost=<rabbit_virtualhost>)

All arguments except ``<config_path>`` are optional. Default values of optional arguments can be modified via changing
``agent-rabbit`` section of the file ``deeppavlov/utils/settings/server_config.json``.