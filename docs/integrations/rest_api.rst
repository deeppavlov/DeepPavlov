REST API
========

Each DeepPavlov model can be easily made available for
inference as a REST web service. The general method is:

.. code:: bash

    python -m deeppavlov riseapi <config_path> [-d] [-p <port>] [--https] [--key <SSL key file path>] \
    [--cert <SSL certificate file path>]


* ``-d``: downloads model specific data before starting the service.
* ``-p <port>``: sets the port to ``<port>``. Overrides default
  value from ``deeppavlov/utils/settings/server_config.json``.
* ``--https``: use https instead of http. Overrides default
  value from ``deeppavlov/utils/settings/server_config.json``.
* ``--key <SSL key file path>``: path to SSL key file. Overrides default
  value from ``deeppavlov/utils/settings/server_config.json``.
* ``--cert <SSL certificate file path>``: path to SSL certificate file. Overrides default
  value from ``deeppavlov/utils/settings/server_config.json``.

The command will print the used host and port. Default web service properties
(host, port, POST request arguments) can be modified via changing
``deeppavlov/utils/settings/server_config.json`` file.

API routes
----------

/model
""""""
Send POST request to ``<host>:<port>/model`` to infer model. See details at
:ref:`rest_api_docs`.

/probe
""""""
Send POST request to ``<host>:<port>/probe`` to check if API is working. The
server will send a response ``["Test passed"]`` if it is working. Requests to
``/probe`` are not logged.

/api
""""
To get model argument names send GET request to ``<host>:<port>/api``. Server
will return list with argument names.

.. _rest_api_docs:

/docs
"""""
To interact with the REST API via graphical interface open
``<host>:<port>/docs`` in a browser (Swagger UI).

/metrics
""""""""
Endpoint to monitor a running service using Prometheus. Metrics:

* ``http_requests_count``: Counter, tracks number of processed requests. Labels: ``endpoint``, ``status_code``.
* ``http_requests_latency_seconds``: Histogram, tracks responses latency (only with 200 status code). Labels:
  ``endpoint``.
* ``http_requests_in_progress``: Gauge, tracks inprogress requests. Labels: ``endpoint``.

Advanced configuration
----------------------

By modifying ``deeppavlov/utils/settings/server_config.json`` you can change
host, port, POST request arguments and other properties of the API service.

Properties from ``common_defaults`` section are used by default unless
they are overridden by model-specific properties, provided in ``model_defaults``
section of the ``server_config.json``. Model-specific properties are bound
to the model by ``server_utils`` label in ``metadata`` section of the model
config. Value of ``server_utils`` label from model config should match with
properties key from ``model_defaults`` section of ``server_config.json``.

For example, adding ``metadata/server_utils`` key to ``go_bot/gobot_dstc2.json``
with value *GoalOrientedBot* will initiate the search of *GoalOrientedBot* tag
at ``model_defaults`` section of ``server_config.json``. Therefore, if this
section is present, all parameters with non empty (i.e. not ``""``,
not ``[]`` etc.) values stored by this tag will overwrite the parameter values
in ``common_defaults``.

If ``model_args_names`` parameter of ``server_config.json`` is empty string,
then model argument names are provided as list from ``chainer/in`` section of
the model config file, where arguments order corresponds to model API.
When inferencing model via REST api, JSON payload keys should match
model arguments names from ``chainer/in`` section.
If ``model_args_names`` parameter of ``server_config.json`` is list, its values
are used as model argument names instead of the list from model config's
``chainer/in`` section.
Here are POST request payload examples for some of the library models:

+-----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| Model                                   | POST request JSON payload example                                                                                                                   |
+=========================================+=====================================================================================================================================================+
| **One argument models**                                                                                                                                                                       |
+-----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| NER model                               | {"x":["Elon Musk launched his cherry Tesla roadster to the Mars orbit"]}                                                                            |
+-----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| Intent classification model             | {"x":["I would like to go to a restaurant with Asian cuisine this evening"]}                                                                        |
+-----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| Automatic spelling correction model     | {"x":["errror"]}                                                                                                                                    |
+-----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| Ranking model                           | {"x":["What is the average cost of life insurance services?"]}                                                                                      |
+-----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| Goal-oriented bot                       | {"x":["Hello, can you help me to find and book a restaurant this evening?"]}                                                                        |
+-----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| **Multiple arguments models**                                                                                                                                                                 |
+-----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| Question Answering model                | | {"context_raw":["After 1765, growing philosophical and political differences strained the relationship between Great Britain and its colonies."], |
|                                         | |  "question_raw":["What strained the relationship between Great Britain and its colonies?"]}                                                       |
+-----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+

