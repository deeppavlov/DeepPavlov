Socket API
==========

Each DeepPavlov model can be made available as a socket server. The general
method is:

.. code:: bash

    python -m deeppavlov risesocket <config_path> [-d] [--socket-type <address_family>] [-p <port>] \
    [--socket-file <unix_socket_file>]


* ``-d``: downloads model specific data before starting the service.
* ``--socket-type <address_family>``: sets socket address family to ``AF_INET``
  if ``<address_family>`` is ``TCP`` or to ``AF_UNIX`` if ``<address_family>``
  is ``UNIX``. Overrides default value from
  ``deeppavlov/utils/settings/server_config.json``.
* ``-p <port>``: sets the port to ``<port>`` if socket address family is
  ``AF_INET``. Overrides default value from
  ``deeppavlov/utils/settings/server_config.json``.
* ``--socket-file <unix_socket_file>``: sets the file for socket binding to
  ``<unix_socket_file>`` if socket address family is ``AF_UNIX``. Overrides
  default value from ``deeppavlov/utils/settings/server_config.json``.

The command will print the binding address: host and port for ``AF_INET``
socket family and path to the UNIX socket file for ``AF_UNIX`` socket family.
Default service properties (socket address family, host, port, path to the UNIX
socket file, socket buffer size, binding message) can be modified via changing
``deeppavlov/utils/settings/server_config.json`` file.

Advanced configuration
~~~~~~~~~~~~~~~~~~~~~~

By modifying ``deeppavlov/utils/settings/server_config.json`` you can change
socket address family, host, port, path to the UNIX socket file and other
properties of the API service.

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
When inferencing model via socket API, serialized JSON payload keys should match
model arguments names from ``chainer/in`` section.
If ``model_args_names`` parameter of ``server_config.json`` is list, its values
are used as model argument names instead of the list from model config's
``chainer/in`` section.

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

Socket client example (Python)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Socket client for :doc:`SQuAD </features/models/squad>` model with a batch of
two elements:

.. code-block:: python

    # squad-client.py

    import json
    import socket
    from struct import unpack

    from deeppavlov.utils.socket import encode

    socket_payload = {
        "context_raw": [
            "All work and no play makes Jack a dull boy",
            "I used to be an adventurer like you, then I took an arrow in the knee"
        ],
        "question_raw": [
            "What makes Jack a dull boy?",
            "Who I used to be?"
        ]
    }
    serialized_socket_payload = encode(socket_payload)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('0.0.0.0', 5000))
        s.sendall(serialized_socket_payload)
        header = s.recv(4)
        body_len = unpack('<I', header)[0]
        serialized_response = s.recv(body_len)
        json_payload = json.loads(serialized_response)

    print(json_payload)

To start socket server with ``squad_bert`` model run:

.. code:: bash

    python -m deeppavlov risesocket -d squad_bert --socket-type TCP -p 5000


To start socket client on another terminal run:

.. code:: bash

    python squad-client.py
