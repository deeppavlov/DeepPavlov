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
  is ``UNIX``. Overrides default settings from
  ``deeppavlov/utils/settings/socket_config.json``.
* ``-p <port>``: sets the port to ``<port>`` if socket address family is
  ``AF_INET``. Overrides default settings from
  ``deeppavlov/utils/settings/socket_config.json``.
* ``--socket-file <unix_socket_file>``: sets the file for socket binding to
  ``<unix_socket_file>`` if socket address family is ``AF_UNIX``. Overrides
  default settings from ``deeppavlov/utils/settings/socket_config.json``.

The command will print the binding address: host and port for ``AF_INET``
socket family and path to the UNIX socket file for ``AF_UNIX`` socket family.
Default service properties (socket address family, host, port, path to the UNIX
socket file, socket buffer size, binding message) can be modified via changing
``deeppavlov/utils/settings/socket_config.json`` file.

Advanced configuration
~~~~~~~~~~~~~~~~~~~~~~

By modifying ``deeppavlov/utils/settings/socket_config.json`` you can change
socket address family, host, port, path to the UNIX socket file and other
properties of the API service.

Properties from ``common_defaults`` section are used by default unless they are
overridden by model-specific properties, provided in ``model_defaults`` section
of the ``socket_config.json``. Model-specific properties are bound to the model
by ``server_utils`` label in ``metadata/labels`` section of the model config.
Value of ``server_utils`` label from model config should match with properties
key from ``model_defaults`` section of ``socket_config.json``.

For example, ``metadata/labels/server_utils`` tag from
``deeppavlov/configs/squad/squad.json`` references to the *SquadModel* section
of ``socket_config.json``. Therefore, ``model_args_names`` (see details below)
parameter in ``common_defaults`` will be overridden with the same parameter
from ``model_defaults/SquadModel``.

Model argument names are provided as list in ``model_args_names`` parameter,
where arguments order corresponds to model API. When inferencing model via
socket API, serialized JSON payload keys should match model arguments names from
``model_args_names``. Default argument name for one argument models is
*"context"*. Here are server request examples for some of the library models:

+-----------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------+
| Model                                   | POST request JSON payload example                                                                                                               |
+=========================================+=================================================================================================================================================+
| **One argument models**                                                                                                                                                                   |
+-----------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------+
| NER model                               | {"context":["Elon Musk launched his cherry Tesla roadster to the Mars orbit"]}                                                                  |
+-----------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------+
| Intent classification model             | {"context":["I would like to go to a restaurant with Asian cuisine this evening"]}                                                              |
+-----------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------+
| Automatic spelling correction model     | {"context":["errror"]}                                                                                                                          |
+-----------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------+
| Ranking model                           | {"context":["What is the average cost of life insurance services?"]}                                                                            |
+-----------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------+
| Goal-oriented bot                       | {"context":["Hello, can you help me to find and book a restaurant this evening?"]}                                                              |
+-----------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------+
| **Multiple arguments models**                                                                                                                                                             |
+-----------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------+
| Question Answering model                | | {"context":["After 1765, growing philosophical and political differences strained the relationship between Great Britain and its colonies."], |
|                                         | |  "question":["What strained the relationship between Great Britain and its colonies?"]}                                                       |
+-----------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------+

Socket client example (Python)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Socket client for :doc:`SQuAD </features/models/squad>` model with a batch of
two elements:

.. code-block:: python

    # squad-client.py

    import json
    import socket

    socket_payload = {
        "context": [
            "All work and no play makes Jack a dull boy",
            "I used to be an adventurer like you, then I took an arrow in the knee."
        ],
        "question": [
            "What makes Jack a dull boy?",
            "Who I used to be?"
        ]
    }
    dumped_socket_payload = json.dumps(socket_payload)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('0.0.0.0', 5000))
        s.sendall(dumped_socket_payload.encode('utf-8'))
        serialized_payload = s.recv(1024)
        json_payload = json.loads(serialized_payload)

    print(json_payload)

To start socket server with ``squad_bert`` model run:

.. code:: bash

    python -m deeppavlov risesocket -d squad_bert --socket-type TCP -p 5000


To start socket client on another terminal run:

.. code:: bash

    python squad-client.py
