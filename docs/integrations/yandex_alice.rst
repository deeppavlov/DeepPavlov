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

Then run:

::

    python -m deeppavlov riseapi --api-mode alice --https --key my.key --cert my.crt  <config_path> [-d] [-p <port>]


* ``-d``: download model specific data before starting the service.
* ``-p <port>``: sets the port to ``<port>``. Overrides default
  settings from ``deeppavlov/utils/settings/server_config.json``.

Now set up and test your dialog (https://dialogs.yandex.ru/developer/).
Detailed documentation of the platform could be found on 
https://tech.yandex.ru/dialogs/alice/doc/about-docpage/. Advanced API
configuration is described in :doc:`REST API </integrations/rest_api>` section.


Python
~~~~~~

To run a model specified by a DeepPavlov config ``<config_path>`` as an Alice
skill, firstly, you have to turn it to a :class:`~deeppavlov.deprecated.skill.skill.Skill`
and then make it an :class:`~deeppavlov.deprecated.agent.agent.Agent`.

.. code:: python

    from deeppavlov import build_model
    from deeppavlov.deprecated.skills.default_skill import DefaultStatelessSkill
    from deeppavlov.deprecated.agents.default_agent import DefaultAgent
    from deeppavlov.deprecated.agents.processors import HighestConfidenceSelector
    from deeppavlov.utils.alice import start_agent_server

    model = build_model("<config_path>", download=True)

    # Step 1: make it a Skill 
    skill = DefaultStatelessSkill(model)
    # Step 2: make it an Agent
    agent = DefaultAgent(skills=[skill])
    # Step 3: run server
    start_agent_server(agent, host='0.0.0.0', port=7051, endpoint='/agent', ssl_key='my.key', ssl_cert='my.crt')

If your model is already a subclass of :class:`~deeppavlov.deprecated.skill.skill.Skill`
or a subclass of :class:`~deeppavlov.deprecated.agent.agent.Agent` (see
:doc:`skills </apiref/skills>` and :doc:`agents </apiref/agents>`) you can skip
corresponding steps.

