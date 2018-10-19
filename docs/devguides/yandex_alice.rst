Yandex Alice integration
========================


Pipelines
~~~~~~~~~


Any DeepPavlov pipeline can be launched as a skill for Yandex.Alice.

Configure host, port, model endpoint, GET request arguments in ``utils/server_config.json`` or see default values there.

Use your own certificate for HTTPS if you have; otherwise, generate self-signed one like that:

::

    openssl req -new -newkey rsa:4096 -days 365 -nodes -x509 -subj "/CN=MY_DOMAIN_OR_IP" -keyout my.key -out my.crt

Then run

::

    python -m deeppavlov riseapi --api-mode alice --https --key my.key --cert my.crt  <config_path> [-d]


Optional ``-d`` key is for dependencies download before service start.

Now set up and test your dialog (https://dialogs.yandex.ru/developer/). Detailed documentation of the platform could be
found on https://tech.yandex.ru/dialogs/alice/doc/about-docpage/, while other library options described in
:doc:`REST API </devguides/rest_api>` section.


Agents
~~~~~~

You can also run :doc:`agents </apiref/agents>` as Alice skills:

.. code:: python

    from deeppavlov.agents.default_agent.default_agent import DefaultAgent
    from deeppavlov.agents.processors.highest_confidence_selector import HighestConfidenceSelector
    from deeppavlov.skills.pattern_matching_skill import PatternMatchingSkill
    from utils.alice import start_agent_server

    skill_hello = PatternMatchingSkill(['Привет, мир!'], patterns=['привет', 'здравствуй', 'добрый день'])
    skill_bye = PatternMatchingSkill(['Пока, мир', 'Ещё увидимся'], patterns=['пока', 'чао', 'увидимся', 'до свидания'])
    skill_fallback = PatternMatchingSkill(['Извини, я не понимаю', 'Я умею здороваться )'])

    agent = DefaultAgent([skill_hello, skill_bye, skill_fallback], skills_processor=HighestConfidenceSelector())

    start_agent_server(agent, host='0.0.0.0', port=7051, endpoint='/agent', ssl_key='my.key', ssl_cert='my.crt')
