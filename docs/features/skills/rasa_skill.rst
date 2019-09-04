Rasa Skill
======================

A :class:`Rasa wrapper implementation<deeppavlov.skills.rasa_skill.rasa_skill.RASASkill>` that reads a folder with Rasa models
(provided by ``path_to_models`` argument), initializes Rasa Agent with this configuration and responds for incoming
utterances according to responses predicted by Rasa. Each response has confidence value estimated as product of
scores of executed actions by Rasa system in the current prediction step (each prediction step in Rasa usually consists of
multiple actions). If Rasa responds with multiple ``BotUttered`` actions, then such phrases are merged into one utterance
divided by ``'\n'``.

Quick Start
-----------
To setup a Rasa Skill you need to have a working Rasa project at some path, then  you can specify the path to Rasa's
models (usually it is a folder with name ``models`` inside the project path) at initialization of Rasa Skill class
by providing ``path_to_models`` attribute.

Dummy Rasa project
------------------
DeepPavlov library has :config:`a template config for RASASkill<skills/rasa_skill.json>`.
This project is in essence a working Rasa project created with ``rasa init`` and ``rasa train`` commands
with minimal additions. The Rasa bot can greet, answer about what he can do and detect user's mood sentiment.

The template DeepPavlov config specifies only one component (RASASkill) in :doc:`a pipeline</intro/configuration>`.
The configuration also specifies: ``metadata.requirements`` which is the file with Rasa dependency and
``metadata.download`` configuration specifies to download and unpack the gzipped template project into subdir
``{DOWNLOADS_PATH}``.

If you create a configuration for a Rasa project hosted on your machine, you don't need to specify ``metadata.download``
and just need to correctly set ``path_to_models`` of the ``rasa_skill`` component.
``path_to_models`` needs to be a path to your Rasa's ``models`` directory.

See `Rasa's documentation <https://rasa.com/docs/rasa/1.2.5/user-guide/rasa-tutorial/>`_ for explanation on how
to create project.

Usage without DeepPavlov configuration files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from deeppavlov.agents.default_agent.default_agent import DefaultAgent
    from deeppavlov.agents.processors.highest_confidence_selector import HighestConfidenceSelector
    from deeppavlov.skills.rasa_skill.rasa_skill import RASASkill

    rasa_skill_config = {
        'path_to_models': <put the path to your Rasa models>,
    }

    rasa_skill = RASASkill(**rasa_skill_config)
    agent = DefaultAgent([rasa_skill], skills_selector=HighestConfidenceSelector())
    responses = agent(["Hello"])
    print(responses)
