RASA Skill
======================

An :doc:`RASA wrapper implementation</apiref/skills/rasa_skill>` that reads a folder with RASA models
(provided by `path_to_models` argument), initilizes RASA Agent with this configuration and responds for incoming
utterances accroding to responses predicted by RASA. Each response has confidence value estimated as product of
scores of executed actions by RASA system in the current prediction step (each prediction step in RASA usually consists of
multiple actions). If RASA responds with multiple `BotUttered` actions, then such phrases are merged into one utterance
divided by '\n'.

Quick Start
-----------
To setup RASA Skill you need to have a working RASA project at some path, then  you can specify the path to RASA's
models (usually it is a folder with name `models` inside the project path) at initialization of RASA Skill class
by providing `path_to_models` attribute.

Dummy RASA project
------------------
DeepPavlov library has a template config for RASASkill here: :config:`configs/skills/rasa_skill.json <skills/rasa_skill.json>`.
This config uses a RASA project hosted at
`DeepPavlov fileshare <http://files.deeppavlov.ai/rasa_skill/rasa_tutorial_project.tar.gz>`_ (it is packed). This
project is used for tests (see test_rasa_skill.py) and in essense it is a working RASA project created with
`rasa init` and `rasa train` commands with minimal additions. The RASA bot can greet, answer about what he can do and
detect user's mood sentiment.

The template DeepPavlov config specifies the only one component (RASASkill) in a pipeline (see `chainer.pipe`
specification). The configuration also specifies: `metadata.requirements` which is the file with RASA dependency and
`metadata.download` configuration specifies to download and unpack the gzipped template project into subdir
`{DOWNLOADS_PATH}` (`{DOWNLOADS_PATH}` is declared in `metadata.variables` configuration).

If you write configuration for a RASA project hosted on your machine, you don't need to specify `metadata.download`
and just need to correctly set `path_to_models` of the `rasa_skill` component.
`path_to_models` must be path to your RASA's `models` directory.

See `RASA's documentation <https://rasa.com/docs/rasa/1.0.6/user-guide/rasa-tutorial/>`_ for explnation on how
to create project.

Usage without DeepPavlov configuration files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from deeppavlov.agents.default_agent.default_agent import DefaultAgent
    from deeppavlov.agents.processors.highest_confidence_selector import HighestConfidenceSelector
    from deeppavlov.skills.rasa_skill.rasa_skill import RASASkill

    rasa_skill_config = {
        'path_to_models': <put the path to your RASA's models>,
    }

    rasa_skill = RASASkill(**rasa_skill_config)
    agent = DefaultAgent([rasa_skill], skills_selector=HighestConfidenceSelector())
    responses = agent(["Hello"])
    print(responses)
