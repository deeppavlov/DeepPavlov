DSL Skill
======================

An :doc:`DSL implementation</apiref/skills/dsl_skill>`. DSL helps to easily create user-defined skills for dialog systems.

For the case when DSL skill matched utterance and found response it outputs response with confidence value.

For the case when no match occurred DSL skill returns the argument `on_invalid_command` ("Простите, я вас не понял" by delault) as utterance and sets confidence to `null_confidence` attribute (0 by default).

`on_invalid_command` and `null_confidence` can be changed in model config


Quick Start
-----------

DeepPavlov library has default config for DSLSkill here: :config:`configs/dsl_skill/dsl_skill.json <dsl_skill/dsl_skill.json>`

Usage
^^^^^^^^

.. code:: python

    from deeppavlov import configs, build_model
    from deeppavlov.core.common.file import read_json
    from deeppavlov.skills.dsl_skill.dsl_skill import DSLMeta


    class DSLSkill(metaclass=DSLMeta):
        @DSLMeta.handler(commands=["hello", "hi", "sup", "greetings"])
        def greeting(context):
            response = "Hello, my friend!"
            confidence = 1.0
            return response, confidence


    skill_config = read_json(configs.dsl_skill.dsl_skill)

    skill = build_model(skill_config, download=True)
    utterance = "Hello"
    user_id = 1
    response = skill([utterance], [user_id])
    print(response)
