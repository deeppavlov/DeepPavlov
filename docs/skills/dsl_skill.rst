DSL Skill
======================

An :doc:`DSL implementation</apiref/skills/dsl_skill>`. DSL helps to easily create user-defined
 skills for dialog systems.

For the case when DSL skill matched utterance and found response it outputs response with confidence
value.

For the case when no match occurred DSL skill returns the argument `on_invalid_command` ("Простите, я вас не понял" by delault)
 as utterance and sets confidence to `null_confidence` attribute (0 by default).


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
        def greeting(utterance, history, state):
            response = "Hello, my friend!"
            confidence = 1.0
            state = None
            return response, confidence, state


    skill_config = read_json(configs.dsl_skill.dsl_skill)

    skill = build_model(skill_config, download=True)
    utterances_batch = ["Hello", "How are you?"]
    responses = skill(utterances_batch)
    print(responses)

