AIML Skill
======================

An :doc:`AIML scripts wrapper implementation</apiref/skills/aiml_skill>` that reads a folder with AIML scripts
(provided by `path_to_aiml_scripts` argument), loads it into AIML's Kernel and responds for incoming utterances
accroding to patterns learned by AIML Kernel.

For the case when AIML kernel matched utterance and found response the AIML Wrapper outputs response with confidence
value (as specified by`positive_confidence` argument).

For the case when no match occured the wrapper returns the argument `null_response` as utterance and sets confidence to
`null_confidence` attribute.


Quick Start
-----------
To setup AIML Skill you need load your AIML scripts to some folder and specify path to it with initilization
parameter `path_to_aiml_scripts`.

You can download bunch of free and ready for use AIML scripts from pandorabots repo:
https://github.com/pandorabots/Free-AIML

DeepPavlov library has default config for AIMLSkill here: :config:`configs/skills/aiml_skill.json <skills/aiml_skill.json>`

Usage
^^^^^^^^

.. code:: python

    from deeppavlov.skills.aiml_skill import AIMLSkill

    aiml_skill_config = {
        'positive_confidence': 0.66,
        'path_to_aiml_scripts': <put the path to your AIML scripts here>,
        'null_response': "I don't know what to answer you",
        'null_confidence': 0.33
    }

    aiml_skill = AIMLSkill(**aiml_skill_config)

    states_batch = None
    for utterance in ["Hello", "Hello to the same user_id"]:
        responses_batch, confidences_batch, states_batch = aiml_skill([utterance], states_batch)
        print(responses_batch[0])
