from pathlib import Path
from logging import getLogger

import pytest

from deeppavlov.agents.default_agent.default_agent import DefaultAgent
from deeppavlov.agents.processors.highest_confidence_selector import HighestConfidenceSelector
from deeppavlov import configs, build_model
from deeppavlov.utils.pip_wrapper.pip_wrapper import install_from_config

log = getLogger(__name__)


class TestAIMLSkill:
    def setup(self):
        config_ref = configs.skills.aiml_skill
        install_from_config(config_ref)
        aiml_skill = build_model(config_ref, download=True)
        self.agent = DefaultAgent([aiml_skill], skills_selector=HighestConfidenceSelector())

    def test_simple_reaction(self):
        user_messages_sequence = ["Hello",
                                  "What s up?",
                                  "Tell me a joke",
                                  "Learn my pants are Red",
                                  "LET DISCUSS MOVIES",
                                  "Comedy movies are nice to watch",
                                  "I LIKE WATCHING COMEDY!",
                                  "Ok, goodbye"
        ]

        history_of_responses = []
        for each_utt in user_messages_sequence:
            log.info(f"User says: {each_utt}")
            responses_batch = self.agent([each_utt])
            log.info(f" Bot says: {responses_batch[0]}")
            history_of_responses.append(responses_batch)

        # check the first greeting message in 0th batch
        assert "Well, hello!" in history_of_responses[0][0]
        # check fifth message in 0th batch
        assert "Yes movies" in history_of_responses[4][0]
