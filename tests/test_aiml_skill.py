import pytest
from pathlib import Path
from deeppavlov.agents.default_agent.default_agent import DefaultAgent
from deeppavlov.agents.processors.highest_confidence_selector import HighestConfidenceSelector
from deeppavlov import configs, build_model


class TestAIMLSkill:
    def setup(self):
        aiml_skill = build_model(configs.aiml_skill.aiml_skill, download=True)
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
            print(f"User says: {each_utt}")
            responses_batch = self.agent([each_utt])
            print(f" Bot says: {responses_batch[0]}")
            history_of_responses.append(responses_batch)

        # check the first greeting message in 0th batch
        assert "Well, hello!" in history_of_responses[0][0]
        # check fifth message in 0th batch
        assert "Yes movies" in history_of_responses[4][0]
