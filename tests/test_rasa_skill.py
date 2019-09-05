from logging import getLogger

from deeppavlov.agents.default_agent.default_agent import DefaultAgent
from deeppavlov.agents.processors.highest_confidence_selector import HighestConfidenceSelector
from deeppavlov import configs, build_model
from deeppavlov.utils.pip_wrapper.pip_wrapper import install_from_config

log = getLogger(__name__)


class TestRASASkill:
    def setup(self):
        # print(configs.aiml_skill)
        config_ref = configs.skills.rasa_skill
        install_from_config(config_ref)
        # rasa_skill = build_model(
        #     "/home/alx/Workspace/DeepPavlov/deeppavlov/configs/aiml_skill/rasa_skill.json",
        #     download=True)
        rasa_skill = build_model(
            config_ref,
            download=True)
        self.agent = DefaultAgent([rasa_skill], skills_selector=HighestConfidenceSelector())

    def test_simple_reaction(self):
        user_messages_sequence = ["Hello",
                                  "What can you do?",
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

        print("history_of_responses:")
        print(history_of_responses)
        # # check the first greeting message in 0th batch
        assert "Hey! How are you?" in history_of_responses[0][0]
        # # check second response message in 0th batch
        assert "I can chat with you. You can greet me" in history_of_responses[1][0]
