# -*- coding: utf-8 -*-
import unittest
import os

from deeppavlov.agents.default_agent.default_agent import DefaultAgent
from deeppavlov.agents.processors.highest_confidence_selector import HighestConfidenceSelector
from deeppavlov.skills.aiml_skill.aiml_skill import AIMLSkill


class TestAIMLSkill(unittest.TestCase):
    """
    Set up basic skill which reacts with standard reaction
    """

    def setUp(self):

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_aiml_scripts = cur_dir + "/aiml_scripts"

        aiml_skill_config = {
            'null_response': "I don't know what to answer you",
            'default_confidence': 0.66
        }
        if path_to_aiml_scripts:
            aiml_skill_config['path_to_aiml_scripts'] = path_to_aiml_scripts

        aiml_skill = AIMLSkill(**aiml_skill_config)
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
        # batch of ids for one new user
        dialog_ids_batch = [1, ]

        history_of_responses = []
        for each_utt in user_messages_sequence:
            print(f"User says: {each_utt}")
            responses_batch = self.agent([each_utt], dialog_ids_batch)
            print(f" Bot says: {responses_batch[0]}")
            history_of_responses.append(responses_batch)

        # check the first greeting message in 0th batch
        self.assertIn("Well, hello!", history_of_responses[0][0])
        # check fifth message in 0th batch
        self.assertIn("Yes movies", history_of_responses[4][0])



if __name__ == "__main__":
    unittest.main()
