# -*- coding: utf-8 -*-
import unittest
# ################# Universal Import ###################################################
# import sys
# import os
#
#
# SELF_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(SELF_DIR)
# PREROOT_DIR = os.path.dirname(ROOT_DIR)
# print(ROOT_DIR)
# sys.path.append(ROOT_DIR)
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ruler_bot.settings")
# # #####################################################
# import django
#
# django.setup()
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
        # import ipdb; ipdb.set_trace()

        for each_utt in user_messages_sequence:
            print(f"User says: {each_utt}")
            responses_batch = self.agent([each_utt], dialog_ids_batch)
            print(f" Bot says: {responses_batch[0]}")
        import ipdb; ipdb.set_trace()
        print(agent)
        # userdialog = self.agent.conjugate_autouser_with_agent(user_messages_sequence, self.user_id)
        # self.assertIn("Well, hello!", userdialog[1].text)


if __name__ == "__main__":
    unittest.main()
