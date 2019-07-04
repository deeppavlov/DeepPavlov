from logging import getLogger

from deeppavlov import configs, build_model
from deeppavlov.core.common.file import read_json
from deeppavlov.skills.dsl_skill.dsl_skill import DSLMeta
from deeppavlov.utils.pip_wrapper.pip_wrapper import install_from_config

log = getLogger(__name__)


class SimpleSkill(metaclass=DSLMeta):
    @DSLMeta.handler(commands=["hello", "hi", "sup", "greetings"])
    def greeting(utterance, history, state):
        response = "Hello, my friend!"
        confidence = 1.0
        state = None
        return response, confidence, state


class TestDSLSkill:
    def setup(self):
        self.skill_config = read_json(configs.dsl_skill.dsl_skill)
        install_from_config(self.skill_config)

    def test_simple_skill(self):
        user_messages_sequence = [
            "Hello",
            "Hi",
            "Tell me a joke",
            "Sup",
            "Ok, goodbye"
        ]
        on_invalid_command = "Sorry, I do not understand you"

        self.skill_config["chainer"]["pipe"][1]["class_name"] = "SimpleSkill"
        self.skill_config["chainer"]["pipe"][1]["on_invalid_command"] = on_invalid_command
        skill = build_model(self.skill_config, download=True)
        history_of_responses = []
        for each_utt in user_messages_sequence:
            log.info(f"User says: {each_utt}")
            responses_batch = skill([each_utt])
            log.info(f" Bot says: {responses_batch[0]}")
            history_of_responses.append(responses_batch)

        # check the first greeting message in 0th batch
        assert "Hello, my friend!" in history_of_responses[0][0]
        # check the second greeting message in 0th batch
        assert "Hello, my friend!" in history_of_responses[1][0]
        # check `on_invalid_command`
        assert on_invalid_command in history_of_responses[2][0]
