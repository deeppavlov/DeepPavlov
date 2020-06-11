from logging import getLogger

from deeppavlov import configs, build_model
from deeppavlov.core.common.file import read_json
from deeppavlov.skills.dsl_skill import DSLMeta
from deeppavlov.utils.pip_wrapper.pip_wrapper import install_from_config

log = getLogger(__name__)


class DSLSkill(metaclass=DSLMeta):
    @DSLMeta.handler(commands=["hello", "hi", "sup", "greetings"])
    def greeting(context):
        response = "Hello, my friend!"
        confidence = 1.0
        return response, confidence


class StateSkill(metaclass=DSLMeta):
    @DSLMeta.handler(commands=["hello", "hi", "sup", "greetings"])
    def greeting(context):
        response = "Hello, my friend!"
        confidence = 1.0
        context.current_state = "state1"
        return response, confidence

    @DSLMeta.handler(commands=["bye"],
                     state="state1")
    def bye(context):
        response = "bb!"
        confidence = 1.0
        return response, confidence


class ContextConditionSkill(metaclass=DSLMeta):
    @DSLMeta.handler(commands=["hello", "hi", "sup", "greetings"],
                     context_condition=lambda context: context.user_id != 1)
    def greeting(context):
        response = "Hello, my friend!"
        confidence = 1.0
        return response, confidence


class TestDSLSkill:
    def setup(self):
        self.skill_config = read_json(configs.skills.dsl_skill)
        install_from_config(self.skill_config)

    def test_simple_skill(self):
        user_messages_sequence = [
            "Hello",
            "Hi",
            "Tell me a joke",
            "Sup",
            "Ok, goodbye"
        ]

        skill = build_model(self.skill_config, download=True)
        history_of_responses = []
        for user_id, each_utt in enumerate(user_messages_sequence):
            log.info(f"User says: {each_utt}")
            responses_batch = skill([each_utt], [user_id])
            log.info(f"Bot says: {responses_batch[0]}")
            history_of_responses.append(responses_batch)

        # check the first greeting message in 0th batch
        assert "Hello, my friend!" in history_of_responses[0][0]
        # check the second greeting message in 0th batch
        assert "Hello, my friend!" in history_of_responses[1][0]
        # check `on_invalid_command`
        assert "Sorry, I do not understand you" in history_of_responses[2][0]

    def test_switch_state(self):
        user_messages_sequence = [
            "Hello",
            "bye",
            "bye"
        ]

        self.skill_config["chainer"]["pipe"][1]["class_name"] = "StateSkill"
        skill = build_model(self.skill_config, download=True)

        history_of_responses = []
        for user_id, each_utt in enumerate(user_messages_sequence):
            log.info(f"User says: {each_utt}")
            responses_batch = skill([each_utt], [user_id % 2])
            log.info(f"Bot says: {responses_batch[0]}")
            history_of_responses.append(responses_batch)
        assert "Hello, my friend!" in history_of_responses[0][0]
        assert "Sorry, I do not understand you" in history_of_responses[1][0]
        assert "bb!" in history_of_responses[2][0]

    def test_context_condition(self):
        user_messages_sequence = [
            "Hello",
            "Hi"
        ]

        self.skill_config["chainer"]["pipe"][1]["class_name"] = "ContextConditionSkill"
        skill = build_model(self.skill_config, download=True)

        history_of_responses = []
        for user_id, each_utt in enumerate(user_messages_sequence):
            log.info(f"User says: {each_utt}")
            responses_batch = skill([each_utt], [user_id])
            log.info(f"Bot says: {responses_batch[0]}")
            history_of_responses.append(responses_batch)
        assert "Hello, my friend!" in history_of_responses[0][0]
        assert "Sorry, I do not understand you" in history_of_responses[1][0]
