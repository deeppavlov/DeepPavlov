import argparse

from deeppavlov.deep import find_config
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.agent.processor import Processor
from deeppavlov.core.agent.agent import Agent
from deeppavlov.core.agent.rich_content import RichMessage
from deeppavlov.agents.rich_content.default_rich_content import PlainText, ButtonsFrame, Button
from utils.ms_bot_framework_utils.server import run_ms_bot_framework_server


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--ms-id", help="microsoft bot framework app id", type=str)
parser.add_argument("-s", "--ms-secret", help="microsoft bot framework app secret", type=str)


class TestRichContentWrapper(Processor):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, utterances: list, batch_history: list, *responses: list) -> list:
        responses, confidences = zip(*[zip(*r) for r in responses])
        indexes = [c.index(max(c)) for c in zip(*confidences)]
        result = []

        for i, *responses in zip(indexes, *responses):
            rich_message = RichMessage()

            plain_text = PlainText(str(responses[i]))
            rich_message.add_control(plain_text)

            button_a = Button('Button A', 'button_a_callback')
            button_b = Button('Button B', 'button_b_callback')

            buttons_frame = ButtonsFrame()
            buttons_frame.add_button(button_a)
            buttons_frame.add_button(button_b)
            rich_message.add_control(buttons_frame)

            result.append(rich_message)

        return result


def make_agent():
    config_path = find_config('ecommerce_bot')
    skill = build_model_from_config(config_path, as_component=True)
    agent = Agent(skills=[skill], skills_processor=TestRichContentWrapper())
    return agent


def main():
    args = parser.parse_args()
    run_ms_bot_framework_server(agent_generator=make_agent,
                                app_id=args.ms_id,
                                app_secret=args.ms_secret,
                                stateful=True)


if __name__ == '__main__':
    main()
