# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from collections import defaultdict
from logging import getLogger
from typing import List, Dict, Any

from deeppavlov.core.commands.infer import build_model
from deeppavlov.deep import find_config
from deeppavlov.deprecated.agent import Agent, RichMessage
from deeppavlov.deprecated.agents.rich_content import PlainText, ButtonsFrame, Button
from deeppavlov.deprecated.skill import Skill
from deeppavlov.utils.ms_bot_framework import start_ms_bf_server

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--ms-id", help="microsoft bot framework app id", type=str)
parser.add_argument("-s", "--ms-secret", help="microsoft bot framework app secret", type=str)

log = getLogger(__name__)


class EcommerceAgent(Agent):
    """DeepPavlov Ecommerce agent.

    Args:
        skill: List of initiated agent skills instances.

    Attributes:
        skill: List of initiated agent skills instances.
        history: Histories for each each dialog with agent indexed
            by dialog ID. Each history is represented by list of incoming
            and outcoming replicas of the dialog.
        states: States for each each dialog with agent indexed by dialog ID.
    """

    def __init__(self, skills: List[Skill], *args, **kwargs) -> None:
        super(EcommerceAgent, self).__init__(skills=skills)
        self.states: dict = defaultdict(lambda: [{"start": 0, "stop": 5} for _ in self.skills])

    def _call(self, utterances_batch: List[str], utterances_ids: List[int] = None) -> List[RichMessage]:
        """Processes batch of utterances and returns corresponding responses batch.

        Args:
            utterances_batch: Batch of incoming utterances.
            utterances_ids: Batch of dialog IDs corresponding to incoming utterances.

        Returns:
            responses: A batch of responses corresponding to the
                utterance batch received by agent.
        """

        rich_message = RichMessage()
        for utt_id, utt in enumerate(utterances_batch):

            if utterances_ids:
                id_ = utterances_ids[utt_id]

            log.debug(f'Utterance: {utt}')

            if utt == "/start":
                welcome = "I am a new e-commerce bot. I will help you to find products that you are looking for. Please type your request in plain text."
                rich_message.add_control(PlainText(welcome))
                continue

            if utt[0] == "@":
                command, *parts = utt.split(":")
                log.debug(f'Actions: {parts}')

                if command == "@details":
                    batch_index = int(parts[0])  # batch index in history list
                    item_index = int(parts[1])  # index in batch
                    rich_message.add_control(PlainText(show_details(
                        self.history[id_][batch_index][item_index])))
                    continue

                if command == "@entropy":
                    state = self.history[id_][int(parts[0])]
                    state[parts[1]] = parts[2]
                    state["start"] = 0
                    state["stop"] = 5
                    utt = state['query']
                    self.states[id_] = state

                if command == "@next":
                    state = self.history[id_][int(parts[0])]
                    state['start'] = state['stop']
                    state['stop'] = state['stop'] + 5
                    utt = state['query']
                    self.states[id_] = state
            else:
                if id_ not in self.states:
                    self.states[id_] = {}

                self.states[id_]["start"] = 0
                self.states[id_]["stop"] = 5

            responses_batch, confidences_batch, state_batch = self.skills[0](
                [utt], self.history[id_], [self.states[id_]])

            # update `self.states` with retrieved results
            self.states[id_] = state_batch[0]
            self.states[id_]["query"] = utt

            items_batch, entropy_batch = responses_batch

            for batch_idx, items in enumerate(items_batch):

                self.history[id_].append(items)
                self.history[id_].append(self.states[id_])

                for idx, item in enumerate(items):
                    rich_message.add_control(_draw_item(item, idx, self.history[id_]))

                if len(items) == self.states[id_]['stop'] - self.states[id_]['start']:
                    buttons_frame = _draw_tail(entropy_batch[batch_idx], self.history[id_])
                    rich_message.add_control(buttons_frame)

        return [rich_message]


def _draw_tail(entropy, history):
    buttons_frame = ButtonsFrame(text="")
    buttons_frame.add_button(Button('More', "@next:" + str(len(history) - 1)))
    caption = "Press More "

    if entropy:
        caption += "specify a " + entropy[0][1]
        for ent_value in entropy[0][2][:4]:
            button_a = Button(ent_value[0], f'@entropy:{len(history) - 1}:{entropy[0][1]}:{ent_value[0]}')
            buttons_frame.add_button(button_a)

    buttons_frame.text = caption
    return buttons_frame


def _draw_item(item, idx, history):
    title = item['Title']
    if 'ListPrice' in item:
        title += " - **$" + item['ListPrice'].split('$')[1] + "**"

    buttons_frame = ButtonsFrame(text=title)
    buttons_frame.add_button(Button('Show details', "@details:" + str(len(history) - 2) + ":" + str(idx)))
    return buttons_frame


def show_details(item_data: Dict[Any, Any]) -> str:
    """Format catalog item output

    Parameters:
        item_data: item's attributes values

    Returns:
        [rich_message]: list of formatted rich message
    """

    txt = ""

    for key, value in item_data.items():
        txt += "**" + str(key) + "**" + ': ' + str(value) + "  \n"

    return txt


def make_agent() -> EcommerceAgent:
    """Make an agent

    Returns:
        agent: created Ecommerce agent
    """

    config_path = find_config('tfidf_retrieve')
    skill = build_model(config_path)
    agent = EcommerceAgent(skills=[skill])
    return agent


def main():
    """Parse parameters and run ms bot framework"""

    args = parser.parse_args()
    start_ms_bf_server(app_id=args.ms_id,
                       app_secret=args.ms_secret)


if __name__ == '__main__':
    main()
