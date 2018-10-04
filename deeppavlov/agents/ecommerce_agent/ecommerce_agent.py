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

from collections import defaultdict
from typing import List, Tuple

import argparse
from deeppavlov.core.agent.agent import Agent
from deeppavlov.core.skill.skill import Skill
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.agent.rich_content import RichMessage
from deeppavlov.agents.rich_content.default_rich_content import PlainText, ButtonsFrame, Button
from deeppavlov.deep import find_config
from utils.ms_bot_framework_utils.server import run_ms_bot_framework_server

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--ms-id", help="microsoft bot framework app id", type=str)
parser.add_argument("-s", "--ms-secret", help="microsoft bot framework app secret", type=str)


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
        self.history: dict = defaultdict(list)
        self.states: dict = defaultdict(lambda: {} * len(self.skills))

    def _call(self, utterances_batch: list, utterances_ids: list=None) -> list:
        """Processes batch of utterances and returns corresponding responses batch.

        Each call of Agent passes incoming utterances batch through skills filter,
        agent skills, skills processor. Batch of dialog IDs can be provided, in
        other case utterances indexes in incoming batch are used as dialog IDs.

        Args:
            utterances: Batch of incoming utterances.
            ids: Batch of dialog IDs corresponding to incoming utterances.

        Returns:
            responses: A batch of responses corresponding to the
                utterance batch received by agent.
        """

        for utt, id_ in zip(utterances_batch, utterances_ids):

            print("utt", utt)

            if id_ not in self.states:
                self.states[id_] = {"start": 0, "stop": 5}

            if utt[0] == "@":
                parts = utt.split(":")
                print("actions", parts)

                if parts[0] == "@details":
                    return show_details(self.history[id_][int(parts[1])][0][int(parts[2])])

                if parts[0] == "@entropy":
                    state = self.history[id_][int(parts[1])]
                    state[parts[2]] = parts[3]
                    state["start"] = 0
                    state["stop"] = 5
                    utt = state['query']
                    self.states[id_] = state

                if parts[0] == "@next":
                    state = self.history[id_][int(parts[1])]
                    print(state)
                    state['start'] = state['stop']
                    state['stop'] = state['stop']+5
                    utt = state['query']
                    self.states[id_] = state

                if parts[0] == "@previous":
                    state = self.history[id_][int(parts[1])]
                    print(state)
                    state['stop'] = state['start']
                    state['start'] = state['start']-5
                    utt = state['query']
                    self.states[id_] = state
            else:
                self.states[id_]["start"] = 0
                self.states[id_]["stop"] = 5

            print('states', self.states[id_])
            print('history', self.history[id_])

            responses, confidences, state = self.skills[0]([utt], self.history[id_], [self.states[id_]])

            # update `self.states` with retrieved results
            self.states[id_] = state
            self.states[id_]["query"] = utt

            rich_message = RichMessage()
            items, entropy, total = responses

            self.history[id_].append(responses)
            for idx, item in enumerate(items):

                title = item['Title']
                if 'ListPrice' in item:
                    title += " - **$" + item['ListPrice'].split('$')[1]+"**"

                rich_message.add_control(PlainText(title))

                button_a = Button('Show details', "@details:"+str(len(self.history[id_])-1)+":"+str(idx))
                buttons_frame = ButtonsFrame(text="")
                buttons_frame.add_button(button_a)
                rich_message.add_control(buttons_frame)


            if self.states[id_]["start"] > 0:
                button_a = Button('Previous', "@previous:"+str(len(self.history[id_])-1))
                buttons_frame.add_button(button_a)

            button_b = Button('Next', "@next:"+str(len(self.history[id_])-1))
            buttons_frame.add_button(button_b)

            buttons_frame = ButtonsFrame(text="")
            rich_message.add_control(buttons_frame)

            self.history[id_].append(self.states[id_])
            if entropy:
                buttons_frame = ButtonsFrame(text="Please specify a "+entropy[0][1])
                for ent_value in entropy[0][2][:3]:
                    button_a = Button(ent_value[0], "@entropy:"+str(len(self.history[id_])-1)+":"+str(entropy[0][1])+":"+str(ent_value[0]))
                    buttons_frame.add_button(button_a)

                rich_message.add_control(buttons_frame)

        return [rich_message]


def show_details(item_data):
    """Formate catalog item output"""
    txt = ""
    cats = ['Title', 'Manufacturer', 'Model', 'ListPrice', 'Binding', 'Color',
            'Genre', 'Author', 'Brand', 'Size', 'Feature']

    for cat in cats:
        if cat in item_data:
            if cat == 'ListPrice':
                txt += "**"+cat+"**" + ': $' + item_data[cat].split('$')[1] + "  \n"
            else:
                txt += "**"+cat+"**" +  ': ' + item_data[cat] + "  \n"

    rich_message = RichMessage()
    rich_message.add_control(PlainText(txt))
    return [rich_message]

def make_agent():
    """Run skill"""
    config_path = find_config('ecommerce_bot')
    skill = build_model_from_config(config_path, as_component=True)
    agent = EcommerceAgent(skills=[skill])
    return agent

def main():
    """Parse parameters and run ms bot framework"""
    args = parser.parse_args()
    run_ms_bot_framework_server(agent_generator=make_agent,
                                app_id=args.ms_id,
                                app_secret=args.ms_secret,
                                stateful=True)


if __name__ == '__main__':
    main()
