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

from deeppavlov.deprecated.agents.default_agent import DefaultAgent
from deeppavlov.deprecated.agents.processors import HighestConfidenceSelector
from deeppavlov.deprecated.skills.pattern_matching_skill import PatternMatchingSkill


def make_hello_bot_agent() -> DefaultAgent:
    """Builds agent based on PatternMatchingSkill and HighestConfidenceSelector.

    This is agent building tutorial. You can use this .py file to check how hello-bot agent works.

    Returns:
        agent: Agent capable of handling several simple greetings.
    """
    skill_hello = PatternMatchingSkill(['Hello world'], patterns=['hi', 'hello', 'good day'])
    skill_bye = PatternMatchingSkill(['Goodbye world', 'See you around'], patterns=['bye', 'chao', 'see you'])
    skill_fallback = PatternMatchingSkill(['I don\'t understand, sorry', 'I can say "Hello world"'])

    agent = DefaultAgent([skill_hello, skill_bye, skill_fallback], skills_processor=HighestConfidenceSelector())

    return agent


if __name__ == '__main__':
    hello_bot_agent = make_hello_bot_agent()
    response = hello_bot_agent(['Hello', 'Bye', 'Or not'])
    print(response)
