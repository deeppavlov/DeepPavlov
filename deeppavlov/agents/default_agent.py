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
import random

from deeppavlov.core.agent.filter import Filter
from deeppavlov.core.agent.processor import Processor
from deeppavlov.agents.default_rich_content import RichMessage, PlainText


class RandomProcessor(Processor):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, utterances, batch_history, *responses):
        return [random.choice([t for t, sc in r if t]) for r in zip(*responses)]


class HighestConfidenceProcessor(Processor):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, utterances, batch_history, *responses):
        responses, confidences = zip(*[zip(*r) for r in responses])
        indexes = [c.index(max(c)) for c in zip(*confidences)]
        return [responses[i] for i, *responses in zip(indexes, *responses)]


class DefaultRichContentProcessor(Processor):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, utterances, batch_history, *responses):
        responses, confidences = zip(*[zip(*r) for r in responses])
        indexes = [c.index(max(c)) for c in zip(*confidences)]
        result = []
        for i, *responses in zip(indexes, *responses):
            rich_message = RichMessage()
            plain_text = PlainText(str(responses[i]))
            rich_message.add_control(plain_text)
            result.append(rich_message)
        return result


class TransparentFilter(Filter):
    def __init__(self, skills_count, *args, **kwargs):
        self.size = skills_count

    def __call__(self, utterances, batch_history):
        return [[True] * self.size] * len(utterances)
