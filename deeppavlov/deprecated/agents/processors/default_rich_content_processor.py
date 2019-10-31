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

from deeppavlov.deprecated.agent import Processor, RichMessage
from deeppavlov.deprecated.agents.rich_content import PlainText


class DefaultRichContentWrapper(Processor):
    """Returns RichControl wrapped responses with highest confidence."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, utterances: list, batch_history: list, *responses: list) -> list:
        """Selects for each utterance response with highest confidence and wraps them to RichControl objects.

        Args:
            utterances_batch: Not used.
            history_batch: Not used.
            responses: Each response positional argument corresponds to
                response of one of Agent skills and is represented by
                batch (list) of (response, confidence) tuple structures.

        Returns:
            result: A batch of responses corresponding to the utterance
                batch received by agent.
        """
        responses, confidences = zip(*[zip(*r) for r in responses])
        indexes = [c.index(max(c)) for c in zip(*confidences)]
        result = []
        for i, *responses in zip(indexes, *responses):
            rich_message = RichMessage()
            plain_text = PlainText(str(responses[i]))
            rich_message.add_control(plain_text)
            result.append(rich_message)
        return result
