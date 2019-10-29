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

from abc import ABCMeta, abstractmethod

from deeppavlov.core.models.component import Component


class Processor(Component, metaclass=ABCMeta):
    """Abstract class for processors. Processor is a DeepPavlov component,
    which is used in Agent to process skills responses and give one final
    response for each utterance.
    """

    # TODO: change *responses to [[], [], ...] argument
    @abstractmethod
    def __call__(self, utterances_batch: list, history_batch: list, *responses: list) -> list:
        """Returns final response for each incoming utterance.

        Processes Agent skills and generates one final response for each
        utterance in incoming batch.

        Args:
            utterances_batch: A batch of utterances of any type
            history_batch: A batch of list typed histories
                for each utterance
            responses: Each response positional argument corresponds to
                response of one of Agent skills and is represented by
                batch (list) of (response, confidence) tuple structures.

        Returns:
            responses: A batch of responses corresponding to the
                utterance batch received by agent.
        """
        pass
