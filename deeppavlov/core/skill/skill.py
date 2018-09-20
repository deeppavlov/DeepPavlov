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


class Skill(Component, metaclass=ABCMeta):
    """Abstract class for skills. Skill is a DeepPavlov component,
    which provides handling dialog state, dialog history and rich content.
    """
    @abstractmethod
    def __call__(self, utterances_batch: list, history_batch: list, states_batch: list = None):
        """Returns skill inference result.

        Returns skill inference result along with estimated confidence level,
        up to date conversation history and state for each utterance in the batch.

        Args:
            utterances_batch (list): A batch of utterances of any type
            history_batch (list): A batch of list typed histories
                for each utterance
            states_batch (list): Optional. A batch of arbitrary typed states
                for each utterance

        Returns:
            response (list): A batch of arbitrary typed skill inference results
            confidence (list): A batch of float typed confidence levels for
                each of skill inference result
            history (list): A batch of list typed histories for each response
            states (list): Optional. A batch of arbitrary typed states
                for each response

        """
        return
