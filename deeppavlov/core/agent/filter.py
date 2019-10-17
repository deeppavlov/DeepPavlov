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


class Filter(Component, metaclass=ABCMeta):
    """Abstract class for filters. Filter is a DeepPavlov component,
    which is used in Agent to select utterances from incoming batch
    to be processed for each Agent skill.
    """

    @abstractmethod
    def __call__(self, utterances_batch: list, history_batch: list) -> list:
        """Returns skills-utterances application matrix.

        Returns skills-utterances application matrix which contains
        information about Agent skills to be applied to each utterance
        from incoming batch.

        Args:
            utterances_batch: A batch of utterances of any type.
            history_batch: A batch of list typed histories
                for each utterance.

        Returns:
            response: Skills-utterances application matrix,
            for example:
            [[True, False, True, True],
             [False, True, True, True]]
            Where each inner dict corresponds to one of the Agent
            skills and each value in the inner dict contains information
            about whether the skill will be applied to the utterance
            with the same position in the utterances_batch.

        """
        pass
