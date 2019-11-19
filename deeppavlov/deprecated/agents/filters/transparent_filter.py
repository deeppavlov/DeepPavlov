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

from deeppavlov.deprecated.agent import Filter


class TransparentFilter(Filter):
    """Filter that applies each agent skill to all of batch utterances.

    Args:
        skills_count: Number of agent skills.

    Attributes:
        size: Number of agent skills.
    """

    def __init__(self, skills_count: int, *args, **kwargs) -> None:
        self.size: int = skills_count

    def __call__(self, utterances_batch: list, history_batch: list) -> list:
        """Returns skills-utterances application matrix.

        Generates skills-utterances application matrix with all True
        elements.

        Args:
            utterances_batch: A batch of utterances of any type.
            history_batch: Not used.

        Returns:
            response: Skills-utterances application matrix with all True
                elements.
        """
        return [[True] * len(utterances_batch)] * self.size
