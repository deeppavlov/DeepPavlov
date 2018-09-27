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
from deeppavlov.core.agent.filter import Filter


class TransparentFilter(Filter):
    """Filer applies each agent skill to all of batch utterances.

    Args:
        skills_count (int): Number of agent skills.

    Attributes:
        size (int): Number of agent skills.
    """
    def __init__(self, skills_count, *args, **kwargs):
        self.size = skills_count

    def __call__(self, utterances_batch, history_batch):
        """Returns skills-utterances application matrix.

        Generates skills-utterances application matrix with all True
        elements.

        Args:
            utterances_batch (list): A batch of utterances of any type.
            history_batch (list): Not used.

        Returns:
            response (list): Skills-utterances application matrix with
                all True elements.
        """
        return [[True] * self.size] * len(utterances_batch)
