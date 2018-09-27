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

from deeppavlov.core.agent.processor import Processor


class RandomSelector(Processor):
    """Returns response of a random skill for each utterance.
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, utterances, batch_history, *responses):
        """Selects result of a random skill for each utterance.

        Args:
            utterances_batch (list): Not used.
            history_batch (list): Not used.
            responses (list): Each response positional argument corresponds to
                response of one of Agent skills and is represented by
                batch (list) of (response, confidence) tuple structures.

        Returns:
            responses (list): A batch of responses corresponding to the
                utterance batch received by agent.
        """
        return [random.choice([t for t, sc in r if t]) for r in zip(*responses)]
