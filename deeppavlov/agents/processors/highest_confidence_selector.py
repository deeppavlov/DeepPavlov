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
from deeppavlov.core.agent.processor import Processor


class HighestConfidenceSelector(Processor):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, utterances, batch_history, *responses):
        responses, confidences = zip(*[zip(*r) for r in responses])
        indexes = [c.index(max(c)) for c in zip(*confidences)]
        return [responses[i] for i, *responses in zip(indexes, *responses)]
