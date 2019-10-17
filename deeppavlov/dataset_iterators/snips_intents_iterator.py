# Copyright 2019 Alexey Romanov
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

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


@register('snips_intents_iterator')
class SnipsIntentIterator(DataLearningIterator):
    @overrides
    def preprocess(self, data, *args, **kwargs):
        result = []
        for query in data:
            text = ''.join(part['text'] for part in query['data'])
            intent = query['intent']
            result.append((text, intent))
        return result
