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

from typing import Set, Tuple

from rusenttokenize import ru_sent_tokenize, SHORTENINGS, JOINING_SHORTENINGS, PAIRED_SHORTENINGS

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register("ru_sent_tokenizer")
class RuSentTokenizer(Component):
    """
    Rule-base sentence tokenizer for Russian language.
    https://github.com/deepmipt/ru_sentence_tokenizer

    Args:
        shortenings: list of known shortenings. Use default value if working on news or fiction texts
        joining_shortenings: list of shortenings after that sentence split is not possible (i.e. "ул").
            Use default value if working on news or fiction texts
        paired_shortenings: list of known paired shotenings (i.e. "т. е.").
            Use default value if working on news or fiction texts

    """

    def __init__(self, shortenings: Set[str] = SHORTENINGS,
                 joining_shortenings: Set[str] = JOINING_SHORTENINGS,
                 paired_shortenings: Set[Tuple[str, str]] = PAIRED_SHORTENINGS,
                 **kwargs):
        self.shortenings = shortenings
        self.joining_shortenings = joining_shortenings
        self.paired_shortenings = paired_shortenings

    def __call__(self, batch: [str]) -> [[str]]:
        return [ru_sent_tokenize(x, self.shortenings, self.joining_shortenings, self.paired_shortenings) for x in batch]
