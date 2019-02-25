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

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger
import re
from answers_generator.answers import

from .feb_objects import *
from .feb_common import FebComponent


log = get_logger(__name__)


@register('feb_t1_text_generator')
class FebT1TextGenerator(FebComponent):
    """Convert utt to strings
      """
    @classmethod
    def component_type(cls):
        return cls.FINAL_COMPONENT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # don't override basic realization
    # def test_and_prepare(self, utt):

    def process(self, utt: FebUtterance, context):
        """
        Main processing function
        :param obj: obj to process
        :param context: dict with processing context
        :return: processed object
        """
        result = '; '.join(intent.result_str for intent in utt.intents if intent.result_str)
        utt.re_text = f'Result: {result} \n {repr(utt)}'
        return  utt

    # don't override basic realization
    # def pack_result(self, utt, ret_obj_l):


