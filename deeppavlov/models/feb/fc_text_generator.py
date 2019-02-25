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
import answers


from .feb_objects import *
from .feb_common import FebComponent


log = get_logger(__name__)


@register('feb_text_generator')
class FebTextGenerator(FebComponent):
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

        gen_context = {}
        gen_context['params'] = [e.to_values_dict() for e in utt.entities]

        if len(utt.intents) > 0:
            # TODO: case of many intents
            intent = utt.intents[0]
            gen_context['query_name'] = intent.type
            if intent.result_val:
                gen_context['results'] = intent.result_val
            else:
                gen_context['results'] = {'error': FebUtterance.ERROR_IN_RESULT}
        else:
            gen_context['query_name'] = FebIntent.INTENT_NOT_SET_TYPE
            gen_context['results'] = {'error': FebUtterance.ERROR_IN_RESULT}

        # TODO:
        # result = answers.answer(gen_context)

        utt.re_text = f'Result: {str(gen_context)} \n {repr(utt)}'
        # result = '; '.join(intent.result_str for intent in utt.intents if intent.result_str)
        #TODO:
        # utt.re_text = f'Result: {result} \n {repr(utt)}'
        return  utt



c_ = {
    'query_name': 'author_genres',
    'params': [
        {'type': 'author',
        'text': 'Достоевского',
        'normal_form': 'Достоевский',
        'qid': 'Q1234',
        'text_from_base': 'Федор Михайлович Достоевский'},
        {'type': 'author',
         'name_from_text': 'Пушкина',
         'name_normal_form': 'Пушки',
         'qid': 'Q1234',
         'name_from_base': 'Александр Сергеевич Пушкин'},
        {'type': 'book',
         'name_from_text': 'войны и мира',
         'name_normal_form': 'война и мир',
         'qid': 'Q1234',
         'name_from_base': 'Война и мир'},
        {'type': 'book',
         'name_from_text': 'Медного всадника',
         'name_normal_form': 'Медный всадник',
         'qid': 'Q1234',
         'name_from_base': 'Медный всадник'}
    ],
    'results': [
        {'error': 'DataNotFound'}]
}

a_ = """
a = {
    'params': {
        'query_name': 'author_genres',
        'author_name': 'Толстой'
    },
    'results': [
        {'genreLabel': 'роман'},
        {'genreLabel': 'драматическая форма'},
        {'genreLabel': 'рассказ'},
        {'genreLabel': 'повесть'}]
"""


    # don't override basic realization
    # def pack_result(self, utt, ret_obj_l):


