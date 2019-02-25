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

from .feb_objects import *
from .feb_common import FebComponent

from question2wikidata import questions, functions


log = get_logger(__name__)


@register('feb_t1_parser')
class FebT1Parser(FebComponent):
    """Convert batch of strings
    sl = ["author_birthplace author Лев Николаевич Толстой",
      -(to)->
        utterence object
      """
    @classmethod
    def component_type(cls):
        return cls.START_COMPONENT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def test_and_prepare(self, utt):
        """
        Test input data and prepare data to process
        :param in_obj:
        :return: list (even if there is only one object to process!) of tuple(object, context)
            object - object for processing (must be instanceof FebObject)
            context - dictionary with context for processing
        """
        # if not isinstance(in_obj, str):
        #     raise TypeError(f"FebT1Parser is not implemented for `{type(in_obj)}`")
        # utt = FebUtterance(in_obj)

        entities = questions.extract_entities(utt.text) #получаем словарь с сущностями. ключ - book_name или author_name, значение - список сущностей
        var_dump(header='entities', msg=entities)

        utt.tokens = FebToken.tokenize(utt.text)
        tokens = [t for t in utt.tokens if t.type != FebToken.PUNCTUATION]

        entity_tokens, normal_form = [], None

        wrapped = questions.wrap_entities(utt.text, entities)
        var_dump(header='Строка с отмеченными сущностями', msg = wrapped)

        question_category = get_category(utt.text)  #получаем интент (get_category - временная, ищет тип вопроса в сообщении)

        entity_code = FebEntity.AUTHOR if FebEntity.AUTHOR in question_category else FebEntity.BOOK #вопрос о книге или об авторе?

        var_dump(header='entity_code', msg=entity_code)


        #пока считаем, что может быть 1 искомая сущность в вопросе

        if (entity_code == FebEntity.AUTHOR) and ('author_name' in entities): #если вопрос об авторе, не рассматриваем книжные сущности
            for author_tokens in entities['author_name']:
                for token in tokens:
                    if (token.start >= author_tokens['start']) and (token.stop <= author_tokens['stop']):
                        token.tags.update({FebToken.TAG_AUTHOR})
                        entity_tokens.append(token)
            normal_form = entities['author_name'][0]['normal_form']

        elif  (entity_code == FebEntity.BOOK) and ('book_name' in entities):  #если вопрос о книге, не рассматриваем авторские сущности
            for book_tokens in entities['book_name']:
                for token in tokens: 
                    if (token.start >= book_tokens['start']) and (token.stop <= book_tokens['stop']):
                        token.tags.update({FebToken.TAG_BOOK})
                        entity_tokens.append(token)  
            normal_form = entities['book_name'][0]['normal_form']

               
        var_dump(header='test_and_prepare t1_parser', msg = tokens)
        var_dump(header='Entity tokens', msg = entity_tokens)
        var_dump(header='entity_code', msg = entity_code)

        if question_category is None:
            utt.add_error(FebError(FebError.ET_INP_DATA, self, {FebError.EC_DATA_LACK: tokens}))


        return [(utt, {'intents_code':question_category,
                           'entity_code': entity_code,
                           'entity_text': (entity_tokens, normal_form)})]

    def process(self, obj, context):
        """
        Main processing function
        :param obj: obj to process
        :param context: dict with processing context
        :return: processed object
        """
        utt = obj
        intents_code = context['intents_code']
        entity_code = context['entity_code']
        entity_text, normal_form = context['entity_text']
        if FebIntent.in_supported_types(intents_code):
            intent = FebIntent(intents_code)
        else:
            intent = FebIntent(FebIntent.UNSUPPORTED_TYPE)
            intent.add_error(FebError(FebError.ET_INP_DATA, self, {FebError.EC_DATA_VAL: intents_code}))
        utt.intents.append(intent)

        if entity_code == FebEntity.AUTHOR:
            utt.entities.append(FebAuthor(tokens=entity_text, normal_form=normal_form))
        elif entity_code == FebEntity.BOOK:
            utt.entities.append(FebBook(tokens=entity_text, normal_form=normal_form))
        else:
            utt.add_error(FebError(FebError.ET_INP_DATA, self, {FebError.EC_DATA_VAL: entity_text}))
        return  utt

    # don't override basic realization
    # def pack_result(self, utt, ret_obj_l):



    # def _splitter(self, pars_str):
    #     log.info(f'feb_t1_parser _splitter pars_str={pars_str}')
    #     res = {}
    #     qnl = re.findall(r'^(\w+)', pars_str)
    #     if len(qnl) > 0:
    #         res['query_name'] = qnl[0]
    #     else:
    #         return {'error': 'question_type_not_found'}
    #     res['nent_lst'] = [{'nent_type': nent_type, 'nent_str': nent_str}
    #                        for nent_type, nent_str in re.findall(r'(?:<(.+?):(.+?)>)', pars_str)]
    #     log.info(f'feb_t1_parser _splitter res={res}')
    #     return res

