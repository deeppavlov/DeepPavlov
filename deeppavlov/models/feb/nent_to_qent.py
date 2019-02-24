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

# from .feb_common import NamedEntity, NamedEntityType, Utterance, UtteranceErrors

from .feb_objects import *
from .feb_common import FebComponent

from question2wikidata import questions, functions


log = get_logger(__name__)


@register('nent_to_qent')
class NentToQent(FebComponent):
    """Convert batch of strings
      """

    @classmethod
    def component_type(cls):
        return cls.INTERMEDIATE_COMPONENT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def test_and_prepare(self, utt: FebUtterance):
        """
        Test input data and prepare data to process
        :param utt: FebUtterance
        :return: list(tuple(FebEntity, {})) - for FebEntity context is void
        """
        return [(e, {}) for e in utt.entities]

    def process(self, entity: FebEntity, context):
        """
        Setting qid for entity
        :param entity: FebEntity
        :param context: void dict
        :return: None (all results saved in place (for arguments))
        """
        entity.qid = functions.get_qid(entity.tokens_to_search_string(), entity.type)
        return entity


    def pack_result(self, utt: FebUtterance, ret_obj_l):
        """
        Trivial packing
        :param utt: current FebUtterance
        :param ret_obj_l: list of entities
        :return: utt with list of updated entities
        """
        utt.entities = ret_obj_l
        return utt



    # @overrides
    # def __call__(self, batch, *args, **kwargs):
    #     for utt in batch:
    #         ne_l = utt.get(Utterance.NAMED_ENTITY_LST.value)
    #         for ne in ne_l:
    #             qid = self._extract_entities(ne.get(NamedEntity.NE_STRING.value),
    #                                          ne.get(NamedEntity.NE_TYPE.value))
    #             if qid:
    #                 ne[NamedEntity.NE_QID.value] = qid
    #             else:
    #                 utt[Utterance.ERROR.value] = UtteranceErrors.QID_NOT_FOUND.value
    #                 utt.get(Utterance.ERROR_VAL_LST.value, list()).append(ne)
    #     return batch
    #
    #
    # def _extract_entities(self, ne_str, param_type):
    #     log.info(f'nent_to_qent _extract_entities query={ne_str}, param_type={param_type}')
    #     qid = functions.get_qid(ne_str, param_type)
    #     log.info(f'nent_to_qent _extract_entities qid={qid}')
    #     return qid

