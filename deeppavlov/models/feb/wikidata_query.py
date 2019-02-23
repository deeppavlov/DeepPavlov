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

from .feb_common import NamedEntity, NamedEntityType, Utterance, UtteranceErrors
from question2wikidata import questions, functions
from question2wikidata.server_queries import queries


log = get_logger(__name__)


@register('wikiddata_query')
class WikidataQuery(Component):
    """Convert batch of strings
    sl = ["my_q ",
          "my_q <author:Лев Николаевич Толстой>",
          "my_q <author:Лев Николаевич Толстой> <t2:Лев Николаевич Толстой>"]

      -(to)->

        {'query_name': 'my_q', 'nent_lst': []}
        {'query_name': 'my_q', 'nent_lst': [{'nent_type': 'author', 'nent_str': 'Лев Николаевич Толстой'}]}
        {'query_name': 'my_q', 'nent_lst': [
            {'nent_type': 'author', 'nent_str': 'Лев Николаевич Толстой'},
            {'nent_type': 't2', 'nent_str': 'Лев Николаевич Толстой'}]}
      """
    def __init__(self, **kwargs):
        pass
        # # the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
        # log.info(f'nent_to_qent __init__ question2wikidata_path={question2wikidata_path}')
        # sys.path.append(question2wikidata_path)
        # import questions

    @overrides
    def __call__(self, batch, *args, **kwargs):
        for utt in batch:
            utt_type = utt.get(Utterance.TYPE.value)
            ne_l = utt.get(Utterance.NAMED_ENTITY_LST.value)
            res_type, res_val = self._make_query(utt_type, ne_l)
            utt[res_type] = res_val
            # for ne in ne_l:
            #     qid = self._make_query(utt_type,
            #                                  ne.get(NamedEntity.NE_TYPE.value))
            #     if qid:
            #         ne[NamedEntity.NE_QID.value] = qid
            #     else:
            #         utt[Utterance.ERROR.value] = UtteranceErrors.QID_NOT_FOUND.value
            #         utt.get(Utterance.ERROR_VAL_LST.value, list()).append(ne)
        return batch


    def _make_query(self, query, ne_l):
        log.info(f'wikiddata_query _make_query query={query}, ne_l={ne_l}')
        try:
            if query in queries:
                qparams = {}
                for param_name, param_type in queries[query]['params'].items():
                    params = [e[NamedEntity.NE_QID.value] for e in ne_l if e[NamedEntity.NE_TYPE.value] == param_type]
                    if len(params) == 1:
                        qparams[param_name] = params[0]
                    else:
                        return Utterance.ERROR.value, UtteranceErrors.WIKIDATA_QUERY_ERROR.value  # TODO: Error!

                qr = queries[query]['query']
                log.info(f'wikiddata_query _make_query query_name={query}, qparams={qparams}, query={qr}')
                query_result = functions.execute_query(qr, **qparams)
                log.info(f'wikiddata_query _make_query query_result={query_result}')
                # return Utterance.ERROR.value, UtteranceErrors.WIKIDATA_QUERY_ERROR.value
                return Utterance.RESULT.value, query_result
            else:
                return Utterance.ERROR.value, UtteranceErrors.WIKIDATA_QUERY_ERROR.value  # TODO: Error!
        except:
            return Utterance.ERROR.value, UtteranceErrors.WIKIDATA_QUERY_ERROR.value  # TODO: Error!

        # query_result = functions.execute_query(query, **qparams)
        # # qid = functions.get_qid(ne_str, param_type)
        # log.info(f'wikiddata_query _make_query query_result={query_result}')
        # return qid

