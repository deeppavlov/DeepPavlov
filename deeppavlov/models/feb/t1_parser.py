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

log = get_logger(__name__)



@register('feb_t1_parser')
class FebT1Parser(Component):
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

    @overrides
    def __call__(self, batch, *args, **kwargs):
        if len(batch) > 0 and isinstance(batch[0], str):
            batch = [self._splitter(utt) for utt in batch]
            return batch
        raise TypeError(
            f"StreamSpacyTokenizer.__call__() is not implemented for `{type(batch[0])}`")

    def _splitter(self, pars_str):
        log.info(f'feb_t1_parser _splitter pars_str={pars_str}')
        res = {}
        qnl = re.findall(r'^(\w+)', pars_str)
        if len(qnl) > 0:
            res['query_name'] = qnl[0]
        else:
            return {'error': 'question_type_not_found'}
        res['nent_lst'] = [{'nent_type': nent_type, 'nent_str': nent_str}
                           for nent_type, nent_str in re.findall(r'(?:<(.+?):(.+?)>)', pars_str)]
        log.info(f'feb_t1_parser _splitter res={res}')
        return res


