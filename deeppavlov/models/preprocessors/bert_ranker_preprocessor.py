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

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.commands.utils import expand_path
from logging import getLogger

from bert_dp.preprocessing import convert_examples_to_features, InputExample
from bert_dp.tokenization import FullTokenizer

logger = getLogger(__name__)


@register('bert_ranker_preprocessor')
class BertRankerPreprocessor(Component):
    # TODO: docs

    def __init__(self, vocab_file, do_lower_case=True, max_seq_length: int = 512,
                 resps= None, resp_vecs=None, conts= None, cont_vecs=None, *args, **kwargs):
        self.max_seq_length = max_seq_length
        vocab_file = str(expand_path(vocab_file))
        self.tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.resp_features = None
        self.cont_features = None
        if resps is not None and resp_vecs is None:
            resp_batch = [[el] for el in resps]
            self.resp_features = self(resp_batch)
        if conts is not None and cont_vecs is None:
            cont_batch = [[el] for el in conts]
            self.cont_features = self(cont_batch)

    def __call__(self, batch):
        if isinstance(batch[0], str):
            batch = [batch]
        samples = []
        for i in range(len(batch[0])):
            s = []
            for el in batch:
                s.append(el[i])
            samples.append(s)
        s_dummy = [None] * len(samples[0])
        # TODO: add unique id
        examples = []
        for s in samples:
            ex = [InputExample(unique_id=0, text_a=text_a, text_b=text_b) for text_a, text_b in
                  zip(s, s_dummy)]
            examples.append(ex)
        features = [convert_examples_to_features(el, self.max_seq_length, self.tokenizer) for el in examples]

        return features

