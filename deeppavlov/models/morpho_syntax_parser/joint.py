# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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

from typing import Union, List

from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

UD_COLUMN_FEAT_MAPPING = {"id": 0, "word": 1, "lemma": 2, "upos": 3, "feats": 5, "head": 6, "deprel": 7}


@register("joint_tagger_parser")
class JointTaggerParser(Component):
    """
    A class to perform joint morphological and syntactic parsing.
    It is just a wrapper that calls the models for tagging and parsing
    and comprises their results in a single output.
    Args:
        tagger: the morphological tagger model (a :class:`~deeppavlov.core.common.chainer.Chainer` instance)
        parser_path: the syntactic parser model (a :class:`~deeppavlov.core.common.chainer.Chainer` instance)
        output_format: the output format, it may be either `ud` (alias: `conllu`) or `json`.
    Attributes:
        tagger: a morphological tagger model (a :class:`~deeppavlov.core.common.chainer.Chainer` instance)
        parser: a syntactic parser model (a :class:`~deeppavlov.core.common.chainer.Chainer` instance)
    """

    def __init__(self, tagger: Chainer, parser: Chainer,
                 output_format: str = "ud", *args, **kwargs):
        if output_format not in ["ud", "conllu", "json", "dict"]:
            UserWarning("JointTaggerParser output_format can be only `ud`, `conllu` or `json`. " \
                        "Unknown format: {}, setting the output_format to `ud`.".format(output_format))
            output_format = "ud"
        self.output_format = output_format
        self.tagger = tagger
        self.parser = parser

    def __call__(self, data: Union[List[str], List[List[str]]]) \
            -> Union[List[List[dict]], List[str], List[List[str]]]:
        tagger_output = self.tagger(data)
        parser_output = self.parser(data)
        answer = []
        for i, (tagger_sent, parser_sent) in enumerate(zip(tagger_output, parser_output)):
            curr_sent_answer = []
            for j, curr_word_tagger_output in enumerate(tagger_sent):
                curr_word_tagger_output = curr_word_tagger_output.split("\t")
                curr_word_parser_output = parser_sent[j].split("\t")
                curr_word_answer = curr_word_tagger_output[:]
                # setting parser output
                curr_word_answer[6:8] = curr_word_parser_output[6:8]
                if self.output_format in ["json", "dict"]:
                    curr_word_answer = {key: curr_word_answer[index]
                                        for key, index in UD_COLUMN_FEAT_MAPPING.items()}
                    curr_word_answer = str(curr_word_answer)
                curr_word_answer = "\t".join(curr_word_answer)
                curr_sent_answer.append(curr_word_answer)
            curr_sent_answer = "\n".join(str(x) for x in curr_sent_answer)
            answer.append(curr_sent_answer)
        return answer
