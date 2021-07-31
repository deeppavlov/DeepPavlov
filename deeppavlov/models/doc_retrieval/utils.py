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

import nltk


def find_answer_sentence(answer_pos: int, context: str) -> str:
    answer_sentence = ""
    context_sentences = nltk.sent_tokenize(context)
    start = 0
    context_sentences_offsets = []
    for sentence in context_sentences:
        end = start + len(sentence)
        context_sentences_offsets.append((start, end))
        start = end + 1

    for sentence, (start_offset, end_offset) in zip(context_sentences, context_sentences_offsets):
        if start_offset < answer_pos < end_offset:
            answer_sentence = sentence
            break

    return answer_sentence
