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

import itertools
from nltk.translate.bleu_score import sentence_bleu

from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('bleu')
def bleu(y_true, y_predicted):
    examples_len = len(y_true)
    bleu_list = (sentence_bleu([y2.lower().split()], y1.lower().split())\
                 for y1, y2 in zip(y_true, y_predicted))
    return sum(bleu_list) / examples_len if examples_len else 0.

@register_metric('per_item_bleu')
def per_item_bleu(y_true, y_predicted):
    if isinstance(y_true[0], (tuple, list)):
        y_true = map(lambda y: y[0], y_true)
    y_true = list(itertools.chain(*y_true))
    y_predicted = itertools.chain(*y_predicted)
    examples_len = len(y_true)
    bleu_list = (sentence_bleu([y2.lower().split()], y1.lower().split())\
                 for y1, y2 in zip(y_true, y_predicted))
    return sum(bleu_list) / examples_len if examples_len else 0.

@register_metric('per_item_dialog_bleu')
def per_item_dialog_bleu(y_true, y_predicted):
    y_true = [y['text'] for dialog in y_true for y in dialog]
    y_predicted = itertools.chain(*y_predicted)
    examples_len = len(y_true)
    bleu_list = (sentence_bleu([y2.lower().split()], y1.lower().split())\
                 for y1, y2 in zip(y_true, y_predicted))
    return sum(bleu_list) / examples_len if examples_len else 0.

