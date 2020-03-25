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
from typing import List, Tuple, Any

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction, brevity_penalty, closest_ref_length

from deeppavlov.core.common.metrics_registry import register_metric
from deeppavlov.metrics.google_bleu import compute_bleu

SMOOTH = SmoothingFunction()


@register_metric('bleu_advanced')
def bleu_advanced(y_true: List[Any], y_predicted: List[Any],
                  weights: Tuple = (1,), smoothing_function=SMOOTH.method1,
                  auto_reweigh=False, penalty=True) -> float:
    """Calculate BLEU score

    Parameters:
        y_true: list of reference tokens
        y_predicted: list of query tokens
        weights: n-gram weights
        smoothing_function: SmoothingFunction
        auto_reweigh: Option to re-normalize the weights uniformly
        penalty: either enable brevity penalty or not

    Return:
        BLEU score
    """

    bleu_measure = sentence_bleu([y_true], y_predicted, weights, smoothing_function, auto_reweigh)

    hyp_len = len(y_predicted)
    hyp_lengths = hyp_len
    ref_lengths = closest_ref_length([y_true], hyp_len)

    bpenalty = brevity_penalty(ref_lengths, hyp_lengths)

    if penalty is True or bpenalty == 0:
        return bleu_measure

    return bleu_measure / bpenalty


@register_metric('bleu')
def bleu(y_true, y_predicted):
    return corpus_bleu([[y_t.lower().split()] for y_t in y_true],
                       [y_p.lower().split() for y_p in y_predicted])


@register_metric('google_bleu')
def google_bleu(y_true, y_predicted):
    return compute_bleu(([y_t.lower().split()] for y_t in y_true),
                        (y_p.lower().split() for y_p in y_predicted))[0]


@register_metric('per_item_bleu')
def per_item_bleu(y_true, y_predicted):
    y_predicted = itertools.chain(*y_predicted)
    return corpus_bleu([[y_t.lower().split()] for y_t in y_true],
                       [y_p.lower().split() for y_p in y_predicted])


@register_metric('per_item_dialog_bleu')
def per_item_dialog_bleu(y_true, y_predicted):
    y_true = (y['text'] for dialog in y_true for y in dialog)
    return corpus_bleu([[y_t.lower().split()] for y_t in y_true],
                       [y.lower().split() for y_p in y_predicted for y in y_p])
