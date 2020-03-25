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

import numpy as np

from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('rank_response')
def rank_response(y_true, y_pred):
    num_examples = float(len(y_pred))
    predictions = np.array(y_pred)
    predictions = np.flip(np.argsort(predictions, -1), -1)
    rank_tot = 0
    for el in predictions:
        for i, x in enumerate(el):
            if x == 0:
                rank_tot += i
                break
    return float(rank_tot) / num_examples


@register_metric('r@1_insQA')
def r_at_1_insQA(y_true, y_pred):
    return recall_at_k_insQA(y_true, y_pred, k=1)


def recall_at_k_insQA(y_true, y_pred, k):
    labels = np.repeat(np.expand_dims(np.asarray(y_true), axis=1), k, axis=1)
    predictions = np.array(y_pred)
    predictions = np.flip(np.argsort(predictions, -1), -1)[:, :k]
    flags = np.zeros_like(predictions)
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if predictions[i][j] in np.arange(labels[i][j]):
                flags[i][j] = 1.
    return np.mean((np.sum(flags, -1) >= 1.).astype(float))
