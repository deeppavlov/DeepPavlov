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


from typing import List

import numpy as np

from deeppavlov.core.common.metrics_registry import register_metric


def recall_at_k(y_true: List[int], y_pred: List[List[np.ndarray]], k: int):
    """
    Calculates recall at k ranking metric.

    Args:
        y_true: Labels. Not used in the calculation of the metric.
        y_predicted: Predictions.
            Each prediction contains ranking score of all ranking candidates for the particular data sample.
            It is supposed that the ranking score for the true candidate goes first in the prediction.

    Returns:
        Recall at k
    """
    num_examples = float(len(y_pred))
    predictions = np.array(y_pred)
    predictions = np.flip(np.argsort(predictions, -1), -1)[:, :k]
    num_correct = 0
    for el in predictions:
        if 0 in el:
            num_correct += 1
    return float(num_correct) / num_examples


@register_metric('r@1')
def r_at_1(y_true, y_pred):
    return recall_at_k(y_true, y_pred, k=1)


@register_metric('r@2')
def r_at_2(y_true, y_pred):
    return recall_at_k(y_true, y_pred, k=2)


@register_metric('r@5')
def r_at_5(labels, predictions):
    return recall_at_k(labels, predictions, k=5)


@register_metric('r@10')
def r_at_10(labels, predictions):
    return recall_at_k(labels, predictions, k=10)
