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


from typing import List, Dict

import numpy as np
from sklearn.metrics import f1_score

from deeppavlov.core.common.metrics_registry import register_metric
from deeppavlov.models.classifiers.utils import labels2onehot


@register_metric('classification_f1')
def classification_fmeasure(y_true: List[list], predicted_labels: List[List[str]],
                            predicted_probabilities: List[Dict[str, float]], average: str="macro") -> float:
    """
    Calculate F1-measure

    Args:
        y_true: true binary labels
        predicted_labels: list of predicted labels for every sample
        predicted_probabilities: dictionary like {"label_i": probability_i} for every sample
        average: determines the type of averaging performed on the data

    Returns:
        F1-measure
    """
    classes = np.array(list(predicted_probabilities[0].keys()))
    y_true_one_hot = labels2onehot(y_true, classes)
    y_pred_one_hot = labels2onehot(predicted_labels, classes)

    return f1_score(y_true_one_hot, y_pred_one_hot, average=average)


@register_metric('classification_f1_weighted')
def classification_fmeasure_weighted(*args, **kwargs) -> float:
    """
    Calculate F1-measure weighted using :func:`classification_fmeasure`

    Returns:
        F1-measure
    """
    kwargs['average'] = 'weighted'

    return classification_fmeasure(*args, **kwargs)
