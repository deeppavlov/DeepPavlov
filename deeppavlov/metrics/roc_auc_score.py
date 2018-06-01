"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sklearn.metrics
import numpy as np

from deeppavlov.core.common.metrics_registry import register_metric
from deeppavlov.models.classifiers.intents.utils import labels2onehot


def roc_auc_score_np(y_true, y_pred):
    """Compute Area Under the Curve (AUC) from prediction scores.

    Args:
        y_true: array of true binary labels
        y_pred: array of target scores, can either be probability estimates of the positive class

    Returns:
        Area Under the Curve (AUC) from prediction scores
    """
    try:
        return sklearn.metrics.roc_auc_score(np.array(y_true), np.array(y_pred), average="macro")
    except ValueError:
        return 0.


@register_metric('classification_roc_auc')
def roc_auc_score(y_true, y_predicted):
    """Compute Area Under the Curve (AUC) from prediction scores.

    Args:
        y_true: true binary labels
        y_predicted: list of predictions.
                Each prediction is a tuple of two elements
                (predicted_labels, dictionary like {"label_i": probability_i} )

    Returns:
        Area Under the Curve (AUC) from prediction scores
    """
    classes = np.array(list(y_predicted[0][1].keys()))
    y_true_one_hot = labels2onehot(y_true, classes)
    y_pred_probas = [list(y_predicted[i][1].values()) for i in range(len(y_predicted))]

    auc_score = roc_auc_score_np(y_true_one_hot, y_pred_probas)
    return auc_score
