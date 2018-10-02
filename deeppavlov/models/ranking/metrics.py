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
from sklearn.metrics import f1_score, log_loss

from deeppavlov.core.common.metrics_registry import register_metric


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

def recall_at_k(y_true, y_pred, k):
    num_examples = float(len(y_pred))
    predictions = np.array(y_pred)
    predictions = np.flip(np.argsort(predictions, -1), -1)[:, :k]
    num_correct = 0
    for el in predictions:
        if 0 in el:
            num_correct += 1
    return float(num_correct) / num_examples

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
    return float(rank_tot)/num_examples

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

@register_metric('s_acc')
def siamese_acc(y_true, y_predicted):
    """
    Calculate accuracy in terms of absolute coincidence

    Args:
        y_true: array of true values
        y_predicted: array of predicted values

    Returns:
        portion of absolutely coincidental samples
    """
    predictions = list(map(lambda x: round(x), list(np.squeeze(y_predicted, 1))))
    examples_len = len(y_true)
    correct = sum([y1 == y2 for y1, y2 in zip(y_true, predictions)])
    return correct / examples_len if examples_len else 0

@register_metric('s_f1')
def siamese_f1(y_true, y_predicted):
    """
    Calculate accuracy in terms of absolute coincidence

    Args:
        y_true: array of true values
        y_predicted: array of predicted values

    Returns:
        F1 score of the positive class in binary classification
    """
    predictions = list(map(lambda x: round(x), list(np.squeeze(y_predicted, 1))))
    return f1_score(y_true, predictions)

@register_metric('s_log_loss')
def siamese_log_loss(y_true, y_predicted):
    """
    Calculate accuracy in terms of absolute coincidence

    Args:
        y_true: array of true values
        y_predicted: array of predicted values

    Returns:
        Log loss in binary classification
    """
    predictions = list(np.squeeze(y_predicted, 1))
    return log_loss(y_true, predictions)
