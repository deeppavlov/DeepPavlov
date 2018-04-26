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

import numpy as np

from keras import backend as K

from deeppavlov.core.common.metrics_registry import register_metric
from deeppavlov.models.classifiers.intents.utils import labels2onehot


def precision_K(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall_K(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score_K(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision_K(y_true, y_pred)
    r = recall_K(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def precision_np(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + 10e-8)
    return precision


def recall_np(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 10e-8)
    return recall


def fbeta_score_np(y_true, y_pred, beta=1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if np.sum(np.round(np.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision_np(y_true, y_pred)
    r = recall_np(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + 10e-8)
    return fbeta_score


@register_metric('classification_f1')
def fmeasure(y_true, y_predicted):
    classes = np.array(list(y_predicted[0][1].keys()))
    y_true_one_hot = labels2onehot(y_true, classes)
    y_pred_labels = [y_predicted[i][0] for i in range(len(y_predicted))]
    y_pred_one_hot = labels2onehot(y_pred_labels, classes)

    try:
        _ = K.is_keras_tensor(y_pred_one_hot)
        return fbeta_score_K(y_true_one_hot, y_pred_one_hot, beta=1)
    except ValueError:
        return fbeta_score_np(y_true_one_hot, y_pred_one_hot, beta=1)
