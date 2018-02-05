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
import sklearn.metrics


def precision_K(y_true, y_pred):
    """
    Calculate precision for keras tensors
    Args:
        y_true: true labels
        y_pred: predicted labels

    Returns:
        precision
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall_K(y_true, y_pred):
    """
    Calculate recall for keras tensors
    Args:
        y_true: true labels
        y_pred: predicted labels

    Returns:
        recall
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score_K(y_true, y_pred, beta=1):
    """
    Calculate f-beta score for keras tensors
    Args:
        y_true: true labels
        y_pred: predicted labels

    Returns:
        f-beta score
    """
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
    """
    Calculate precision for numpy arrays
    Args:
        y_true: true labels
        y_pred: predicted labels

    Returns:
        f-beta score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + 10e-8)
    return precision


def recall_np(y_true, y_pred):
    """
    Calculate recall for numpy arrays
    Args:
        y_true: true labels
        y_pred: predicted labels

    Returns:
        recall
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 10e-8)
    return recall


def fbeta_score_np(y_true, y_pred, beta=1):
    """
    Calculate f-beta score for numpy arrays
    Args:
        y_true: true labels
        y_pred: predicted labels

    Returns:
        f-beta score
    """
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


def fmeasure(y_true, y_pred):
    """
    Calculate F1 score for given numpy arrays or keras tensors
    Args:
        y_true: true labels
        y_pred: predicted labels

    Returns:
        F1 score
    """
    try:
        _ = K.is_keras_tensor(y_pred)
        return fbeta_score_K(y_true, y_pred, beta=1)
    except ValueError:
        return fbeta_score_np(y_true, y_pred, beta=1)


def roc_auc_score(y_true, y_pred):
    """
    Compute Area Under the Curve (AUC) from prediction scores
    Args:
        y_true: true binary labels
        y_pred: target scores, can either be probability estimates of the positive class

    Returns:
        Area Under the Curve (AUC) from prediction scores
    """
    try:
        return sklearn.metrics.roc_auc_score(y_true.reshape(-1), y_pred.reshape(-1))
    except ValueError:
        return 0.
