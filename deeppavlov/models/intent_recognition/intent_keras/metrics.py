from keras import backend as K
import sklearn.metrics
import numpy as np


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

def fmeasure(y_true, y_pred):
    try:
        _ = K.is_keras_tensor(y_pred)
        return fbeta_score_K(y_true, y_pred, beta=1)
    except ValueError:
        return fbeta_score_np(y_true, y_pred, beta=1)

def roc_auc_score(y_true, y_pred):
    """Compute Area Under the Curve (AUC) from prediction scores.

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
