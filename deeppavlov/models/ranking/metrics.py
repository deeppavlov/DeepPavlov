from deeppavlov.core.common.metrics_registry import register_metric

import numpy as np


@register_metric('r@1')
def r_at_1(y_true, y_pred):
    return recall_at_k(y_true, y_pred, k=1)


@register_metric('r@2')
def r_at_2(y_true, y_pred):
    return recall_at_k(y_true, y_pred, k=2)


@register_metric('r@5')
def r_at_5(labels, predictions):
    return recall_at_k(labels, predictions, k=5)


def recall_at_k(y_true, y_pred, k):
    labels = np.array(y_true)
    predictions = np.array(y_pred)
    predictions = np.flip(np.argsort(predictions, -1), -1)[:, :k]
    flags = np.zeros_like(predictions)
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if predictions[i][j] in np.arange(labels[i][j]):
                flags[i][j] = 1.
    return np.mean((np.sum(flags, -1) >= 1.).astype(float))


@register_metric('rank_response')
def rank_response(y_true, y_pred):
    labels = np.array(y_true)
    predictions = np.array(y_pred)
    predictions = np.flip(np.argsort(predictions, -1), -1)
    ranks = []
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if predictions[i][j] in np.arange(labels[i][j]):
                ranks.append(j)
                break
    return np.mean(np.asarray(ranks).astype(float))


@register_metric('loss')
def triplet_loss(y_true, y_pred):
    margin = 0.1
    predictions = np.array(y_pred)
    pos_scores = predictions[:, 0]
    neg_scores = predictions[:, -1]
    return np.mean(np.maximum(margin - pos_scores + neg_scores, np.zeros(len(y_pred))), axis=-1)
