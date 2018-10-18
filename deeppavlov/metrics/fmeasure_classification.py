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


@register_metric('m_measure')
def M_measure(y_true: List[list], y_predicted: List[Tuple[list, dict]]) -> float:
    """
    Calculate F1-measure weighted

    Args:
        y_true: true labels
        y_predicted: predictions. \
            Each prediction is a tuple of two elements \
            (predicted_labels, dictionary like {"label_i": probability_i} ) \
            where probability is float or keras.tensor

    Returns:
        F1-measure
    """
    # constants
    ERR_COEF = 6.67
    default_class = 0  # "Прочее"
    rejection_class = -1  # "***REJECT***"
    thresholds_range = list(np.arange(0.00, 0.6, 0.01)) + list(np.arange(0.6, 0.99, 0.001)) + \
                       list(np.arange(0.99, 0.9999, 0.0001)) + list(np.arange(0.9999, 0.99999, 0.00001)) + \
                       list(np.arange(0.99999, 0.999999, 0.000001)) + list(np.arange(0.999999, 1, 0.0000001))

    # prepare data
    vocab_dict = {key: i for i, key in enumerate(list(y_predicted[0][1].keys()))}
    default_class = vocab_dict['Прочее']
    # print("default_class: ", default_class)
    # print("M classes: ", vocab_dict)

    y_true = np.array([vocab_dict[y[0]] for y in y_true])  # y[0] because y_true is [[tag_1], [tag_2], ... ]
    y_probas_dict = [y_predicted[i][1] for i in range(len(y_predicted))]
    y_pred = []
    for prob_dict in y_probas_dict:
        tmp_vec = []
        for key, item in prob_dict.items():
            tmp_vec.append(item)
        y_pred.append(np.array(tmp_vec))

    def metric(true_val, pred_val, normalize=False):
        if not isinstance(true_val, np.ndarray):
            true_val = np.array(true_val)
            pred_val = np.array(pred_val)

        n_error = sum((pred_val != true_val) & (pred_val != rejection_class) & (pred_val != default_class))
        n_rejection = sum(pred_val == rejection_class)
        n_defaults = sum(pred_val == default_class)

        m = -n_defaults - n_rejection - ERR_COEF * n_error

        if normalize:
            m /= len(true_val) * ERR_COEF

        return m

    def predict_with_threshold(preds, threshold=0.):
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)

        mask = preds > threshold
        reject = np.sum(mask, axis=1) == 0
        _preds = np.argmax(preds * mask, axis=1)
        _preds = _preds - reject
        return _preds

    def grid_search(true, pred, thresholds):
        metrics = np.array([
            metric(true, predict_with_threshold(pred, 0.5), normalize=True)
            for t in thresholds
        ])
        best_idx = np.argmax(metrics)
        best_t = thresholds[best_idx]
        best_m = metrics[best_idx]
        return best_t, best_m

    # calculate metric best metric
    best_threshold, best_metric = grid_search(y_true, y_pred, thresholds_range)

    # print("Best threshold: {}".format(best_threshold))

    return best_metric


@register_metric('m_measure_threshold')
def M_measure_threshold(y_true: List[list], y_predicted: List[Tuple[list, dict]]) -> float:
    """
    Calculate F1-measure weighted

    Args:
        y_true: true labels
        y_predicted: predictions. \
            Each prediction is a tuple of two elements \
            (predicted_labels, dictionary like {"label_i": probability_i} ) \
            where probability is float or keras.tensor

    Returns:
        F1-measure
    """
    # constants
    ERR_COEF = 6.67
    default_class = 0  # "Прочее"
    rejection_class = -1  # "***REJECT***"
    thresholds_range = list(np.arange(0.00, 0.6, 0.01)) + list(np.arange(0.6, 0.99, 0.001)) + \
                       list(np.arange(0.99, 0.9999, 0.0001)) + list(np.arange(0.9999, 0.99999, 0.00001)) + \
                       list(np.arange(0.99999, 0.999999, 0.000001)) + list(np.arange(0.999999, 1, 0.0000001))

    # prepare data
    vocab_dict = {key: i for i, key in enumerate(list(y_predicted[0][1].keys()))}
    default_class = vocab_dict['Прочее']
    # print("default_class: ", default_class)
    # print("M classes: ", vocab_dict)

    y_true = np.array([vocab_dict[y[0]] for y in y_true])  # y[0] because y_true is [[tag_1], [tag_2], ... ]
    y_probas_dict = [y_predicted[i][1] for i in range(len(y_predicted))]
    y_pred = []
    for prob_dict in y_probas_dict:
        tmp_vec = []
        for key, item in prob_dict.items():
            tmp_vec.append(item)
        y_pred.append(np.array(tmp_vec))

    def metric(true_val, pred_val, normalize=False):
        if not isinstance(true_val, np.ndarray):
            true_val = np.array(true_val)
            pred_val = np.array(pred_val)

        n_error = sum((pred_val != true_val) & (pred_val != rejection_class) & (pred_val != default_class))
        n_rejection = sum(pred_val == rejection_class)
        n_defaults = sum(pred_val == default_class)

        m = -n_defaults - n_rejection - ERR_COEF * n_error

        if normalize:
            m /= len(true_val) * ERR_COEF

        return m

    def predict_with_threshold(preds, threshold=0.):
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)

        mask = preds > threshold
        reject = np.sum(mask, axis=1) == 0
        _preds = np.argmax(preds * mask, axis=1)
        _preds = _preds - reject
        return _preds

    def grid_search(true, pred, thresholds):
        metrics = np.array([
            metric(true, predict_with_threshold(pred, t), normalize=True)
            for t in thresholds
        ])
        best_idx = np.argmax(metrics)
        best_t = thresholds[best_idx]
        best_m = metrics[best_idx]
        return best_t, best_m

    # calculate metric best metric
    best_threshold, best_metric = grid_search(y_true, y_pred, thresholds_range)

    # print("Best threshold: {}".format(best_threshold))

    return best_metric


@register_metric('threshold')
def threshold(y_true: List[list], y_predicted: List[Tuple[list, dict]]) -> float:
    """
    Calculate F1-measure weighted

    Args:
        y_true: true labels
        y_predicted: predictions. \
            Each prediction is a tuple of two elements \
            (predicted_labels, dictionary like {"label_i": probability_i} ) \
            where probability is float or keras.tensor

    Returns:
        F1-measure
    """
    # constants
    ERR_COEF = 6.67
    default_class = 0  # "Прочее"
    rejection_class = -1  # "***REJECT***"
    thresholds_range = list(np.arange(0.00, 0.6, 0.01)) + list(np.arange(0.6, 0.99, 0.001)) + \
                       list(np.arange(0.99, 0.9999, 0.0001)) + list(np.arange(0.9999, 0.99999, 0.00001)) + \
                       list(np.arange(0.99999, 0.999999, 0.000001)) + list(np.arange(0.999999, 1, 0.0000001))

    # prepare data
    vocab_dict = {key: i for i, key in enumerate(list(y_predicted[0][1].keys()))}
    default_class = vocab_dict['Прочее']
    # print("default_class: ", default_class)
    # print("M classes: ", vocab_dict)

    y_true = np.array([vocab_dict[y[0]] for y in y_true])  # y[0] because y_true is [[tag_1], [tag_2], ... ]
    y_probas_dict = [y_predicted[i][1] for i in range(len(y_predicted))]
    y_pred = []
    for prob_dict in y_probas_dict:
        tmp_vec = []
        for key, item in prob_dict.items():
            tmp_vec.append(item)
        y_pred.append(np.array(tmp_vec))

    def metric(true_val, pred_val, normalize=False):
        if not isinstance(true_val, np.ndarray):
            true_val = np.array(true_val)
            pred_val = np.array(pred_val)

        n_error = sum((pred_val != true_val) & (pred_val != rejection_class) & (pred_val != default_class))
        n_rejection = sum(pred_val == rejection_class)
        n_defaults = sum(pred_val == default_class)

        m = -n_defaults - n_rejection - ERR_COEF * n_error

        if normalize:
            m /= len(true_val) * ERR_COEF

        return m

    def predict_with_threshold(preds, threshold=0.):
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)

        mask = preds > threshold
        reject = np.sum(mask, axis=1) == 0
        _preds = np.argmax(preds * mask, axis=1)
        _preds = _preds - reject
        return _preds

    def grid_search(true, pred, thresholds):
        metrics = np.array([
            metric(true, predict_with_threshold(pred, t), normalize=True)
            for t in thresholds
        ])
        best_idx = np.argmax(metrics)
        best_t = thresholds[best_idx]
        best_m = metrics[best_idx]
        return best_t, best_m

    # calculate metric best metric
    best_threshold, best_metric = grid_search(y_true, y_pred, thresholds_range)

    # print("Best threshold: {}".format(best_threshold))

    return best_threshold


@register_metric('simple_accuracy')
def simple_acc(y_true, y_predicted) -> float:
    """
    Calculate accuracy

    Args:
        y_true: true binary labels
        y_predicted: predictions

    Returns:
        Accuracy
    """
    return accuracy_score(y_true, y_predicted)


@register_metric('simple_f1_macro')
def simple_f1_macro(y_true: List[float], y_predicted: List[float], average="macro") -> float:
    """
    Calculate F1-measure macro

    Args:
        y_true: true binary labels
        y_predicted: predictions
        average: determines the type of averaging performed on the data

    Returns:
        F1-measure macro
    """
    return f1_score(y_true, y_predicted, average=average)


@register_metric('simple_f1_weighted')
def simple_f1_weighted(y_true, y_predicted, average="weighted") -> float:
    """
    Calculate F1-measure weighted

    Args:
        y_true: true binary labels
        y_predicted: predictions
        average: determines the type of averaging performed on the data

    Returns:
        F1-measure weighted
    """
    return f1_score(y_true, y_predicted, average=average)
