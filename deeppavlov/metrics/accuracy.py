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


import itertools

import numpy as np

from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('accuracy')
def accuracy(y_true: [list, np.ndarray], y_predicted: [list, np.ndarray]) -> float:
    """
    Calculate accuracy in terms of absolute coincidence

    Args:
        y_true: array of true values
        y_predicted: array of predicted values

    Returns:
        portion of absolutely coincidental samples
    """
    examples_len = len(y_true)
    correct = sum([y1 == y2 for y1, y2 in zip(y_true, y_predicted)])
    return correct / examples_len if examples_len else 0


@register_metric('sets_accuracy')
def sets_accuracy(y_true: [list, np.ndarray], y_predicted: [list, np.ndarray]) -> float:
    """
    Calculate accuracy in terms of sets coincidence

    Args:
        y_true: true values
        y_predicted: predicted values

    Returns:
        portion of samples with absolutely coincidental sets of predicted values
    """
    examples_len = len(y_true)
    correct = sum([set(y1) == set(y2) for y1, y2 in zip(y_true, y_predicted)])
    return correct / examples_len if examples_len else 0


@register_metric('slots_accuracy')
def slots_accuracy(y_true, y_predicted):
    y_true = [{tag.split('-')[-1] for tag in s if tag != 'O'} for s in y_true]
    y_predicted = [set(s.keys()) for s in y_predicted]
    return accuracy(y_true, y_predicted)


@register_metric('per_item_accuracy')
def per_item_accuracy(y_true, y_predicted):
    if isinstance(y_true[0], (tuple, list)):
        y_true = (y[0] for y in y_true)
    y_true = list(itertools.chain(*y_true))
    y_predicted = itertools.chain(*y_predicted)
    examples_len = len(y_true)
    correct = sum([y1 == y2 for y1, y2 in zip(y_true, y_predicted)])
    return correct / examples_len if examples_len else 0


@register_metric('per_token_accuracy')
def per_token_accuracy(y_true, y_predicted):
    y_true = list(itertools.chain(*y_true))
    y_predicted = itertools.chain(*y_predicted)
    examples_len = len(y_true)
    correct = sum([y1 == y2 for y1, y2 in zip(y_true, y_predicted)])
    return correct / examples_len if examples_len else 0


@register_metric('per_item_dialog_accuracy')
def per_item_dialog_accuracy(y_true, y_predicted):
    y_true = [y['text'] for dialog in y_true for y in dialog]
    y_predicted = itertools.chain(*y_predicted)
    examples_len = len(y_true)
    correct = sum([y1.strip().lower() == y2.strip().lower() for y1, y2 in zip(y_true, y_predicted)])
    return correct / examples_len if examples_len else 0


@register_metric('acc')
def round_accuracy(y_true, y_predicted):
    """
    Rounds predictions and calculates accuracy in terms of absolute coincidence.

    Args:
        y_true: list of true values
        y_predicted: list of predicted values

    Returns:
        portion of absolutely coincidental samples
    """
    predictions = [round(x) for x in y_predicted]
    examples_len = len(y_true)
    correct = sum([y1 == y2 for y1, y2 in zip(y_true, predictions)])
    return correct / examples_len if examples_len else 0
