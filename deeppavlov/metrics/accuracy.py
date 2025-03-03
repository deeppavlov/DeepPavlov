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
import re
from logging import getLogger
from typing import List

import numpy as np

from deeppavlov.core.common.metrics_registry import register_metric

log = getLogger(__name__)


@register_metric('accuracy')
def accuracy(y_true: [list, np.ndarray], y_predicted: [list, np.ndarray]) -> float:
    """
    Calculate accuracy in terms of absolute coincidence

    Args:
        y_true: array of true values
        y_predicted: array of predicted values

    Returns:
        fraction of absolutely coincidental samples
    """
    examples_len = len(y_true)
    # if y1 and y2 are both arrays, == can be erroneously interpreted as element-wise equality

    def _are_equal(y1, y2):
        answer = (y1 == y2)
        if isinstance(answer, np.ndarray):
            answer = answer.all()
        return answer

    equalities = [_are_equal(y1, y2) for y1, y2 in zip(y_true, y_predicted)]
    correct = sum(equalities)
    return correct / examples_len if examples_len else 0


@register_metric('kbqa_accuracy')
def kbqa_accuracy(questions_batch, pred_answer_labels_batch, pred_answer_ids_batch, pred_query_batch,
                  gold_answer_labels_batch, gold_answer_ids_batch, gold_query_batch) -> float:
    num_samples = len(pred_answer_ids_batch)
    correct = 0
    for question, pred_answer_label, pred_answer_ids, pred_query, gold_answer_labels, gold_answer_ids, gold_query in \
            zip(questions_batch, pred_answer_labels_batch, pred_answer_ids_batch, pred_query_batch,
                gold_answer_labels_batch, gold_answer_ids_batch, gold_query_batch):
        found_date = False
        if pred_answer_ids and gold_answer_ids and re.findall(r"[\d]{3,4}", pred_answer_ids[0]) and \
                re.findall(r"[\d]{3,4}", pred_answer_ids[0]) == re.findall(r"[\d]{3,4}", gold_answer_ids[0]):
            found_date = True
        found_label = False
        if len(gold_answer_labels) == 1 and len(pred_answer_label) > 1 and pred_answer_label == gold_answer_labels[0]:
            found_label = True
        no_answer = False
        if pred_answer_label == "Not Found" and not gold_answer_ids:
            no_answer = True
        if set(pred_answer_ids) == set(gold_answer_ids) or gold_query in pred_query or found_date or found_label \
                or no_answer:
            correct += 1
        log.debug(f"question: {question} -- gold_answer_ids: {gold_answer_ids} -- pred_answer_ids: {pred_answer_ids}")
    return correct / num_samples if num_samples else 0


@register_metric('multitask_accuracy')
def multitask_accuracy(*args) -> float:
    """
    Accuracy for multiple simultaneous tasks.

    Args:
        *args: a list of `2n` inputs. The first `n` inputs are the correct answers for `n` tasks,
            and the last `n` are the predicted ones.

    Returns:
        The percentage of inputs where the answers for all `n` tasks are correct.
    """
    n = len(args)
    y_true_by_tasks, y_predicted_by_tasks = args[:n // 2], args[n // 2:]
    answers = []
    for true, pred in zip(y_true_by_tasks, y_predicted_by_tasks):
        answers.append(accuracy(true, pred))
    final_answer = sum(answers)/len(answers)
    return final_answer


@register_metric('multitask_sequence_accuracy')
def multitask_sequence_accuracy(*args) -> float:
    """
    Accuracy for multiple simultaneous sequence labeling (tagging) tasks.
    For each sequence the model checks whether all its elements
    are labeled correctly for all the individual taggers.

    Args:
        *args: a list of `2n` inputs. The first `n` inputs are the correct answers for `n` tasks,
            and the last `n` are the predicted ones. For each task an

    Returns:
        The percentage of sequences where all the items has correct answers for all `n` tasks.

    """
    n = len(args)
    y_true_by_tasks, y_predicted_by_tasks = args[:n // 2], args[n // 2:]
    y_true_by_sents = list(zip(*y_true_by_tasks))
    y_predicted_by_sents = list(zip(*y_predicted_by_tasks))
    y_true = list(list(zip(*elem)) for elem in y_true_by_sents)
    y_predicted = list(list(zip(*elem)) for elem in y_predicted_by_sents)
    return accuracy(y_true, y_predicted)


@register_metric('multitask_token_accuracy')
def multitask_token_accuracy(*args) -> float:
    """
        Per-item accuracy for multiple simultaneous sequence labeling (tagging) tasks.

        Args:
            *args: a list of `2n` inputs. The first `n` inputs are the correct answers for `n` tasks
                and the last `n` are the predicted ones. For each task an

        Returns:
            The percentage of sequence elements for which the answers for all `n` tasks are correct.

        """
    n = len(args)
    y_true_by_tasks, y_predicted_by_tasks = args[:n // 2], args[n // 2:]
    y_true_by_sents = list(zip(*y_true_by_tasks))
    y_predicted_by_sents = list(zip(*y_predicted_by_tasks))
    y_true = list(list(zip(*elem)) for elem in y_true_by_sents)
    y_predicted = list(list(zip(*elem)) for elem in y_predicted_by_sents)
    return per_token_accuracy(y_true, y_predicted)


@register_metric('sets_accuracy')
def sets_accuracy(y_true: [list, np.ndarray], y_predicted: [list, np.ndarray]) -> float:
    """
    Calculate accuracy in terms of sets coincidence

    Args:
        y_true: true values
        y_predicted: predicted values

    Returns:
        portion of samples with absolutely coincidental sets of predicted values

    Alias:
        sets_accuracy
    """
    examples_len = len(y_true)
    correct = sum([set(y1) == set(y2) for y1, y2 in zip(y_true, y_predicted)])
    return correct / examples_len if examples_len else 0


@register_metric('slots_accuracy')
def slots_accuracy(y_true, y_predicted):
    y_true = [{tag.split('-')[-1] for tag in s if tag != 'O'} for s in y_true]
    y_predicted = [set(s.keys()) for s in y_predicted]
    return accuracy(y_true, y_predicted)


@register_metric('per_token_accuracy')
def per_token_accuracy(y_true, y_predicted):
    y_true = list(itertools.chain(*y_true))
    y_predicted = itertools.chain(*y_predicted)
    examples_len = len(y_true)
    correct = sum([y1 == y2 for y1, y2 in zip(y_true, y_predicted)])
    return correct / examples_len if examples_len else 0


# region go-bot metrics

@register_metric('per_item_dialog_accuracy')
def per_item_dialog_accuracy(y_true, y_predicted: List[List[str]]):
    # todo metric classes???
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
    if isinstance(y_predicted[0], np.ndarray):
        predictions = [np.round(x) for x in y_predicted]
    else:
        predictions = [round(x) for x in y_predicted]
    examples_len = len(y_true)
    correct = sum([y1 == y2 for y1, y2 in zip(y_true, predictions)])
    return correct / examples_len if examples_len else 0
