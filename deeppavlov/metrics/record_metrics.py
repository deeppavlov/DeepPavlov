import re
import string
import collections
from typing import List

import numpy as np

from deeppavlov.models.preprocessors.torch_transformers_preprocessor import RecordNestedExample
from deeppavlov.core.common.metrics_registry import register_metric


@register_metric("record_f1_score")
def record_f1_score(record_examples: List[RecordNestedExample]):
    """Calculate F1 score for given nested ReCoRD examples

    Args:
        record_examples: processed ReCoRD examples

    Returns:
        float: F1 score
    """
    if not record_examples:
        return 0.
    f1_scores = []
    for example in record_examples:
        example_f1s = []
        for answer in example.answers:
            example_f1s.append(exact_match_score(example.prediction, answer))
        f1_scores.append(max(example_f1s))
    return np.mean(f1_scores)


@register_metric("record_em_score")
def record_em_score(record_examples: List[RecordNestedExample]):
    """Calculate Exact Match score for given nested ReCoRD examples

    Args:
        record_examples: processed ReCoRD examples

    Returns:
        float: Exact Match score
    """
    if not record_examples:
        return 0.
    em_scores = []
    for example in record_examples:
        example_ems = []
        for answer in example.answers:
            example_ems.append(string_f1_score(example.prediction, answer))
        em_scores.append(max(example_ems))
    return np.mean(em_scores)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace.
    From official ReCoRD eval script
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def string_f1_score(prediction, ground_truth):
    """Compute normalized token level F1
    From official ReCoRD eval script
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """Compute normalized exact match
    From official ReCoRD eval script
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)
