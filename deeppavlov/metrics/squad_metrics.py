import re
import string
from collections import Counter

from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('exact_match')
def exact_match(y_true, y_predicted):
    y_predicted = hotfix(y_predicted)
    examples_len = len(y_true)
    correct = sum([normalize_answer(y1) == normalize_answer(y2) for (y1, _), (y2, _) in zip(y_true, y_predicted)])
    return 100 * correct / examples_len if examples_len else 0


@register_metric('squad_f1')
def squad_f1(y_true, y_predicted):
    y_predicted = hotfix(y_predicted)
    f1 = 0.0
    for (ground_truth, _), (prediction, _) in zip(y_true, y_predicted):
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 += (2 * precision * recall) / (precision + recall)
    return 100 * f1 / len(y_true) if len(y_true) > 0 else 0


def hotfix(y_predicted):
    """
    hotfix:
    [[arg1, arg1, ...], [arg2, arg2, ...], [arg1, arg1, ...], [arg2, arg2, ...], ...]
    ->
    [(arg1, arg2), (arg1, arg2), ...]
    """
    assert len(y_predicted) % 2 == 0
    a, p = [], []
    for i in range(len(y_predicted) // 2):
        a.extend(y_predicted[i * 2])
        p.extend(y_predicted[i * 2 + 1])

    return list(zip(a, p))

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
