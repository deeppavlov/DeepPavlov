import itertools
from nltk.translate.bleu_score import sentence_bleu

from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('bleu')
def bleu(y_true, y_predicted):
    examples_len = len(y_true)
    bleu_list = (sentence_bleu([y2.lower().split()], y1.lower().split())\
                 for y1, y2 in zip(y_true, y_predicted))
    return sum(bleu_list) / examples_len if examples_len else 0.

@register_metric('per_item_bleu')
def per_item_bleu(y_true, y_predicted):
    if isinstance(y_true[0], (tuple, list)):
        y_true = map(lambda y: y[0], y_true)
    y_true = list(itertools.chain(*y_true))
    y_predicted = itertools.chain(*y_predicted)
    examples_len = len(y_true)
    bleu_list = (sentence_bleu([y2.lower().split()], y1.lower().split())\
                 for y1, y2 in zip(y_true, y_predicted))
    return sum(bleu_list) / examples_len if examples_len else 0.

@register_metric('per_item_dialog_bleu')
def per_item_dialog_bleu(y_true, y_predicted):
    y_true = [y['text'] for dialog in y_true for y in dialog]
    y_predicted = itertools.chain(*y_predicted)
    examples_len = len(y_true)
    bleu_list = (sentence_bleu([y2.lower().split()], y1.lower().split())\
                 for y1, y2 in zip(y_true, y_predicted))
    return sum(bleu_list) / examples_len if examples_len else 0.

