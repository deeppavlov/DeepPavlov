import itertools
from nltk.translate.bleu_score import sentence_bleu
from deeppavlov.metrics.google_bleu import compute_bleu

from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('bleu')
def bleu(y_true, y_predicted):
    examples_len = len(y_true)
    bleu_list = (sentence_bleu([y2.lower().split()], y1.lower().split())
                 for y1, y2 in zip(y_true, y_predicted))
    return sum(bleu_list) / examples_len if examples_len else 0.


@register_metric('per_item_bleu')
def per_item_bleu(y_true, y_predicted):
    if 0:
        print("y_true = {}".format(y_true[:2]))
        print("y_predicted = {}".format(y_predicted[:2]))
    if isinstance(y_true[0], (tuple, list)):
        y_true = [y[0] for y in y_true]
    if 0:
        y_predicted = list(y_predicted)
        print("y_true = {}".format(y_true[:2]))
        print("y_predicted = {}".format(y_predicted[:2]))
    examples_len = len(y_true)
    bleu_list = (sentence_bleu([y2.lower().split()], y1.lower().split())
                 for y1, y2 in zip(y_true, y_predicted))
    return sum(bleu_list) / examples_len if examples_len else 0.


@register_metric('per_item_google_bleu')
def per_item_google_bleu(y_true, y_predicted):
    if isinstance(y_true[0], (tuple, list)):
        y_true = (y[0] for y in y_true)
    return compute_bleu(([y.lower().split()] for y in y_predicted),
                        (y.lower().split() for y in y_true))[0]


@register_metric('per_item_dialog_bleu')
def per_item_dialog_bleu(y_true, y_predicted):
    y_true = [y['text'] for dialog in y_true for y in dialog]
    examples_len = len(y_true)
    bleu_list = (sentence_bleu([y2.lower().split()], y1.lower().split())
                 for y1, y2 in zip(y_true, y_predicted))
    return sum(bleu_list) / examples_len if examples_len else 0.

