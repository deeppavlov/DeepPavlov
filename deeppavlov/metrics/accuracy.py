from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('accuracy')
def accuracy(y_true, y_predicted):
    examples_len = len(y_true)
    correct = sum([int(y1 == y2) for y1, y2 in zip(y_true, y_predicted)])
    return correct / examples_len if examples_len else 0


@register_metric('sets_accuracy')
def sets_accuracy(y_true, y_predicted):
    examples_len = len(y_true)
    correct = sum([int(set(y1) == set(y2)) for y1, y2 in zip(y_true, y_predicted)])
    return correct / examples_len if examples_len else 0
