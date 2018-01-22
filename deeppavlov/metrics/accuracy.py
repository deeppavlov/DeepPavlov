from .registry import register_metric


@register_metric('accuracy')
def accuracy(y_true, y_predicted):
    examples_len = len(y_true)
    correct = sum([int(y1 == y2) for y1, y2 in zip(y_true, y_predicted)])
    return correct / examples_len if examples_len else 0
