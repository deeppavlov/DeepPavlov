from deeppavlov.core.common.metrics_registry import register_metric
from deeppavlov.models.ner.evaluation import precision_recall_f1
from itertools import chain


@register_metric('ner_f1')
def ner_f1(y_true, y_predicted):
    _, y_predicted = zip(*y_predicted)
    y_true = list(chain(*y_true))
    y_predicted = list(chain(*y_predicted))
    results = precision_recall_f1(y_true,
                                  y_predicted,
                                  print_results=False)
    f1 = results['__total__']['f1']
    return f1
