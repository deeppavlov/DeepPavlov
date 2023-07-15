from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('ppl')
def ppl(ppl_list):
    return sum(ppl_list) / len(ppl_list)
