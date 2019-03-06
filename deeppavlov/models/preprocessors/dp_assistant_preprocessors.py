from typing import List

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('keras_probas_converter')
class KerasClassificationProbasConverter(Component):

    def __init__(self, **kwargs):
        pass

    def __call__(self, preds: List[List[float]]) -> List[str]:
        skill_names = []
        for pred in preds:
            if pred[0] > pred[1]:
                skill_names.append('chitchat')
            else:
                skill_names.append('odqa')
        return skill_names
