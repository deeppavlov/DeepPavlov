from typing import Tuple, List

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('states_parser')
class StatesParser(Component):
    def __init__(self, **kwargs):
        pass

    def __call__(self, states: dict) -> Tuple[List[str], List[dict], List[List[str]], List[List[dict]],
                                              List[str], List[str]]:
        utterances_histories = []
        last_utterances = []
        annotations_histories = []
        last_annotations = []
        dialog_ids = []
        user_ids = []

        for state in states:
            utterances_history = []
            annotations_history = []
            for utterance in state['utterances']:
                utterances_history.append(utterance['text'])
                annotations_history.append(utterance['annotations'])

            last_utterances.append(utterances_history[-1])
            utterances_histories.append(utterances_history)
            last_annotations = annotations_history[-1]
            annotations_histories.append(annotations_history)

            dialog_ids.append(state['id'])
            user_ids.append(state['user']['id'])

        return last_utterances, last_annotations, utterances_histories, annotations_histories, dialog_ids, user_ids
