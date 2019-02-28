from typing import Tuple, List

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('dialogs_parser')
class DialogsParser(Component):
    def __init__(self, **kwargs):
        pass

    def __call__(self, dialogs: List[dict]) -> Tuple[List[str], List[dict], List[List[str]], List[List[dict]],
                                                     List[str], List[str]]:
        utterances_histories = []
        last_utterances = []
        annotations_histories = []
        last_annotations = []
        dialog_ids = []
        user_ids = []

        for dialog in dialogs:
            utterances_history = []
            annotations_history = []
            for utterance in dialog['utterances']:
                utterances_history.append(utterance['text'])
                annotations_history.append(utterance['annotations'])

            last_utterances.append(utterances_history[-1])
            utterances_histories.append(utterances_history)
            last_annotations = annotations_history[-1]
            annotations_histories.append(annotations_history)

            dialog_ids.append(dialog['id'])
            user_ids.append(dialog['user']['id'])

        return last_utterances, last_annotations, utterances_histories, annotations_histories, dialog_ids, user_ids
