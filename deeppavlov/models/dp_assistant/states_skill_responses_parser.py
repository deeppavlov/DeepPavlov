# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, List

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('dialogs_skill_responses_parser')
class DialogsSkillResponsesParser(Component):
    def __init__(self, **kwargs):
        pass

    def __call__(self, dialogs: List[dict]) -> Tuple[List[str], List[dict], List[List[str]], List[List[dict]],
                                                     List[str], List[str], List[List[Tuple[str, str]]]]:
        utterances_histories = []
        last_utterances = []
        annotations_histories = []
        last_annotations = []
        dialog_ids = []
        user_ids = []
        possible_answers_batch = []

        for dialog in dialogs:
            utterances_history = []
            annotations_history = []
            possible_answers = []

            for utterance in dialog['utterances']:
                utterances_history.append(utterance['text'])
                annotations_history.append(utterance['annotations'])

            last_utterances.append(utterances_history[-1])
            utterances_histories.append(utterances_history)
            last_annotations = annotations_history[-1]
            annotations_histories.append(annotations_history)

            dialog_ids.append(dialog['id'])
            user_ids.append(dialog['user']['id'])

            for k, v in dialog['utterances'][-1]['selected_skills'].items():
                possible_answers.append((k, v['text']))
            possible_answers_batch.append(possible_answers)

        return last_utterances, last_annotations, utterances_histories, annotations_histories, dialog_ids, user_ids, \
               possible_answers_batch
