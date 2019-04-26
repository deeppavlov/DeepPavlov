from typing import Sequence

from deeppavlov.core.agent_v2.state_schema import Dialog
from deeppavlov.models.preprocessors.person_normalizer import PersonNormalizer
from deeppavlov.models.tokenizers.utils import detokenize


class DefaultPostprocessor:
    def __init__(self) -> None:
        self.person_normalizer = PersonNormalizer(per_tag='PER')

    def __call__(self, dialogs: Sequence[Dialog]) -> Sequence[str]:
        new_responses = []
        for d in dialogs:
            # get tokens & tags
            response = d['utterances'][-1]
            ner_annotations = response['annotations']['ner']
            user_name = d['user']['profile']['name']
            # replace names with user name
            if ner_annotations and (response['active_skill'] == 'chitchat'):
                response_toks_norm, _ = \
                    self.person_normalizer([ner_annotations['tokens']],
                                           [ner_annotations['tags']],
                                           [user_name])
                response_toks_norm = response_toks_norm[0]
                # detokenize
                new_responses.append(detokenize(response_toks_norm))
            else:
                new_responses.append(response['text'])
        return new_responses
