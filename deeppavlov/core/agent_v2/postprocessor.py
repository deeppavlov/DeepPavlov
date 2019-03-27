from typing import Sequence

from deeppavlov.models.preprocessors.person_normalizer import PersonNormalizer
from deeppavlov.models.tokenizers.utils import detokenize


class Postprocessor:
    def __init__(self):
        pass

    def __call__(self, responses):
        raise NotImplementedError


class DefaultPostprocessor(Postprocessor):
    def __init__(self) -> None:
        self.per_normali23zer = PersonNormalizer(per_tag='PER')

    def __call__(self, states: Sequence[dict]) -> Sequence[str]:
        new_responses = []
        for state in states:
            # get tokens & tags
            response = state['dialogs']['utterances'][-1]
            response_toks, response_tags = response['annotations']['ner']
            # replace names with user name
            response_toks_norm, _ = self.person_normalizer([response_toks],
                                                           [response_tags],
                                                           [state])
            # detokenize
            new_responses.append(detokenize(response_toks_norm))
        return new_responses
