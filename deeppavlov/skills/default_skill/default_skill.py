from typing import Tuple

from deeppavlov.core.models.component import Component
from deeppavlov.core.skill.skill import Skill


class DefaultStatelessSkill(Skill):
    """Default stateless skill class.

    The class is intended to be used for as a default skill wrapping
    DeepPavlov models.

    Attributes:
        model: DeepPavlov model to be wrapped into default skill instance.
    """
    def __init__(self, model: Component):
        self.model: Component = model

    def __call__(self, utterances_batch: [list, tuple], history_batch: [list, tuple],
                 states_batch: [list, tuple] = None) -> Tuple[list, list, list]:
        # TODO: methods inputs should be lists, not tuples
        """Returns skill inference result.

        Returns batches of skill inference results, estimated confidence
            levels and up to date states corresponding to incoming utterance
            batch. Also handles interaction with multiargument models using
            skill states.

        Args:
            utterances_batch: A batch of utterances of any type.
            history_batch: Not used. A batch of list typed histories for each
                utterance.
            states_batch: A batch of states for each utterance.

        Returns:
            response: A batch of arbitrary typed skill inference results.
            confidence: A batch of float typed confidence levels for each of
                skill inference result.
            states: Optional. A batch of states for each response.
        """
        utterances_batch = list(utterances_batch)
        states_batch = list(states_batch)

        batch_len = len(utterances_batch)
        confidence_batch = [1.0] * batch_len

        if len(self.model.in_x) > 1:
            response_batch = [None] * batch_len
            infer_indexes = []

            if not states_batch:
                states_batch = [None] * batch_len

            for utt_i, utterance in enumerate(utterances_batch):
                if not states_batch[utt_i]:
                    states_batch[utt_i] = {'expected_args': list(self.model.in_x), 'received_values': []}

                states_batch[utt_i]['expected_args'].pop(0)
                states_batch[utt_i]['received_values'].append(utterance)

                if states_batch[utt_i]['expected_args']:
                    response = 'expecting_arg:{}'.format(states_batch[utt_i]['expected_args'][0])
                    response_batch[utt_i] = response
                else:
                    infer_indexes.append(utt_i)

            if infer_indexes:
                infer_utterances = [tuple(states_batch[i]['received_values']) for i in infer_indexes]
                infer_results = self.model(infer_utterances)

                for infer_i, infer_result in zip(infer_indexes, infer_results):
                    response_batch[infer_i] = infer_result
                    states_batch[infer_i] = None
        else:
            response_batch = self.model(utterances_batch)

        return response_batch, confidence_batch, states_batch
