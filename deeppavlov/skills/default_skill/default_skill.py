from deeppavlov.core.models.component import Component
from deeppavlov.core.skill.skill import Skill

EXPECTING_ARG_MESSAGE = 'expecting_arg:{}'


class DefaultStatelessSkill(Skill):
    def __init__(self, model: Component):
        self.model = model

    def __call__(self, utterances_batch: [list, tuple], history_batch: [list, tuple],
                 states_batch: [list, tuple] = None):
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
                    response = EXPECTING_ARG_MESSAGE.format(states_batch[utt_i]['expected_args'][0])
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

        return response_batch, confidence_batch, history_batch, states_batch


if __name__ == '__main__':
    from pprint import pprint
    from deeppavlov.core.common.file import read_json
    from deeppavlov.core.commands.infer import build_model_from_config

    #config = '/home/litinsky/repo/DeepPavlov/deeppavlov/configs/ner/ner_rus.json'
    config = '/home/litinsky/repo/DeepPavlov/deeppavlov/configs/squad/squad.json'

    ner_infer = ['Элон Маск запустил в космос Tesla.', 'В Долгопе у меня отжали мобильник и новые кеды.']
    squad_infer_1 = ['Elon Musk launched his cherry Tesla roadster to the Mars orbit',
                     'I want Doshirac with mayonnaise']

    squad_infer_2 = ['Where cherry Tesla roadster was launched?',
                     'What I want?']



    model = build_model_from_config(read_json(config))
    skill = DefaultStatelessSkill(model)

    result1 = skill(squad_infer_1, [], [])
    pprint(result1)

    result2 = skill(squad_infer_2, [], result1[3])
    pprint(result2)