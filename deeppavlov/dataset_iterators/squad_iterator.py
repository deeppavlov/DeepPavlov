from random import Random

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_iterator import BasicDatasetIterator


@register('squad_iterator')
class SquadIterator(BasicDatasetIterator):
    def __init__(self, data, seed, shuffle=False):
        self.raw_data = data
        self.seed = seed
        self.shuffle = shuffle

        self.train = []
        self.valid = []
        self.test = []

        self.data = dict()

        self.random = Random(seed)

        for dt in ['train', 'valid']:
            self.data[dt] = SquadIterator._extract_cqas(self.raw_data[dt])

        # there is no public test set for SQuAD
        self.data['test'] = SquadIterator._extract_cqas(self.raw_data['valid'])

    @staticmethod
    def _extract_cqas(data):
        """ Extracts context, question, answer, answer_start from SQuAD data

        Args:
            data: data in squad format

        Returns:
            list of (context, question), (answer_text, answer_start)
            answer text and answer_start are lists

        """
        cqas = []
        for article in data['data']:
            for par in article['paragraphs']:
                context = par['context']
                for qa in par['qas']:
                    q = qa['question']
                    ans_text = []
                    ans_start = []
                    for answer in qa['answers']:
                        ans_text.append(answer['text'])
                        ans_start.append(answer['answer_start'])
                    cqas.append(((context, q), (ans_text, ans_start)))
        return cqas
