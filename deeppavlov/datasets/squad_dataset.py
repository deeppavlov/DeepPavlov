import random

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset import Dataset


@register('squad_dataset')
class SquadDataset(Dataset):
    def __init__(self, data, seed, shuffle=False):
        self.raw_data = data
        self.seed = seed
        self.shuffle = shuffle

        self.train = []
        self.valid = []
        self.test = []

        self.data = dict()

        rs = random.getstate()
        random.seed(seed)
        self.random_state = random.getstate()
        random.setstate(rs)

        for dt in ['train', 'valid']:
            self.data[dt] = SquadDataset._extract_cqas(self.raw_data[dt])

        # there is no public test set for SQuAD
        self.data['test'] = SquadDataset._extract_cqas(self.raw_data['valid'])

    @staticmethod
    def _extract_cqas(data):
        """Extracts context, question, answer, answer_start from SQuAD data

        :param data: data in squad format
        :return: list of (context, question), (answer_text, answer_start)
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
