import random

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset import Dataset


@register('typos_dataset')
class TyposDataset(Dataset):
    def split(self, *args, **kwargs):
        """Split all data into train and test

        Args:
            test_ratio (float): ratio of test data to train, from 0. to 1. Defaults to 0.15
        """
        self.train += self.valid + self.test

        test_ratio = args[0] if args else kwargs.get('test_ratio', 0.15)

        split = int(len(self.train) * test_ratio)

        rs = random.getstate()
        random.setstate(self.random_state)
        random.shuffle(self.train)
        self.random_state = random.getstate()
        random.setstate(rs)

        self.test = self.train[:split]
        self.train = self.train[split:]
        self.valid = []
