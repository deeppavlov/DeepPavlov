from overrides import overrides
from typing import Generator

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset import Dataset


@register('hcn_dataset')
class HCNDataset(Dataset):

    def __init__(self, data):
        super().__init__(data)

    @overrides
    def split(self, num_train_instances=200, num_test_instances=50):
        self.test = self.train[num_train_instances:num_train_instances+num_test_instances]
        self.train = self.train[:num_train_instances]

    @overrides
    def iter_all(self, data_type: str = 'train') -> Generator:
        data = self.data[data_type]
        for instance in data:
            yield instance


