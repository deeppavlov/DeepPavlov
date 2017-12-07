import random
from typing import List, Dict, Generator, Tuple, Any


class Dataset:
    def split(self, *args, **kwargs):
        pass

    def __init__(self, data: Dict[str, List[Tuple[Any, Any]]], *args, **kwargs) -> None:
        r""" Dataset takes a dict with fields 'train', 'test', 'valid'. A list of samples (pairs x, y) is stored
        in each field.

        Args:
            data: list of (x, y) pairs. Each pair is a sample from the dataset. x as well as y can be a tuple
                of different input features.
        """
        self.train = data.get('train', [])
        self.valid = data.get('valid', [])
        self.test = data.get('test', [])
        self.split(*args, **kwargs)
        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self.train + self.test + self.valid
        }

    def batch_generator(self, batch_size: int, data_type: str = 'train') -> Generator:
        r"""This function returns a generator, which serves for generation of raw (no preprocessing such as tokenization)
         batches

        Args:
            batch_size: number of samples in batch
            data_type: can be either 'train', 'test', or 'valid'

        Returns:
            batch_gen: a generator, that iterates through the part (defined by data_type) of dataset
        """
        data = self.data[data_type]
        data_len = len(data)
        order = list(range(data_len))
        random.shuffle(order)
        for i in range((data_len - 1) // batch_size + 1):
            yield list(zip(*[data[o] for o in order[i*batch_size:(i+1)*batch_size]]))

    def iter_all(self, data_type: str = 'train') -> Generator:
        r"""Iterate through all data. It can be used for building dictionary or

        Returns:
            samples_gen: a generator, that iterates through the all samples in the 'train' part of dataset

        """
        data = self.data[data_type]
        for x, y in data:
            yield (x, y)
