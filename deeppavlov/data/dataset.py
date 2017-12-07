from typing import List, Dict


class Dataset:

    def __init__(self, data: Dict[str, List]) -> None:
        r""" Dataset takes a dict with fields 'train', 'test', 'valid'. A list of samples (pairs x, y) is stored
        in each field.

        Args:
            data: list of (x, y) pairs. Each pair is a sample from the dataset. x as well as y can be a tuple
                of different input features.
        """
        raise NotImplementedError

    def batch_generator(self, batch_size: int, data_type: str):
        r"""This function returns a generator, which serves for generation of raw (no preprocessing such as tokenization)
         batches

        Args:
            batch_size: number of samples in batch
            data_type: can be either 'train', 'test', or 'valid'

        Returns:
            batch_gen: a generator, that iterates through the part (defined by data_type) of dataset
        """
        raise NotImplementedError('Batch generator must be implemented for Dataset class!')

    def iter_all(self):
        r"""Iterate through all data. It can be used for building dictionary or

        Returns:
            samples_gen: a generator, that iterates through the all samples in the 'train' part of dataset

        """
        raise NotImplementedError
