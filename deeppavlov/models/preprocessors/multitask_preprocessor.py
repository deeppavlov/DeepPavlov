from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterable
from logging import getLogger

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.models.preprocessors.torch_transformers_preprocessor import *

log = getLogger(__name__)


@register("multitask_input_splitter")
class MultiTaskInputSplitter:
    """
    The instance of these class in pipe splits a batch of sequences of identical length or dictionaries with
    identical keys into tuple of batches.

    Args:
        keys_to_extract: a sequence of ints or strings that have to match keys of split dictionaries.
    """

    def __init__(self, keys_to_extract: Union[List[str], Tuple[str, ...]], **kwargs):
        self.keys_to_extract = keys_to_extract

    def __call__(self, inp: Union[List[dict], List[List[int]], List[Tuple[int]]]) -> Union[List[list], List[str]]:
        """
        Returns batches of values from ``inp``. Every batch contains values that have same key from
        ``keys_to_extract`` attribute. The order of elements of ``keys_to_extract`` is preserved.

        Args:
            inp: A sequence of dictionaries with identical keys

        Returns:
            A list of lists of values of dictionaries from ``inp``
        """
        if all([isinstance(k, str) for k in inp]):
            log.warning(f'You want to split an input that is already {type(inp[0])}')
            return inp

        extracted = [[] for _ in self.keys_to_extract]
        for item in inp:
            for i, key in enumerate(self.keys_to_extract):
                if item is not None:
                    if key < len(item):
                        extracted[i].append(item[key])
        return extracted


@register('multitask_pipeline_preprocessor')
class MultiTaskPipelinePreprocessor(Component):
    """
    Extracts out the task_id from the first index of each example for each task.
    Then splits the input and performs tokenization
    Params:
    
    vocab_file(str): vocabulary file for tokenization
    do_lower_case(bool): if True, tokenization is lower-cased. Default: True
    preprocessor(str): name of DeepPavlov class that is used for tokenization. 
    Default: TorchTransformersPreprocessor
    preprocessors(List[str]): list of names of DeepPavlov classes that are used for tokenization.
    Overrides preprocessor . The length of list must be equal to the number of tasks
    max_seq_length(int): Maximum sequence length for tokenizer. Default: 512
    strict(bool): if True, we always try to split data assuming predefined modes as in multitask_example.json  
    If False, we go without splitting if we are not sure how to split the data. Default, False
    """

    def __init__(self,
                 vocab_file,
                 do_lower_case: bool = True,
                 preprocessor: str = 'TorchTransformersPreprocessor',
                 preprocessors: List[str] = None,
                 max_seq_length: int = 512,
                 strict=False,
                 *args, **kwargs):
        self.n_task = n_task
        self.strict = strict
        if preprocessors is None:
            log.info(
                f'Assuming the same preprocessor name for all : {preprocessor}')
            self.preprocessor = eval(preprocessor)
            if self.n_task is not None:
                self.preprocessors = [self.preprocessor(vocab_file, do_lower_case, max_seq_length, *args, **kwargs)
                                      for _ in range(self.n_task)]
            else:
                self.preprocessors = None
        else:
            for i in range(len(preprocessors)):
                preprocessors[i] = eval(preprocessors[i])
            self.preprocessors = [preprocessors[i](vocab_file, do_lower_case, max_seq_length, *args, **kwargs)
                                  for i in range(len(preprocessors))]
            self.n_task = len(preprocessors)

    def split(self, features):
        if all([isinstance(k, str) for k in features]) or all([k is None for k in features]):
            # single sentence classification
            log.debug('Assuming single sentence classification')
            texts_a, texts_b = features, None
        elif all([isinstance(k, tuple) and len(k) == 2 for k in features]):
            log.debug(
                'Assuming sentence pair classification or classification for multichoice')
            texts_a, texts_b = [], []
            for feature in features:
                text_a, text_b = feature
                texts_a.append(text_a)
                texts_b.append(text_b)
        elif all([isinstance(k, list) for k in features]):
            log.debug('Assuming ner classification')
            texts_a, texts_b = list(features), None
        else:
            if self.strict:
                raise Exception(f'Unsupported task data {features}')
            else:
                log.warning('Data not split.Going without splitting')
                texts_a, texts_b = features, None
        return texts_a, texts_b

    def __call__(self, *args):
        """
        Returns batches of values from ``inp``. Every batch contains values that have same key from
        ``keys_to_extract`` attribute. The order of elements of ``keys_to_extract`` is preserved.

        Args:
            inp: A sequence of dictionaries with identical keys

        Returns:
            A list of lists of values of dictionaries from ``inp``
        """
        self.n_task = len(args)
        if self.preprocessors is None:
             # Defining preprocessor list while we call the function, as only he
             self.preprocessors = [self.preprocessor(vocab_file, do_lower_case,
                                                     max_seq_length, *args, **kwargs)
                                   for _ in range(self.n_task)]
        answer = []
        for i in range(len(args)):
            if all([j is None for j in args[i]]):
                log.debug('All nones received')
                answer.append([])
            else:
                texts_a, texts_b = self.split(args[i])
                log.debug(f'Preprocessor {self.preprocessors[i]}')
                if all([j is None for j in texts_a]):
                    log.debug('All nones')
                    answer.append([])
                else:
                    if 'choice' in str(self.preprocessors[i]):
                        if isinstance(texts_a[0], str) and isinstance(texts_b[0], list):
                            # transform multiple choice to format suitable for preprocessor
                            texts_a = [[text for _ in range(len(texts_b[0]))] for text in texts_a]
                    answer.append(self.preprocessors[i](texts_a, texts_b))
        assert answer != [[]], 'Empty answer'
        return answer
