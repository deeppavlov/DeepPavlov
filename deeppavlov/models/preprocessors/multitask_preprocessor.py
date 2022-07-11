from typing import Iterable, Union, List,Tuple
from logging import getLogger

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.models.preprocessors.torch_transformers_preprocessor import *

log = getLogger(__name__)





from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from deeppavlov.core.common.registry import register
from logging import getLogger

log = getLogger(__name__)



@register("multitask_input_splitter")
class MultiTaskInputSplitter:
    """The instance of these class in pipe splits a batch of sequences of identical length or dictionaries with 
    identical keys into tuple of batches.

    Args:
        keys_to_extract: a sequence of ints or strings that have to match keys of split dictionaries.
    """

    def __init__(self, keys_to_extract: Union[List[str], Tuple[str, ...]], **kwargs):
        self.keys_to_extract = keys_to_extract

    def __call__(self, inp: Union[List[dict], List[List[int]], List[Tuple[int]]]) -> Union[List[list], List[str]]:
        """Returns batches of values from ``inp``. Every batch contains values that have same key from 
        ``keys_to_extract`` attribute. The order of elements of ``keys_to_extract`` is preserved.

        Args:
            inp: A sequence of dictionaries with identical keys

        Returns:
            A list of lists of values of dictionaries from ``inp``
        """
        if all([isinstance(k, str) for k in inp]):
            log.warning('You want to split an input that is already string')
            return inp

        extracted = [[] for _ in self.keys_to_extract]
        for item in inp:
            for i, key in enumerate(self.keys_to_extract):
                if key < len(item):
                    extracted[i].append(item[key])
        return extracted

@register('multitask_pipeline_preprocessor')
class MultiTaskPipelinePreprocessor(Component):
    """
    Extracts out the task_id from the first index of each example for each task.
    Then splits the input and performs tokenization
    """

    def __init__(self,
                 vocab_file,
                 do_lower_case: bool = True,
                 preprocessor: str='TorchTransformersPreprocessor',
                 preprocessors=None, 
                 max_seq_length: int = 512, 
                 return_tokens: bool = False, 
                 strict=False,
                 n_task: int = 3,
                 *args, **kwargs):
        self.n_task = n_task
        self.strict=strict
        if preprocessors is None:
            log.info(f'Assuming the same preprocessor name for all : {preprocessor}')
            assert preprocessor is not None
            preprocessor = eval(preprocessor)
            self.preprocessors=[preprocessor(vocab_file, do_lower_case, max_seq_length,*args,**kwargs)
                                for _ in range(self.n_task)]
        else:
            assert len(preprocessors) == self.n_task
            for i in range(len(preprocessors)):
                preprocessors[i] = eval(preprocessors[i]) 
            self.preprocessors = [preprocessors[i](vocab_file, do_lower_case, max_seq_length,*args,**kwargs)
                                  for i in range(self.n_task)]

    def split(self, features):
        if all([isinstance(k,str) for k in features]) or all([k==None for k in features]):
            # single sentence classification
            log.debug('Assuming single sentence classification')
            texts_a, texts_b = features, None
        elif all([isinstance(k,tuple) and len(k)==2 for k in features]):
            log.debug('Assuming sentence pair classification or classification for multichoice')
            texts_a,texts_b = [], []
            for feature in features:
                text_a,text_b=feature
                texts_a.append(text_a)
                texts_b.append(text_b)
        elif all([isinstance(k,list) for k in features]):
            log.debug('Assuming ner classification')
            texts_a, texts_b = list(features),None
        else:
            if self.strict:
                raise Exception(f'Unsupported task data {features}')
            else:
                log.warning('Data not split.Going without splitting')
                texts_a,texts_b = features, None
        return texts_a, texts_b
    def __call__(self, *args):
        """Returns batches of values from ``inp``. Every batch contains values that have same key from 
        ``keys_to_extract`` attribute. The order of elements of ``keys_to_extract`` is preserved.

        Args:
            inp: A sequence of dictionaries with identical keys

        Returns:
            A list of lists of values of dictionaries from ``inp``
        """
        assert len(args) == self.n_task, f"Seen examples from {len(args)} tasks but n_task specified to {self.n_task}"
        answer = []
        for i in range(len(args)):
            if all([j== None for j in args[i]]):
                log.debug('All nones received')
                answer.append([])
            else:
                texts_a, texts_b = self.split(args[i])
                log.debug(f'Preprocessor {self.preprocessors[i]}')
                if all([j==None for j in texts_a]):
                    log.debug('All nones')
                    answer.append([])
                else:
                    answer.append(self.preprocessors[i](texts_a, texts_b))
        assert answer != [[]], 'Empty answer'
        return answer

