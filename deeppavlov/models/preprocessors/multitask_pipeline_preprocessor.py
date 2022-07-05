from typing import Iterable, Union, List,Tuple
from logging import getLogger

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.models.preprocessors.input_splitter import InputSplitter
from deeppavlov.models.preprocessors.torch_transformers_preprocessor import *

log = getLogger(__name__)


@register('multitask_pipeline_preprocessor')
class MultitaskPipelinePreprocessor(Component):
    """
    Extracts out the task_id from the first index of each example for each task.
    Then splits the input and performs tokenization
    """

    def __init__(self, possible_keys_to_extract: Union[List[int], List[List[int]],
                 vocab_file: str, do_lower_case: bool = True,
                 preprocessor_name: str='TorchTransformerPreprocessor'
                 max_seq_length: int = 512, return_tokens: bool = False, *args, **kwargs):
        if isinstance(possible_keys_to_extract[0], int):
            self.input_splitters = [InputSplitter(keys_to_extract=possible_keys_to_extract)]
        else:
            self.input_splitters = [InputSplitter(keys_to_extract=keys) for keys in possible_keys_to_extract]
        self.id_extractor = MultitaskPreprocessor()
        self.preprocessor = eval(preprocessor_name)(vocab_file, do_lower_case, max_seq_length, return_tokens)

    def __call__(self, *args):
        """Returns batches of values from ``inp``. Every batch contains values that have same key from 
        ``keys_to_extract`` attribute. The order of elements of ``keys_to_extract`` is preserved.

        Args:
            inp: A sequence of dictionaries with identical keys

        Returns:
            A list of lists of values of dictionaries from ``inp``
        """
        # breakpoint()
        print('ARGS')
        print(args)
        self.id_extractor.n_task = len(args)
        values = self.id_extractor(*args)
        # breakpoint()
        for i in range(1, len(values)):
            try:
                texts_a, texts_b = self.input_splitters[1](values[i])
            except:
                texts_a = self.input_splitters[0](values[i])
                texts_b = None
            values[i] = self.preprocessor(texts_a, texts_b)
        print('ANSWERS ARE ')
        print(values)
        # breakpoint()
        return values

 
@register('multitask_preprocessor')
class MultitaskPreprocessor(Component):
    """
    Extracts out the task_id from the first index of each example for each task
    """

    def __init__(self, *args, **kwargs):
        try:
            self.n_task = len(kwargs["in"])
        except:
            self.n_task = None

    def __call__(self, *args):
        out = []
        final_id=(args[0][0][0])
        if self.n_task == 0:
            self.n_task = len(args)
        for task_no in range(self.n_task):
            examples = args[task_no]
            task_data = []
            for values in examples:
                if isinstance(values, Iterable):
                    task_id = task_no
                    if isinstance(task_id, int):
                        task_data.extend([*values[1:]])
                    else:
                        task_data.append(values)
                else:
                    pass
            if task_data:
                out.append(tuple(task_data))
        ans = [final_id, *out]
        return ans
