from typing import Iterable, Union, List,Tuple
from logging import getLogger

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.models.preprocessors.input_splitter import InputSplitter
from deeppavlov.models.preprocessors.torch_transformers_preprocessor import *

log = getLogger(__name__)


@register('multitask_pipeline_preprocessor')
class MultiTaskPipelinePreprocessor(Component):
    """
    Extracts out the task_id from the first index of each example for each task.
    Then splits the input and performs tokenization
    """

    def __init__(self, possible_keys_to_extract,
                 vocab_file,
                 do_lower_case: bool = True,
                 preprocessor_name: str='TorchTransformersPreprocessor',
                 preprocessor_names=None, 
                 max_seq_length: int = 512, 
                 return_tokens: bool = False, 
                 n_task: int = 2,
                 *args, **kwargs):
        self.n_task = n_task
        if isinstance(possible_keys_to_extract, str):
            log.info(f'Assuming {possible_keys_to_extract} can be casted to list or list of lists')
            possible_keys_to_extract = eval(possible_keys_to_extract)
        if not isinstance(possible_keys_to_extract[0], list):
            self.input_splitters = [InputSplitter(keys_to_extract=possible_keys_to_extract)]
        if isinstance(possible_keys_to_extract[0],list):
            assert len(possible_keys_to_extract) == n_task
            log.info(f'Utilizing many input splitters with sets {possible_keys_to_extract}')
            self.input_splitters = [InputSplitter(keys_to_extract=keys) for keys in possible_keys_to_extract]
        else:
            self.input_splitters = [InputSplitter(keys_to_extract=possible_keys_to_extract)
                                    for _ in range(self.n_task)]
        self.id_extractor = MultiTaskPreprocessor()
        if preprocessor_names is None:
            log.info(f'Assuming the same preprocessors name for all for all {preprocessor_name}')
            preprocessor_name = eval(preprocessor_name)
            self.preprocessors=[preprocessor_name(vocab_file, do_lower_case, max_seq_length)
                                for _ in range(self.n_task)]
        else:
            assert len(preprocessor_names) == self.n_task
            for i in range(len(preprocessor_names)):
                preprocessor_names[i] = eval(preprocessor_names[i]) 
            self.preprocessors = [preprocessor_names[i](vocab_file, do_lower_case, max_seq_length)
                                  for i in range(self.n_task)]


    def __call__(self, *args):
        """Returns batches of values from ``inp``. Every batch contains values that have same key from 
        ``keys_to_extract`` attribute. The order of elements of ``keys_to_extract`` is preserved.

        Args:
            inp: A sequence of dictionaries with identical keys

        Returns:
            A list of lists of values of dictionaries from ``inp``
        """
        #print('calling pipeline')
        #print(args)
        self.id_extractor.n_task = len(args)
        values = self.id_extractor(*args)
        #print('obtained')
        #print(values)
        task_id = values[0]
        all_task_data = values[1:]
        answer = [task_id]
        for i in range(len(all_task_data)):
            texts_a, texts_b = self.input_splitters[i](all_task_data[i])
            #input splitters to return None if not found
            assert texts_a is not None
            answer.append(self.preprocessors[i](texts_a, texts_b))
        #print('returned')
        #print(answer)
        return answer
        

 
@register('multitask_preprocessor')
class MultiTaskPreprocessor(Component):
    """
    Extracts out the task_id from the first index of each example for each task
    """

    def __init__(self, *args, **kwargs):
        try:
            self.n_task = len(kwargs["in"])
        except:
            self.n_task = None

    def __call__(self, *args):
        print('EXTRACTOR INPUT')
        print(args)
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
        print('EXTRACTOR OUTPUT')
        print(ans)
        return ans
