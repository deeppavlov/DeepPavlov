from pathlib import Path
from logging import exception, getLogger
from typing import List, Optional
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, BatchEncoding

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)

@register('dnnc_input_preprocessor')
class InputPreprocessor(Component):
    def __init__(self,
                 support_dataset_path: str,
                 format: str = "csv",
                 *args, **kwargs) -> None:
        file = Path(support_dataset_path).expanduser()
        if file.exists():
            if format == 'csv':
                keys = ('sep', 'header', 'names')
                options = {k: kwargs[k] for k in keys if k in kwargs}
                df = pd.read_csv(file, **options)
            elif format == 'json':
                keys = ('orient', 'lines')
                options = {k: kwargs[k] for k in keys if k in kwargs}
                df = pd.read_json(file, **options)
            else:
                raise Exception('Unsupported file format: {}'.format(format))

            x = kwargs.get("x", "text")
            y = kwargs.get('y', 'labels')
            
            self.support_dataset = [(row[x], str(row[y])) for _, row in df.iterrows()]
        else:
            self.support_dataset = None
            log.warning("Cannot find {} file".format(support_dataset_path))
    
    def __call__(self,
                 input_texts : List[str]) -> List[List[str]]:
        '''
            Generates all possible ordread pairs from 'input_texts' and 'self.support_dataset'
        '''
        if self.support_dataset:
            hypotesis_batch = []
            premise_batch = []
            hypotesis_labels_batch = []

            for [premise, [hypotesis, hypotesis_labels]] in zip(input_texts * len(self.support_dataset),
                                                                np.repeat(self.support_dataset, len(input_texts), axis=0)):
   
                premise_batch.append(premise)
                hypotesis_batch.append(hypotesis)
                hypotesis_labels_batch.append(hypotesis_labels)

            return hypotesis_batch, premise_batch, hypotesis_labels_batch
        else:
            log.warning("Error: no support dataset")