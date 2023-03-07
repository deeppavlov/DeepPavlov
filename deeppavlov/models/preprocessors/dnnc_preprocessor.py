from pathlib import Path
from logging import exception, getLogger
from typing import List, Optional, Tuple
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
                 support_dataset_path: str = None,
                 bidirectional: bool = False,
                 *args, **kwargs) -> None:
        self.bidirectional = bidirectional
        if support_dataset_path:
            file = Path(support_dataset_path).expanduser()
            if file.exists():
                df = pd.read_json(file, orient='split')
                self.support_dataset = [(row["text"], str(row["category"])) for _, row in df.iterrows()]
            else:
                log.error(f"Cannot find {support_dataset_path} file")
                self.support_dataset = None

    def __call__(self, input) -> List[List[str]]:
        '''
            Generates all possible ordread pairs from 'input_texts' and 'self.support_dataset'
        '''

        if len(input) <= 1 or isinstance(input[1], str):
            texts = input
        else:
            texts, self.support_dataset = input

        hypotesis_batch = []
        premise_batch = []
        hypotesis_labels_batch = []
        for [premise, [hypotesis, hypotesis_labels]] in zip(texts * len(self.support_dataset),
                                                            np.repeat(self.support_dataset, len(texts), axis=0)):
            premise_batch.append(premise)
            hypotesis_batch.append(hypotesis)
            hypotesis_labels_batch.append(hypotesis_labels)

            if self.bidirectional:
                premise_batch.append(hypotesis)
                hypotesis_batch.append(premise)
                hypotesis_labels_batch.append(hypotesis_labels)
        return texts, hypotesis_batch, premise_batch, hypotesis_labels_batch
        