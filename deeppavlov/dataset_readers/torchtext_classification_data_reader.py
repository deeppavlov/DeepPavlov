# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from logging import getLogger
import random
import numpy as np
from overrides import overrides

import torch
import torchtext
import torchtext.datasets as torch_texts
from torch.utils.data.dataset import random_split

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader

log = getLogger(__name__)


def _init_fn():
    np.random.seed(12)
    torch.manual_seed(12)
    torch.cuda.manual_seed(12)


@register("torchtext_classifiction_data_reader")
class TorchtextClassificationDataReader(DatasetReader):
    """Class initializes datasets as a key of `torchtext.datasets.DATASETS` or
        as an attribute of `torchtext.datasets`
    """
    @overrides
    def read(self, dataset_title: str, data_path: str, batch_size: int, valid_portion=0.05,
             vocab=None, ngrams=1, split_seed=42, tokenize="spacy", *args, **kwargs) -> dict:

        if dataset_title in torch_texts.DATASETS:
            log.info(f"Dataset {dataset_title} is used as a key of `torchtext.datasets.DATASETS`.")
            train_dataset, test_dataset = torch_texts.DATASETS[dataset_title](
                root=data_path, ngrams=ngrams, vocab=vocab)

            train_len = int(len(train_dataset) * (1. - valid_portion))
            train_dataset, valid_dataset = random_split(train_dataset, [train_len, len(train_dataset) - train_len])
        elif hasattr(torch_texts, dataset_title) and callable(getattr(torch_texts, dataset_title)):
            log.info(f"Dataset {dataset_title} is used as an attribute of `torchtext.datasets`.")
            _text = torchtext.data.Field(tokenize=tokenize)
            _label = torchtext.data.LabelField(dtype=torch.float)
            train_dataset, test_dataset = getattr(torch_texts, dataset_title).splits(_text, _label)
            train_dataset, valid_dataset = train_dataset.split(1 - valid_portion, random_state=random.seed(split_seed))
        else:
            raise NotImplementedError(f"Dataset {dataset_title} was not found.")

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, worker_init_fn=_init_fn(),
                                                  shuffle=True, num_workers=2)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, worker_init_fn=_init_fn(),
                                                  shuffle=False, num_workers=1)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        return {"train": trainloader, "valid": validloader, "test": testloader}
