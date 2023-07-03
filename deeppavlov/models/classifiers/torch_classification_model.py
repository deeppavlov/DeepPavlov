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

import logging
from typing import List, Union, Optional

import numpy as np
import torch

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel
from .torch_nets import ShallowAndWideCnn

log = logging.getLogger(__name__)


@register('torch_text_classification_model')
class TorchTextClassificationModel(TorchModel):
    """Class implements torch model for classification of texts.
    Input can either be embedded tokenized texts OR indices of words in the vocabulary.
    Number of tokens is not fixed while the samples in batch should be padded to the same (e.g. longest) lengths.

    Args:
        n_classes: number of classes
        kernel_sizes_cnn: list of kernel sizes of convolutions
        filters_cnn: number of filters for convolutions
        dense_size: number of units for dense layer
        dropout_rate: dropout rate, after convolutions and between dense
        embedding_size: size of vector representation of words
        multilabel: is multi-label classification (if so, `sigmoid` activation will be used, otherwise, softmax)
        criterion: criterion name from `torch.nn`
        embedded_tokens: True, if input contains embedded tokenized texts;
                         False, if input containes indices of words in the vocabulary
        vocab_size: vocabulary size in case of `embedded_tokens=False`, and embedding is a layer in the Network
        return_probas: whether to return probabilities or index of classes (only for `multilabel=False`)

    Attributes:
        model: torch model itself
        epochs_done: number of epochs that were done
        criterion: torch criterion instance
    """

    def __init__(self, n_classes: int,
                 kernel_sizes_cnn: List[int],
                 filters_cnn: int,
                 dense_size: int,
                 dropout_rate: float = 0.0,
                 embedding_size: Optional[int] = None,
                 multilabel: bool = False,
                 criterion: str = "CrossEntropyLoss",
                 embedded_tokens: bool = True,
                 vocab_size: Optional[int] = None,
                 return_probas: bool = True,
                 **kwargs):

        if n_classes == 0:
            raise ConfigError("Please, provide vocabulary with considered classes or number of classes.")

        if multilabel and not return_probas:
            raise RuntimeError('Set return_probas to True for multilabel classification!')

        self.multilabel = multilabel
        self.return_probas = return_probas
        model = ShallowAndWideCnn(
            n_classes=n_classes, embedding_size=embedding_size,
            kernel_sizes_cnn=kernel_sizes_cnn, filters_cnn=filters_cnn,
            dense_size=dense_size, dropout_rate=dropout_rate,
            embedded_tokens=embedded_tokens,
            vocab_size=vocab_size
        )
        self.criterion = getattr(torch.nn, criterion)()
        super().__init__(model, **kwargs)

    def __call__(self, texts: List[np.ndarray], *args) -> Union[List[List[float]], List[int]]:
        """Infer on the given data.

        Args:
            texts: list of tokenized text samples
            labels: labels
            *args: additional arguments

        Returns:
            for each sentence:
                vector of probabilities to belong with each class
                or list of labels sentence belongs with
        """
        with torch.no_grad():
            features = np.array(texts)
            inputs = torch.from_numpy(features)
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            if self.multilabel:
                outputs = torch.nn.functional.sigmoid(outputs)
            else:
                outputs = torch.nn.functional.softmax(outputs, dim=-1)

        outputs = outputs.cpu().detach().numpy()
        if self.return_probas:
            return outputs.tolist()
        else:
            return np.argmax(outputs, axis=-1).tolist()

    def train_on_batch(self, texts: List[List[np.ndarray]], labels: list) -> Union[float, List[float]]:
        """Train the model on the given batch.

        Args:
            texts: vectorized texts
            labels: list of labels

        Returns:
            metrics values on the given batch
        """
        features, labels = np.array(texts), np.array(labels)

        inputs, labels = torch.from_numpy(features), torch.from_numpy(labels)
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.model(inputs)
        labels = labels.view(-1).long()
        loss = self.criterion(outputs, labels)
        self._make_step(loss)
        return loss.item()
