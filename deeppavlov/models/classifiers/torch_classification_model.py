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
from overrides import overrides
from typing import List, Union, Optional

import numpy as np
import torch
import torch.nn as nn

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.models.torch_model import TorchModel
from deeppavlov.core.common.registry import register
from .torch_nets import ShallowAndWideCnn

log = logging.getLogger(__name__)


@register('torch_text_classification_model')
class TorchTextClassificationModel(TorchModel):
    """Class implements torch model for classification of texts.
    Input can either be embedded tokenized texts OR indices of words in the vocabulary.
    Number of tokens is not fixed while the samples in batch should be padded to the same (e.g. longest) lengths.

    Args:
        n_classes: number of classes
        model_name: name of `TorchTextClassificationModel` methods which initializes model architecture
        embedding_size: size of vector representation of words
        multilabel: is multi-label classification (if so, `sigmoid` activation will be used, otherwise, softmax)
        criterion: criterion name from `torch.nn`
        optimizer: optimizer name from `torch.optim`
        optimizer_parameters: dictionary with optimizer's parameters,
                              e.g. {'lr': 0.1, 'weight_decay': 0.001, 'momentum': 0.9}
        lr_scheduler: string name of scheduler class from `torch.optim.lr_scheduler`
        lr_scheduler_parameters: parameters for scheduler
        embedded_tokens: True, if input contains embedded tokenized texts;
                         False, if input containes indices of words in the vocabulary
        vocab_size: vocabulary size in case of `embedded_tokens=False`, and embedding is a layer in the Network
        lr_decay_every_n_epochs: how often to decay lr
        learning_rate_drop_patience: how many validations with no improvements to wait
        learning_rate_drop_div: the divider of the learning rate after `learning_rate_drop_patience` unsuccessful
            validations
        return_probas: whether to return probabilities or index of classes (only for `multilabel=False`)

    Attributes:
        opt: dictionary with all model parameters
        n_classes: number of considered classes
        model: torch model itself
        epochs_done: number of epochs that were done
        optimizer: torch optimizer instance
        criterion: torch criterion instance
    """

    def __init__(self, n_classes: int,
                 model_name: str,
                 embedding_size: Optional[int] = None,
                 multilabel: bool = False,
                 criterion: str = "CrossEntropyLoss",
                 optimizer: str = "AdamW",
                 optimizer_parameters: dict = {"lr": 0.1},
                 lr_scheduler: Optional[str] = None,
                 lr_scheduler_parameters: dict = {},
                 embedded_tokens: bool = True,
                 vocab_size: Optional[int] = None,
                 lr_decay_every_n_epochs: Optional[int] = None,
                 learning_rate_drop_patience: Optional[int] = None,
                 learning_rate_drop_div: Optional[float] = None,
                 return_probas: bool = True,
                 **kwargs):

        if n_classes == 0:
            raise ConfigError("Please, provide vocabulary with considered classes or number of classes.")

        if multilabel and not return_probas:
            raise RuntimeError('Set return_probas to True for multilabel classification!')

        super().__init__(
            embedding_size=embedding_size,
            n_classes=n_classes,
            model_name=model_name,
            optimizer=optimizer,
            criterion=criterion,
            multilabel=multilabel,
            optimizer_parameters=optimizer_parameters,
            embedded_tokens=embedded_tokens,
            vocab_size=vocab_size,
            lr_decay_every_n_epochs=lr_decay_every_n_epochs,
            learning_rate_drop_patience=learning_rate_drop_patience,
            learning_rate_drop_div=learning_rate_drop_div,
            lr_scheduler=lr_scheduler,
            lr_scheduler_parameters=lr_scheduler_parameters,
            return_probas=return_probas,
            **kwargs)

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
            if self.opt["multilabel"]:
                outputs = torch.nn.functional.sigmoid(outputs)
            else:
                outputs = torch.nn.functional.softmax(outputs, dim=-1)

        outputs = outputs.cpu().detach().numpy()
        if self.opt["return_probas"]:
            return outputs.tolist()
        else:
            return np.argmax(outputs, axis=-1).tolist()

    @overrides
    def process_event(self, event_name: str, data: dict):
        """Process event after epoch

        Args:
            event_name: whether event is send after epoch or batch.
                    Set of values: ``"after_epoch", "after_batch"``
            data: event data (dictionary)
        Returns:
            None
        """
        super().process_event(event_name, data)

        if event_name == "after_epoch" and self.opt.get("lr_decay_every_n_epochs", None) is not None:
            if self.epochs_done % self.opt["lr_decay_every_n_epochs"] == 0:
                log.info(f"----------Current LR is decreased in 10 times----------")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / 10

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
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return loss.item()

    def cnn_model(self, kernel_sizes_cnn: List[int], filters_cnn: int, dense_size: int, dropout_rate: float = 0.0,
                  **kwargs) -> nn.Module:
        """Build un-compiled model of shallow-and-wide CNN.

        Args:
            kernel_sizes_cnn: list of kernel sizes of convolutions.
            filters_cnn: number of filters for convolutions.
            dense_size: number of units for dense layer.
            dropout_rate: dropout rate, after convolutions and between dense.
            kwargs: other parameters

        Returns:
            torch.models.Model: instance of torch Model
        """
        model = ShallowAndWideCnn(n_classes=self.opt["n_classes"], embedding_size=self.opt["embedding_size"],
                                  kernel_sizes_cnn=kernel_sizes_cnn, filters_cnn=filters_cnn,
                                  dense_size=dense_size, dropout_rate=dropout_rate,
                                  embedded_tokens=self.opt["embedded_tokens"],
                                  vocab_size=self.opt["vocab_size"])
        return model
