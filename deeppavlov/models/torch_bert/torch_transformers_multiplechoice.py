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
from pathlib import Path
from typing import List, Dict, Union, Optional

import numpy as np
import torch
from transformers import AutoModelForMultipleChoice, AutoConfig

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel

log = getLogger(__name__)


@register('torch_transformers_multiplechoice')
class TorchTransformersMultiplechoiceModel(TorchModel):
    """Bert-based model for text classification on PyTorch.

    It uses output from [CLS] token and predicts labels using linear transformation.

    Args:
        n_classes: number of classes
        pretrained_bert: pretrained Bert checkpoint path or key title (e.g. "bert-base-uncased")
        multilabel: set True if it is multi-label classification
        return_probas: set True if return class probabilites instead of most probable label needed
        attention_probs_keep_prob: keep_prob for Bert self-attention layers
        hidden_keep_prob: keep_prob for Bert hidden layers
        bert_config_file: path to Bert configuration file (not used if pretrained_bert is key title)
    """

    def __init__(self, n_classes,
                 pretrained_bert,
                 multilabel: bool = False,
                 return_probas: bool = False,
                 attention_probs_keep_prob: Optional[float] = None,
                 hidden_keep_prob: Optional[float] = None,
                 bert_config_file: Optional[str] = None,
                 **kwargs) -> None:

        self.return_probas = return_probas
        self.multilabel = multilabel
        self.n_classes = n_classes

        if self.multilabel and not self.return_probas:
            raise RuntimeError('Set return_probas to True for multilabel classification!')

        if self.return_probas and self.n_classes == 1:
            raise RuntimeError('Set return_probas to False for regression task!')

        if pretrained_bert:
            log.debug(f"From pretrained {pretrained_bert}.")
            config = AutoConfig.from_pretrained(pretrained_bert, num_labels=self.n_classes,
                                                output_attentions=False, output_hidden_states=False)

            model = AutoModelForMultipleChoice.from_pretrained(pretrained_bert, config=config)

        elif bert_config_file and Path(bert_config_file).is_file():
            bert_config = AutoConfig.from_json_file(str(expand_path(bert_config_file)))
            if attention_probs_keep_prob is not None:
                bert_config.attention_probs_dropout_prob = 1.0 - attention_probs_keep_prob
            if hidden_keep_prob is not None:
                bert_config.hidden_dropout_prob = 1.0 - hidden_keep_prob
            model = AutoModelForMultipleChoice.from_config(config=bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")

        super().__init__(model, **kwargs)

    def train_on_batch(self, features: Dict[str, torch.tensor], y: Union[List[int], List[List[int]]]) -> Dict:
        """Train model on given batch.
        This method calls train_op using features and y (labels).

        Args:
            features: batch of InputFeatures
            y: batch of labels (class id or one-hot encoding)

        Returns:
            dict with loss and learning_rate values
        """

        _input = {key: value.to(self.device) for key, value in features.items()}

        _input["labels"] = torch.tensor(y).long().to(self.device)

        self.optimizer.zero_grad()

        tokenized = {key: value for (key, value) in _input.items() if key in self.model.forward.__code__.co_varnames}

        loss = self.model(**tokenized).loss
        self._make_step(loss)

        return {'loss': loss.item()}

    def __call__(self, features: Dict[str, torch.tensor]) -> Union[List[int], List[List[float]]]:
        """Make prediction for given features (texts).

        Args:
            features: batch of InputFeatures

        Returns:
            predicted classes or probabilities of each class

        """

        _input = {key: value.to(self.device) for key, value in features.items()}

        with torch.no_grad():
            tokenized = {key: value for (key, value) in _input.items()
                         if key in self.model.forward.__code__.co_varnames}

            # Forward pass, calculate logit predictions
            logits = self.model(**tokenized)
            logits = logits[0]

        if self.return_probas:
            if not self.multilabel:
                pred = torch.nn.functional.softmax(logits, dim=-1)
            else:
                pred = torch.nn.functional.sigmoid(logits)
            pred = pred.detach().cpu().numpy()
        elif self.n_classes > 1:
            logits = logits.detach().cpu().numpy()
            pred = np.argmax(logits, axis=1)
        else:  # regression
            pred = logits.squeeze(-1).detach().cpu().numpy()

        return pred
