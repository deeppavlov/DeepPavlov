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
from overrides import overrides
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers.data.processors.utils import InputFeatures

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel

log = getLogger(__name__)


@register('torch_bert_ranker')
class TorchBertRankerModel(TorchModel):
    """BERT-based model for interaction-based text ranking on PyTorch.

    Linear transformation is trained over the BERT pooled output from [CLS] token.
    Predicted probabilities of classes are used as a similarity measure for ranking.

    Args:
        pretrained_bert: pretrained Bert checkpoint path or key title (e.g. "bert-base-uncased")
        bert_config_file: path to Bert configuration file (not used if pretrained_bert is key title)
        n_classes: number of classes
        return_probas: set True if class probabilities are returned instead of the most probable label
        optimizer: optimizer name from `torch.optim`
        optimizer_parameters: dictionary with optimizer's parameters,
                              e.g. {'lr': 0.1, 'weight_decay': 0.001, 'momentum': 0.9}
    """

    def __init__(self, pretrained_bert: str,
                 bert_config_file: Optional[str] = None,
                 n_classes: int = 2,
                 return_probas: bool = True,
                 optimizer: str = "AdamW",
                 clip_norm: Optional[float] = None,
                 optimizer_parameters: Optional[dict] = None,
                 **kwargs) -> None:

        if not optimizer_parameters:
            optimizer_parameters = {"lr": 2e-5,
                                    "weight_decay": 0.01,
                                    "betas": (0.9, 0.999),
                                    "eps": 1e-6}

        self.return_probas = return_probas
        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file
        self.n_classes = n_classes
        self.clip_norm = clip_norm

        if self.return_probas and self.n_classes == 1:
            raise RuntimeError('Set return_probas to False for regression task!')

        super().__init__(optimizer=optimizer,
                         optimizer_parameters=optimizer_parameters,
                         **kwargs)

    def train_on_batch(self, features_li: List[List[InputFeatures]], y: Union[List[int], List[List[int]]]) -> Dict:
        """Train the model on the given batch.

        Args:
            features_li: list with the single element containing the batch of InputFeatures
            y: batch of labels (class id or one-hot encoding)

        Returns:
            dict with loss and learning rate values
        """
        features = features_li[0]

        input_ids = [f.input_ids for f in features]
        input_masks = [f.attention_mask for f in features]

        b_input_ids = torch.cat(input_ids, dim=0).to(self.device)
        b_input_masks = torch.cat(input_masks, dim=0).to(self.device)
        b_labels = torch.from_numpy(np.array(y)).to(self.device)

        self.optimizer.zero_grad()

        loss, logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_masks,
                                  labels=b_labels)
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {'loss': loss.item()}

    def __call__(self, features_li: List[List[InputFeatures]]) -> Union[List[int], List[List[float]]]:
        """Calculate scores for the given context over candidate responses.

        Args:
            features_li: list of elements where each element contains the batch of features
             for contexts with particular response candidates

        Returns:
            predicted scores for contexts over response candidates
        """
        if len(features_li) == 1 and len(features_li[0]) == 1:
            msg = f"It is not intended to use the {self.__class__} in the interact mode."
            log.error(msg)
            return [msg]

        predictions = []
        for features in features_li:

            input_ids = [f.input_ids for f in features]
            input_masks = [f.attention_mask for f in features]

            b_input_ids = torch.cat(input_ids, dim=0).to(self.device)
            b_input_masks = torch.cat(input_masks, dim=0).to(self.device)

            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_masks)
                logits = logits[0]

            if self.return_probas:
                pred = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
                pred = pred.detach().cpu().numpy()
            else:
                logits = logits.detach().cpu().numpy()
                pred = np.argmax(logits, axis=1)

            predictions.append(pred)

        if len(features_li) == 1:
            predictions = predictions[0]
        else:
            predictions = np.hstack([np.expand_dims(el, 1) for el in predictions])

        return predictions

    @overrides
    def load(self, fname=None):
        if fname is not None:
            self.load_path = fname

        if self.pretrained_bert:
            log.info(f"From pretrained {self.pretrained_bert}.")
            config = AutoConfig.from_pretrained(self.pretrained_bert,
                                                # num_labels=self.n_classes,
                                                output_attentions=False,
                                                output_hidden_states=False)

            self.model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_bert, config=config)

            try:
                hidden_size = self.model.classifier.out_proj.in_features

                if self.n_classes != self.model.num_labels:
                    self.model.classifier.out_proj.weight = torch.nn.Parameter(torch.randn(self.n_classes, hidden_size))
                    self.model.classifier.out_proj.bias = torch.nn.Parameter(torch.randn(self.n_classes))
                    self.model.classifier.out_proj.out_features = self.n_classes
                    self.model.num_labels = self.n_classes

            except torch.nn.modules.module.ModuleAttributeError:
                hidden_size = self.model.classifier.in_features

                if self.n_classes != self.model.num_labels:
                    self.model.classifier.weight = torch.nn.Parameter(torch.randn(self.n_classes, hidden_size))
                    self.model.classifier.bias = torch.nn.Parameter(torch.randn(self.n_classes))
                    self.model.classifier.out_features = self.n_classes
                    self.model.num_labels = self.n_classes


        elif self.bert_config_file and Path(self.bert_config_file).is_file():
            self.bert_config = AutoConfig.from_json_file(str(expand_path(self.bert_config_file)))
            if self.attention_probs_keep_prob is not None:
                self.bert_config.attention_probs_dropout_prob = 1.0 - self.attention_probs_keep_prob
            if self.hidden_keep_prob is not None:
                self.bert_config.hidden_dropout_prob = 1.0 - self.hidden_keep_prob
            self.model = AutoModelForSequenceClassification.from_config(config=self.bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")

        self.model.to(self.device)

        self.optimizer = getattr(torch.optim, self.optimizer_name)(
            self.model.parameters(), **self.optimizer_parameters)
        if self.lr_scheduler_name is not None:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                self.optimizer, **self.lr_scheduler_parameters)

        if self.load_path:
            log.info(f"Load path {self.load_path} is given.")
            if isinstance(self.load_path, Path) and not self.load_path.parent.is_dir():
                raise ConfigError("Provided load path is incorrect!")

            weights_path = Path(self.load_path.resolve())
            weights_path = weights_path.with_suffix(f".pth.tar")
            if weights_path.exists():
                log.info(f"Load path {weights_path} exists.")
                log.info(f"Initializing `{self.__class__.__name__}` from saved.")

                # now load the weights, optimizer from saved
                log.info(f"Loading weights from {weights_path}.")
                checkpoint = torch.load(weights_path, map_location=self.device)
                # set strict flag to False if position_ids are missing
                # this is needed to load models trained on older versions
                # of transformers library
                strict_load_flag = bool([key for key in checkpoint["model_state_dict"].keys()
                                         if key.endswith("embeddings.position_ids")])
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict_load_flag)
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.epochs_done = checkpoint.get("epochs_done", 0)
            else:
                log.info(f"Init from scratch. Load path {weights_path} does not exist.")
