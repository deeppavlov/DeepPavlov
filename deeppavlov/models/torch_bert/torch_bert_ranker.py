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
    """

    def __init__(self, pretrained_bert: str = None,
                 bert_config_file: Optional[str] = None,
                 n_classes: int = 2,
                 return_probas: bool = True,
                 **kwargs) -> None:

        self.return_probas = return_probas

        if self.return_probas and n_classes == 1:
            raise RuntimeError('Set return_probas to False for regression task!')

        if pretrained_bert:
            log.debug(f"From pretrained {pretrained_bert}.")
            if Path(expand_path(pretrained_bert)).exists():
                pretrained_bert = str(expand_path(pretrained_bert))
            config = AutoConfig.from_pretrained(pretrained_bert,
                                                # num_labels=self.n_classes,
                                                output_attentions=False,
                                                output_hidden_states=False)

            model = AutoModelForSequenceClassification.from_pretrained(pretrained_bert, config=config)

            # TODO: make better exception handling here and at
            # deeppavlov.models.torch_bert.torch_transformers_classifier.TorchTransformersClassifierModel.load
            try:
                hidden_size = model.classifier.out_proj.in_features

                if n_classes != model.num_labels:
                    model.classifier.out_proj.weight = torch.nn.Parameter(torch.randn(n_classes, hidden_size))
                    model.classifier.out_proj.bias = torch.nn.Parameter(torch.randn(n_classes))
                    model.classifier.out_proj.out_features = n_classes
                    model.num_labels = n_classes

            except AttributeError:
                hidden_size = model.classifier.in_features

                if n_classes != model.num_labels:
                    model.classifier.weight = torch.nn.Parameter(torch.randn(n_classes, hidden_size))
                    model.classifier.bias = torch.nn.Parameter(torch.randn(n_classes))
                    model.classifier.out_features = n_classes
                    model.num_labels = n_classes


        elif bert_config_file and expand_path(bert_config_file).is_file():
            self.bert_config = AutoConfig.from_pretrained(str(expand_path(bert_config_file)))
            model = AutoModelForSequenceClassification.from_config(config=self.bert_config)

        else:
            raise ConfigError("No pre-trained BERT model is given.")

        super().__init__(model, **kwargs)

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
                                  labels=b_labels, return_dict=False)
        self._make_step(loss)

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
