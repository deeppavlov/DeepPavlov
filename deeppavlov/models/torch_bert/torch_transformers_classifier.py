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
from typing import List, Dict, Union, Optional, Tuple

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel

log = getLogger(__name__)


@register('torch_transformers_classifier')
class TorchTransformersClassifierModel(TorchModel):
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
        is_binary: whether classification task is binary or multi-class
        num_special_tokens: number of special tokens used by classification model
    """

    def __init__(self, n_classes,
                 pretrained_bert,
                 multilabel: bool = False,
                 return_probas: bool = False,
                 attention_probs_keep_prob: Optional[float] = None,
                 hidden_keep_prob: Optional[float] = None,
                 bert_config_file: Optional[str] = None,
                 is_binary: Optional[bool] = False,
                 num_special_tokens: int = None,
                 **kwargs) -> None:

        self.return_probas = return_probas
        self.multilabel = multilabel
        self.n_classes = n_classes
        self.is_binary = is_binary

        if self.multilabel and not self.return_probas:
            raise RuntimeError('Set return_probas to True for multilabel classification!')

        if self.return_probas and self.n_classes == 1:
            raise RuntimeError('Set return_probas to False for regression task!')

        if pretrained_bert:
            log.debug(f"From pretrained {pretrained_bert}.")
            config = AutoConfig.from_pretrained(pretrained_bert,
                                                # num_labels=self.n_classes,
                                                output_attentions=False,
                                                output_hidden_states=False)

            if self.is_binary:
                config.add_pooling_layer = False
                model = AutoModelForBinaryClassification(pretrained_bert, config)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(pretrained_bert, config=config)

                # TODO need a better solution here and at
                # deeppavlov.models.torch_bert.torch_bert_ranker.TorchBertRankerModel.load
                try:
                    hidden_size = model.classifier.out_proj.in_features

                    if self.n_classes != model.num_labels:
                        model.classifier.out_proj.weight = torch.nn.Parameter(torch.randn(self.n_classes,
                                                                                               hidden_size))
                        model.classifier.out_proj.bias = torch.nn.Parameter(torch.randn(self.n_classes))
                        model.classifier.out_proj.out_features = self.n_classes
                        model.num_labels = self.n_classes

                except AttributeError:
                    hidden_size = model.classifier.in_features

                    if self.n_classes != model.num_labels:
                        model.classifier.weight = torch.nn.Parameter(torch.randn(self.n_classes, hidden_size))
                        model.classifier.bias = torch.nn.Parameter(torch.randn(self.n_classes))
                        model.classifier.out_features = self.n_classes
                        model.num_labels = self.n_classes

        elif bert_config_file and Path(bert_config_file).is_file():
            bert_config = AutoConfig.from_pretrained(str(expand_path(bert_config_file)))
            if attention_probs_keep_prob is not None:
                bert_config.attention_probs_dropout_prob = 1.0 - attention_probs_keep_prob
            if hidden_keep_prob is not None:
                bert_config.hidden_dropout_prob = 1.0 - hidden_keep_prob
            model = AutoModelForSequenceClassification.from_config(config=bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")

        tokenizer = AutoTokenizer.from_pretrained(pretrained_bert)
        if num_special_tokens is not None:
            model.resize_token_embeddings(len(tokenizer) + num_special_tokens)

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

        if self.n_classes > 1 and not self.is_binary:
            _input["labels"] = torch.from_numpy(np.array(y)).to(self.device)

        # regression
        else:
            _input["labels"] = torch.from_numpy(np.array(y, dtype=np.float32)).unsqueeze(1).to(self.device)

        self.optimizer.zero_grad()

        tokenized = {key: value for (key, value) in _input.items()
                     if key in self.accepted_keys}

        loss = self.model(**tokenized).loss
        if self.is_data_parallel:
            loss = loss.mean()
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
                         if key in self.accepted_keys}

            # Forward pass, calculate logit predictions
            logits = self.model(**tokenized)
            logits = logits[0]

        if self.return_probas:
            if self.is_binary:
                pred = torch.sigmoid(logits).squeeze(1)
            elif not self.multilabel:
                pred = torch.nn.functional.softmax(logits, dim=-1)
            else:
                pred = torch.nn.functional.sigmoid(logits)
            pred = pred.detach().cpu().numpy()
        elif self.n_classes > 1:
            logits = logits.detach().cpu().numpy()
            pred = np.argmax(logits, axis=1)
        # regression
        else:
            pred = logits.squeeze(-1).detach().cpu().numpy()

        return pred

    # TODO move to the super class
    @property
    def accepted_keys(self) -> Tuple[str]:
        if self.is_data_parallel:
            accepted_keys = self.model.module.forward.__code__.co_varnames
        else:
            accepted_keys = self.model.forward.__code__.co_varnames
        return accepted_keys


class AutoModelForBinaryClassification(torch.nn.Module):

    def __init__(self, pretrained_bert, config):
        super().__init__()
        self.pretrained_bert = pretrained_bert
        self.config = config

        self.model = AutoModel.from_pretrained(self.pretrained_bert, self.config)
        self.classifier = BinaryClassificationHead(config)

        self.classifier.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states,
                             return_dict=return_dict)

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss,
                                        logits=logits,
                                        hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)


class BinaryClassificationHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(config.hidden_size, 1)

    def init_weights(self):
        self.dense.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if self.dense.bias is not None:
            self.dense.bias.data.zero_()

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
