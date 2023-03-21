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
from typing import List, Optional, Dict, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig, AutoModel, AutoTokenizer

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel

log = getLogger(__name__)


@register('torch_transformers_nll_ranker')
class TorchTransformersNLLRanker(TorchModel):
    """Class for ranking of relations using the model trained with NLL loss
    Args:
        model_name: name of the function which initialises and returns the model class
        pretrained_bert: pretrained transformer checkpoint path or key title (e.g. "bert-base-uncased")
        encoder_save_path: path to save the encoder checkpoint
        linear_save_path: path to save linear layer checkpoint
        bert_config_file: path to Bert configuration file, or None, if `pretrained_bert` is a string name
        criterion: name of loss function
        optimizer: optimizer name from `torch.optim`
        optimizer_parameters: dictionary with optimizer's parameters,
                              e.g. {'lr': 0.1, 'weight_decay': 0.001, 'momentum': 0.9}
        return_probas: set this to `True` if you need the probabilities instead of raw answers
        attention_probs_keep_prob: keep_prob for Bert self-attention layers
        hidden_keep_prob: keep_prob for Bert hidden layers
        clip_norm: clip gradients by norm
    """

    def __init__(
            self,
            model_name: str,
            pretrained_bert: str = None,
            encoder_save_path: str = None,
            linear_save_path: str = None,
            bert_config_file: Optional[str] = None,
            criterion: str = "CrossEntropyLoss",
            optimizer: str = "AdamW",
            optimizer_parameters: Dict = None,
            return_probas: bool = False,
            attention_probs_keep_prob: Optional[float] = None,
            hidden_keep_prob: Optional[float] = None,
            clip_norm: Optional[float] = None,
            **kwargs
    ):
        self.pretrained_bert = pretrained_bert
        self.encoder_save_path = encoder_save_path
        self.linear_save_path = linear_save_path
        self.bert_config_file = bert_config_file
        self.return_probas = return_probas
        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob
        self.clip_norm = clip_norm
        if optimizer_parameters is None:
            optimizer_parameters = {"lr": 1e-5, "weight_decay": 0.01, "eps": 1e-6}

        super().__init__(
            model_name=model_name,
            optimizer=optimizer,
            criterion=criterion,
            optimizer_parameters=optimizer_parameters,
            return_probas=return_probas,
            **kwargs)

    def train_on_batch(self, input_features: Dict[str, Any], positive_idx: List[int]) -> float:
        _input = {'positive_idx': positive_idx,
                  "input_ids": torch.LongTensor(input_features["input_ids"]).to(self.device),
                  "attention_mask": torch.LongTensor(input_features["attention_mask"]).to(self.device),
                  "token_type_ids": torch.LongTensor(input_features["token_type_ids"]).to(self.device)}

        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()  # zero the parameter gradients

        loss, softmax_scores = self.model(**_input)
        loss.backward()
        self.optimizer.step()

        # Clip the norm of the gradients to prevent the "exploding gradients" problem
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item()

    def __call__(self, input_features: Dict[str, Any]) -> Union[List[int], List[np.ndarray]]:
        self.model.eval()
        _input = {"input_ids": torch.LongTensor(input_features["input_ids"]).to(self.device),
                  "attention_mask": torch.LongTensor(input_features["attention_mask"]).to(self.device),
                  "token_type_ids": torch.LongTensor(input_features["token_type_ids"]).to(self.device)}

        with torch.no_grad():
            output = self.model(**_input)
            if isinstance(output, tuple) and len(output) == 2:
                loss, softmax_scores = output
            else:
                softmax_scores = output
        if self.return_probas:
            softmax_scores = softmax_scores.cpu().numpy().tolist()
            return softmax_scores
        else:
            pred = torch.argmax(softmax_scores, dim=1)
            pred = pred.cpu()
            pred = pred.numpy()
            return pred

    def in_batch_ranking_model(self, **kwargs) -> nn.Module:
        return NLLRanking(
            pretrained_bert=self.pretrained_bert,
            encoder_save_path=self.encoder_save_path,
            linear_save_path=self.linear_save_path,
            bert_tokenizer_config_file=self.pretrained_bert,
            device=self.device
        )

    def save(self, fname: Optional[str] = None, *args, **kwargs) -> None:
        if fname is None:
            fname = self.save_path
        if not fname.parent.is_dir():
            raise ConfigError("Provided save path is incorrect!")
        weights_path = Path(fname).with_suffix(f".pth.tar")
        log.info(f"Saving model to {weights_path}.")
        torch.save({
            "model_state_dict": self.model.cpu().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epochs_done": self.epochs_done
        }, weights_path)
        self.model.to(self.device)


class NLLRanking(nn.Module):
    """Class which implements the relation ranking model
    Args:
        pretrained_bert: pretrained transformer checkpoint path or key title (e.g. "bert-base-uncased")
        encoder_save_path: path to save the encoder checkpoint
        linear_save_path: path to save linear layer checkpoint
        bert_tokenizer_config_file: path to configuration file of transformer tokenizer
        bert_config_file: path to transformer configuration file, or None, if `pretrained_bert` is a string name
        device: cpu or gpu
    """

    def __init__(
            self,
            pretrained_bert: str = None,
            encoder_save_path: str = None,
            linear_save_path: str = None,
            bert_tokenizer_config_file: str = None,
            bert_config_file: str = None,
            device: str = "gpu"
    ):
        super().__init__()
        self.pretrained_bert = pretrained_bert
        self.encoder_save_path = encoder_save_path
        self.linear_save_path = linear_save_path
        self.bert_config_file = bert_config_file
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")

        # initialize parameters that would be filled later
        self.encoder, self.config, self.bert_config = None, None, None
        self.load()

        if Path(bert_tokenizer_config_file).is_file():
            vocab_file = str(expand_path(bert_tokenizer_config_file))
            tokenizer = AutoTokenizer(vocab_file=vocab_file)
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_bert)
        self.encoder.resize_token_embeddings(len(tokenizer) + 7)

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor,
            token_type_ids: Tensor,
            positive_idx: List[List[int]] = None
    ) -> Union[Tuple[Any, Tensor], Tuple[Tensor]]:

        bs, samples_num, seq_len = input_ids.size()
        input_ids = input_ids.reshape(bs * samples_num, -1)
        attention_mask = attention_mask.reshape(bs * samples_num, -1)
        token_type_ids = token_type_ids.reshape(bs * samples_num, -1)
        if hasattr(self.config, "type_vocab_size"):
            encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                          token_type_ids=token_type_ids)
        else:
            encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = encoder_output.last_hidden_state[:, :1, :].squeeze(1)
        scores = self.fc(cls_emb)
        scores = scores.reshape(bs, samples_num)

        if positive_idx is not None:
            scores = F.log_softmax(scores, dim=1)
            positive_idx = []
            for i in range(bs):
                positive_idx.append(0)
            loss = F.nll_loss(scores, torch.tensor(positive_idx).to(scores.device), reduction="mean")
            return loss, scores
        else:
            return scores

    def load(self) -> None:
        if self.pretrained_bert:
            log.info(f"From pretrained {self.pretrained_bert}.")
            self.config = AutoConfig.from_pretrained(
                self.pretrained_bert, output_hidden_states=True
            )
            self.encoder = AutoModel.from_pretrained(self.pretrained_bert, config=self.config)
            self.fc = nn.Linear(self.config.hidden_size, 1)

        elif self.bert_config_file and Path(self.bert_config_file).is_file():
            self.config = AutoConfig.from_json_file(str(expand_path(self.bert_config_file)))
            self.encoder = AutoModel.from_config(config=self.bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")

        self.encoder.to(self.device)
        self.fc.to(self.device)
