# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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

import os
from pathlib import Path
from logging import getLogger
from typing import List, Union, Dict
from overrides import overrides

import numpy as np
import torch
from transformers import BertForTokenClassification, BertConfig

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel

log = getLogger(__name__)


@register('torch_bert_sequence_tagger')
class TorchBertSequenceTagger(TorchModel):
    """BERT-based model for text tagging. It predicts a label for every token (not subtoken) in the text.
    You can use it for sequence labeling tasks, such as morphological tagging or named entity recognition.
    See :class:`deeppavlov.models.bert.bert_sequence_tagger.BertSequenceNetwork`
    for the description of inherited parameters.

    Args:
        n_tags: number of distinct tags
        keep_prob: dropout keep_prob for non-Bert layers
        return_probas: set this to `True` if you need the probabilities instead of raw answers
        bert_config_file: path to Bert configuration file
        pretrained_bert: pretrained Bert checkpoint
        attention_probs_keep_prob: keep_prob for Bert self-attention layers
        hidden_keep_prob: keep_prob for Bert hidden layers
        encoder_layer_ids: list of averaged layers from Bert encoder (layer ids)
            optimizer: name of tf.train.* optimizer or None for `AdamWeightDecayOptimizer`
            weight_decay_rate: L2 weight decay for `AdamWeightDecayOptimizer`
        encoder_dropout: dropout probability of encoder output layer
        ema_decay: what exponential moving averaging to use for network parameters, value from 0.0 to 1.0.
            Values closer to 1.0 put weight on the parameters history and values closer to 0.0 corresponds put weight
            on the current parameters.
        ema_variables_on_cpu: whether to put EMA variables to CPU. It may save a lot of GPU memory
        freeze_embeddings: set True to not train input embeddings set True to
            not train input embeddings set True to not train input embeddings
        learning_rate: learning rate of BERT head
        bert_learning_rate: learning rate of BERT body
        min_learning_rate: min value of learning rate if learning rate decay is used
        learning_rate_drop_patience: how many validations with no improvements to wait
        learning_rate_drop_div: the divider of the learning rate after `learning_rate_drop_patience` unsuccessful
            validations
        load_before_drop: whether to load best model before dropping learning rate or not
        clip_norm: clip gradients by norm
    """

    def __init__(self,
                 n_tags: int,
                 keep_prob: float,
                 bert_config_file: str = None,
                 return_probas: bool = False,
                 pretrained_bert: str = None,
                 attention_probs_keep_prob: float = None,
                 hidden_keep_prob: float = None,
                 encoder_layer_ids: List[int] = (-1,),
                 encoder_dropout: float = 0.0,
                 optimizer: str = None,
                 optimizer_parameters={"lr": 1e-3, "weight_decay_rate": 1e-6},
                 freeze_embeddings: bool = False,
                 bert_learning_rate: float = 2e-5,
                 min_learning_rate: float = 1e-07,
                 learning_rate_drop_patience: int = 20,
                 learning_rate_drop_div: float = 2.0,
                 load_before_drop: bool = True,
                 clip_norm: float = 1.0,
                 **kwargs) -> None:

        self.n_classes = n_tags
        self.return_probas = return_probas
        self.keep_prob = keep_prob
        self.encoder_layer_ids = encoder_layer_ids
        self.encoder_dropout = encoder_dropout
        self.freeze_embeddings = freeze_embeddings
        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob

        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file

        super().__init__(optimizer=optimizer,
                         optimizer_parameters=optimizer_parameters,
                         learning_rate_drop_patience=learning_rate_drop_patience,
                         learning_rate_drop_div=learning_rate_drop_div,
                         load_before_drop=load_before_drop,
                         min_learning_rate=min_learning_rate,
                         clip_norm=clip_norm,
                         **kwargs)

        self.load()
        self.model.to(self.device)
        # need to move it to `eval` mode because it can be used in `build_model` (not by `torch_trainer`
        self.model.eval()

    def train_on_batch(self,
                       input_ids: Union[List[List[int]], np.ndarray],
                       input_masks: Union[List[List[int]], np.ndarray],
                       y: List[List[int]],
                       *args, **kwargs) -> Dict[str, float]:
        """

        Args:
            input_ids: batch of indices of subwords
            input_masks: batch of masks which determine what should be attended
            args: arguments passed  to _build_feed_dict
                and corresponding to additional input
                and output tensors of the derived class.
            kwargs: keyword arguments passed to _build_feed_dict
                and corresponding to additional input
                and output tensors of the derived class.

        Returns:
            dict with fields 'loss', 'head_learning_rate', and 'bert_learning_rate'
        """
        b_input_ids = torch.from_numpy(input_ids).to(self.device)
        b_input_masks = torch.from_numpy(input_masks).to(self.device)
        b_labels = torch.from_numpy(np.array(y)).to(self.device)

        self.optimizer.zero_grad()

        loss, logits = self.model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_masks,
                                  labels=b_labels)
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {'loss': loss.item()}

    def __call__(self,
                 input_ids: Union[List[List[int]], np.ndarray],
                 input_masks: Union[List[List[int]], np.ndarray],
                 y_masks: Union[List[List[int]], np.ndarray]) -> Union[List[List[int]], List[np.ndarray]]:
        """ Predicts tag indices for a given subword tokens batch

        Args:
            input_ids: indices of the subwords
            input_masks: mask that determines where to attend and where not to
            y_masks: mask which determines the first subword units in the the word

        Returns:
            Label indices or class probabilities for each token (not subtoken)

        """
        b_input_ids = torch.from_numpy(input_ids).to(self.device)
        b_input_masks = torch.from_numpy(input_masks).to(self.device)

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_masks)

        # Move logits and labels to CPU and to numpy arrays
        logits = logits[0].detach().cpu().numpy()

        if self.return_probas:
            pred = logits
        else:
            pred = np.argmax(logits, axis=1)
        return pred

    @overrides
    def load(self):
        if self.pretrained_bert and not os.path.isfile(self.pretrained_bert):
            self.model = BertForTokenClassification.from_pretrained(
                self.pretrained_bert, num_labels=self.n_classes,
                output_attentions=False, output_hidden_states=False)
        elif self.bert_config_file and os.path.isfile(self.bert_config_file):
            self.bert_config = BertConfig.from_json_file(str(expand_path(self.bert_config_file)))

            if self.attention_probs_keep_prob is not None:
                self.bert_config.attention_probs_dropout_prob = 1.0 - self.attention_probs_keep_prob
            if self.hidden_keep_prob is not None:
                self.bert_config.hidden_dropout_prob = 1.0 - self.hidden_keep_prob
            self.model = BertForTokenClassification(config=self.bert_config)

        self.optimizer = getattr(torch.optim, self.opt["optimizer"])(
            self.model.parameters(), **self.opt.get("optimizer_parameters", {}))
        if self.opt.get("lr_scheduler", None):
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.opt["lr_scheduler"])(
                self.optimizer, **self.opt.get("lr_scheduler_parameters", {}))

        if self.load_path:
            log.info(f"Load path {self.load_path} is given.")
            if isinstance(self.load_path, Path) and not self.load_path.parent.is_dir():
                raise ConfigError("Provided load path is incorrect!")

            weights_path = Path("{}.pth.tar".format(str(self.load_path.resolve())))
            if weights_path.exists():
                log.info(f"Load path {weights_path} exists.")
                log.info(f"Initializing `{self.__class__.__name__}` from saved.")

                # now load the weights, optimizer from saved
                log.info(f"Loading weights from {weights_path}.")
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.epochs_done = checkpoint.get("epochs_done", 0)
            else:
                log.info(f"Init from scratch. Load path {weights_path} does not exist.")

        self.model.to(self.device)
