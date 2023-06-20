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

import math
from logging import getLogger
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel
from deeppavlov.models.torch_bert.torch_transformers_sequence_tagger import token_from_subtoken

logger = getLogger(__name__)


class Biaffine(nn.Module):
    def __init__(self, in1_features: int, in2_features: int, out_features: int):
        super().__init__()
        self.bilinear = PairwiseBilinear(in1_features + 1, in2_features + 1, out_features)
        self.bilinear.weight.data.zero_()
        self.bilinear.bias.data.zero_()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], dim=input1.dim() - 1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], dim=input2.dim() - 1)
        return self.bilinear(input1, input2)


class PairwiseBilinear(nn.Module):
    """
    https://github.com/stanfordnlp/stanza/blob/v1.1.1/stanza/models/common/biaffine.py#L5  # noqa
    """

    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in1_features, out_features, in2_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(0))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        d1, d2, out = self.in1_features, self.in2_features, self.out_features
        n1, n2 = input1.size(1), input2.size(1)
        # (b * n1, d1) @ (d1, out * d2) => (b * n1, out * d2)
        x1W = torch.mm(input1.view(-1, d1), self.weight.view(d1, out * d2))
        # (b, n1 * out, d2) @ (b, d2, n2) => (b, n1 * out, n2)
        x1Wx2 = x1W.view(-1, n1 * out, d2).bmm(input2.transpose(1, 2))
        y = x1Wx2.view(-1, n1, self.out_features, n2).transpose(2, 3)
        if self.bias is not None:
            y.add_(self.bias)
        return y  # (b, n1, n2, out)

    def extra_repr(self) -> str:
        return "in1_features={}, in2_features={}, out_features={}, bias={}".format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None
        )


@torch.no_grad()
def mask_arc(lengths: torch.Tensor, mask_diag: bool = True) -> Optional[torch.Tensor]:
    b, n = lengths.numel(), lengths.max()
    if torch.all(lengths == n):
        if not mask_diag:
            return None
        mask = torch.ones(b, n, n + 1)
    else:
        mask = torch.zeros(b, n, n + 1)
        for i, length in enumerate(lengths):
            mask[i, :length, :length + 1] = 1
    if mask_diag:
        mask.masked_fill_(torch.eye(n, dtype=torch.bool), 0)
    return mask


class SyntaxParserNetwork(torch.nn.Module):
    """The model which defines heads in syntax tree and dependencies for text tokens.
       Text token ids are fed into Transformer encoder, hidden states are passed into dense layers followed by
       two biaffine layers (first for prediction of pairwise probabilities of a token to be the head for other token,
       second - for prediction of syntax dependency of a token).
    """

    def __init__(self, n_deps: int, pretrained_bert: str, encoder_layer_ids: List[int] = (-1,),
                 bert_config_file: Optional[str] = None, attention_probs_keep_prob: Optional[float] = None,
                 hidden_keep_prob: Optional[float] = None, state_size: int = 256, device: str = "gpu"):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.n_deps = n_deps
        self.encoder_layer_ids = encoder_layer_ids
        self.state_size = state_size
        if pretrained_bert:
            logger.debug(f"From pretrained {pretrained_bert}.")
            config = AutoConfig.from_pretrained(pretrained_bert, output_attentions=False, output_hidden_states=False)
            self.encoder = AutoModel.from_pretrained(pretrained_bert, config=config)

        elif bert_config_file and Path(bert_config_file).is_file():
            bert_config = AutoConfig.from_json_file(str(expand_path(bert_config_file)))
            if attention_probs_keep_prob is not None:
                bert_config.attention_probs_dropout_prob = 1.0 - attention_probs_keep_prob
            if hidden_keep_prob is not None:
                bert_config.hidden_dropout_prob = 1.0 - hidden_keep_prob
            self.encoder = AutoModel(config=bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")

        self.head_embs1 = torch.nn.Linear(config.hidden_size, state_size)
        self.dep_embs1 = torch.nn.Linear(config.hidden_size, state_size)
        self.head_embs2 = torch.nn.Linear(config.hidden_size, state_size)
        self.dep_embs2 = torch.nn.Linear(config.hidden_size, state_size)
        self.zero_emb1 = torch.nn.Parameter(torch.randn(state_size, ), requires_grad=True)
        self.zero_emb2 = torch.nn.Parameter(torch.randn(state_size, ), requires_grad=True)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.biaf_head = Biaffine(state_size, state_size, 1)
        self.biaf_dep = Biaffine(state_size, state_size, n_deps)

    def forward(self, input_ids, attention_mask, subtoken_mask, y_heads=None, y_dep=None):
        input_ids = torch.from_numpy(input_ids).to(self.device)
        attention_mask = torch.from_numpy(attention_mask).to(self.device)
        subtoken_mask = torch.from_numpy(subtoken_mask)

        outputs = self.encoder(input_ids, attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        layer_output_list = []
        for layer_id in self.encoder_layer_ids:
            layer_id = layer_id + 1 if layer_id != -1 else layer_id
            layer_output_list.append(hidden_states[layer_id])
        layer_output = torch.stack(layer_output_list)
        layer_output = torch.sum(layer_output, dim=0)

        layer_output = token_from_subtoken(layer_output, subtoken_mask)
        bs, seq_len, dim = layer_output.size()

        layer_output = layer_output.float().to(self.device)
        lengths = torch.sum(subtoken_mask, dim=-1)

        head1 = self.head_embs1(layer_output)
        dep1 = self.dep_embs1(layer_output)
        dep1_zero = [self.zero_emb1 for _ in range(bs)]
        dep1_zero = torch.stack(dep1_zero).unsqueeze(1).to(self.device)
        dep1 = torch.cat([dep1_zero, dep1], dim=1)

        head2 = self.head_embs2(layer_output)
        dep2 = self.dep_embs2(layer_output)
        dep2_zero = [self.zero_emb2 for _ in range(bs)]
        dep2_zero = torch.stack(dep2_zero).unsqueeze(1).to(self.device)
        dep2 = torch.cat([dep2_zero, dep2], dim=1)

        head1 = self.dropout(head1)
        dep1 = self.dropout(dep1)
        head2 = self.dropout(head2)
        dep2 = self.dropout(dep2)

        logits_head_init = self.biaf_head(head1, dep1).squeeze_(3)
        logits_deprel = self.biaf_dep(head2, dep2)
        mask = mask_arc(lengths, mask_diag=False)
        if mask is not None:
            logits_head_init.masked_fill_(mask.logical_not().to(logits_head_init.device), -10.0)
        logits_head = F.softmax(logits_head_init, dim=-1)

        head_loss, dep_loss = None, None
        if y_heads is not None:
            y_heads = tuple(torch.LongTensor(yh).to(self.device) for yh in y_heads)
            y_heads_pd = nn.utils.rnn.pad_sequence(y_heads, batch_first=True, padding_value=-1)

            logits_head_flatten = logits_head.contiguous().view(-1, logits_head.size(-1))
            y_heads_flatten = y_heads_pd.contiguous().view(-1)
            head_loss = F.cross_entropy(logits_head_flatten, y_heads_flatten, ignore_index=-1, reduction="sum")
            head_loss.div_((y_heads_flatten != -1).sum())

            y_dep = tuple(torch.LongTensor(ydp).to(self.device) for ydp in y_dep)
            y_dep_pd = nn.utils.rnn.pad_sequence(y_dep, batch_first=True, padding_value=-1)
            y_heads_new = y_heads_pd.masked_fill(y_heads_pd == -1, 0)
            gather_index = y_heads_new.view(*y_heads_new.size(), 1, 1).expand(-1, -1, -1, logits_deprel.size(-1))

            logits_deprel = torch.gather(logits_deprel, dim=2, index=gather_index)
            logits_deprel_flatten = logits_deprel.contiguous().view(-1, logits_deprel.size(-1))
            y_dep_flatten = y_dep_pd.contiguous().view(-1)
            dep_loss = F.cross_entropy(logits_deprel_flatten, y_dep_flatten, ignore_index=-1, reduction="sum")
            dep_loss.div_((y_dep_flatten != -1).sum())
        else:
            logits_head = logits_head.detach().cpu().numpy()
            head_ids = np.argmax(logits_head, axis=-1).tolist()

            head_ids_new = torch.LongTensor(head_ids)
            steps = torch.arange(head_ids_new.size(1))
            logits_deprel = [logits_deprel[i, steps, heads] for i, heads in enumerate(head_ids_new)]
            logits_deprel = torch.stack(logits_deprel, dim=0)
            deprels = logits_deprel.argmax(dim=2).detach().cpu().numpy().tolist()

            head_probas = [head_probas_list[:l, :l + 1] for l, head_probas_list in zip(lengths, logits_head)]
            deprels = [deprel[:l] for l, deprel in zip(lengths, deprels)]

        if y_heads is not None:
            return head_loss + dep_loss
        else:
            return head_probas, deprels


@register('torch_transformers_syntax_parser')
class TorchTransformersSyntaxParser(TorchModel):
    """Transformer-based model on PyTorch for syntax parsing. It predicts probabilities of heads and
       dependency ids for text tokens. 

    Args:
        pretrained_bert: pretrained Bert checkpoint path or key title (e.g. "bert-base-uncased")
        n_deps: number of syntax dependencies
        encoder_layer_ids: list of indexes of encoder layers which will be used for further predicting of heads and
            dependencies with biaffine layer
        state_size: size of dense layers which follow after transformer encoder
        attention_probs_keep_prob: keep_prob for Bert self-attention layers
        hidden_keep_prob: keep_prob for Bert hidden layers
        bert_config_file: path to Bert configuration file, or None, if `pretrained_bert` is a string name
    """

    def __init__(self, pretrained_bert: str,
                 n_deps: int,
                 encoder_layer_ids: List[int] = (-1,),
                 state_size: int = 256,
                 attention_probs_keep_prob: Optional[float] = None,
                 hidden_keep_prob: Optional[float] = None,
                 bert_config_file: Optional[str] = None,
                 **kwargs) -> None:

        model = SyntaxParserNetwork(n_deps, pretrained_bert, encoder_layer_ids,
                                    bert_config_file, attention_probs_keep_prob, hidden_keep_prob,
                                    state_size)
        super().__init__(model, **kwargs)

    def train_on_batch(self, input_ids: Union[List[List[int]], np.ndarray],
                       input_masks: Union[List[List[int]], np.ndarray],
                       y_masks: Union[List[List[int]], np.ndarray],
                       y_heads: List[List[int]], y_dep: List[List[int]]) -> Dict:
        """

        Args:
            input_ids: indices of the subwords
            input_masks: mask that determines where to attend and where not to
            y_masks: mask which determines the first subword units in the the word
            y_heads: for each token - id fo token which is the head in syntax tree for the token
            y_dep: syntax dependencies for each tokens
        """
        self.optimizer.zero_grad()
        loss = self.model(input_ids, input_masks, y_masks, y_heads, y_dep)
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        self.optimizer.step()

        return {'loss': loss.item()}

    def __call__(self, input_ids: Union[List[List[int]], np.ndarray],
                 input_masks: Union[List[List[int]], np.ndarray],
                 y_masks: Union[List[List[int]], np.ndarray]) -> Tuple[List[List[List[float]]], List[List[int]]]:
        """ Predicts probas of heads and dependency ids for tokens

        Args:
            input_ids: indices of the subwords
            input_masks: mask that determines where to attend and where not to
            y_masks: mask which determines the first subword units in the the word

        Returns:
            Probas of heads and dependency ids for each token (not subtoken)

        """
        with torch.no_grad():
            head_probas, dep_ids = self.model(input_ids, input_masks, y_masks)
        return head_probas, dep_ids
