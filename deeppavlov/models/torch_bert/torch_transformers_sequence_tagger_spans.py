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

from logging import getLogger
from pathlib import Path
from typing import List, Union, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import transformers
from overrides import overrides
from transformers import AutoModelForTokenClassification, AutoConfig, AutoModel

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel

log = getLogger(__name__)


def token_from_subtoken(units: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ Assemble token level units from subtoken level units

    Args:
        units: torch.Tensor of shape [batch_size, SUBTOKEN_seq_length, n_features]
        mask: mask of token beginnings. For example: for tokens

                [[``[CLS]`` ``My``, ``capybara``, ``[SEP]``],
                [``[CLS]`` ``Your``, ``aar``, ``##dvark``, ``is``, ``awesome``, ``[SEP]``]]

            the mask will be

                [[0, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0]]

    Returns:
        word_level_units: Units assembled from ones in the mask. For the
            example above this units will correspond to the following

                [[``My``, ``capybara``],
                [``Your`, ``aar``, ``is``, ``awesome``,]]

            the shape of this tensor will be [batch_size, TOKEN_seq_length, n_features]
    """
    shape = units.size()
    batch_size = shape[0]
    nf = shape[2]
    nf_int = units.size()[-1]

    # number of TOKENS in each sentence
    token_seq_lengths = torch.sum(mask, 1).to(torch.int64)
    # for a matrix m =
    # [[1, 1, 1],
    #  [0, 1, 1],
    #  [1, 0, 0]]
    # it will be
    # [3, 2, 1]

    n_words = torch.sum(token_seq_lengths)
    # n_words -> 6

    max_token_seq_len = torch.max(token_seq_lengths)
    # max_token_seq_len -> 3

    idxs = torch.stack(torch.nonzero(mask, as_tuple=True), dim=1)
    # for the matrix mentioned above
    # tf.where(mask) ->
    # [[0, 0],
    #  [0, 1]
    #  [0, 2],
    #  [1, 1],
    #  [1, 2]
    #  [2, 0]]

    sample_ids_in_batch = torch.nn.functional.pad(input=idxs[:, 0], pad=[1, 0])
    # for indices
    # [[0, 0],
    #  [0, 1]
    #  [0, 2],
    #  [1, 1],
    #  [1, 2],
    #  [2, 0]]
    # it is
    # [0, 0, 0, 0, 1, 1, 2]
    # padding is for computing change from one sample to another in the batch

    a = torch.logical_not(torch.eq(sample_ids_in_batch[1:], sample_ids_in_batch[:-1]).to(torch.int64))
    # for the example above the result of this statement equals
    # [0, 0, 0, 1, 0, 1]
    # so data samples begin in 3rd and 5th positions (the indexes of ones)

    # transforming sample start masks to the sample starts themselves
    q = a * torch.arange(n_words).to(torch.int64)
    # [0, 0, 0, 3, 0, 5]
    count_to_substract = torch.nn.functional.pad(torch.masked_select(q, q.to(torch.bool)), [1, 0])
    # [0, 3, 5]

    new_word_indices = torch.arange(n_words).to(torch.int64) - torch.gather(
        count_to_substract, dim=0, index=torch.cumsum(a, 0))
    # tf.range(n_words) -> [0, 1, 2, 3, 4, 5]
    # tf.cumsum(a) -> [0, 0, 0, 1, 1, 2]
    # tf.gather(count_to_substract, tf.cumsum(a)) -> [0, 0, 0, 3, 3, 5]
    # new_word_indices -> [0, 1, 2, 3, 4, 5] - [0, 0, 0, 3, 3, 5] = [0, 1, 2, 0, 1, 0]
    # new_word_indices is the concatenation of range(word_len(sentence))
    # for all sentences in units

    n_total_word_elements = (batch_size * max_token_seq_len).to(torch.int32)
    word_indices_flat = (idxs[:, 0] * max_token_seq_len + new_word_indices).to(torch.int64)
    x_mask = torch.sum(torch.nn.functional.one_hot(word_indices_flat, n_total_word_elements), 0)
    x_mask = x_mask.to(torch.bool)
    # to get absolute indices we add max_token_seq_len:
    # idxs[:, 0] * max_token_seq_len -> [0, 0, 0, 1, 1, 2] * 2 = [0, 0, 0, 3, 3, 6]
    # word_indices_flat -> [0, 0, 0, 3, 3, 6] + [0, 1, 2, 0, 1, 0] = [0, 1, 2, 3, 4, 6]
    # total number of words in the batch (including paddings)
    # batch_size * max_token_seq_len -> 3 * 3 = 9
    # tf.one_hot(...) ->
    # [[1. 0. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 1. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 1. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0. 0. 1. 0. 0.]]
    #  x_mask -> [1, 1, 1, 1, 1, 0, 1, 0, 0]

    full_range = torch.arange(batch_size * max_token_seq_len).to(torch.int64)
    # full_range -> [0, 1, 2, 3, 4, 5, 6, 7, 8]
    nonword_indices_flat = torch.masked_select(full_range, torch.logical_not(x_mask))

    # # y_idxs -> [5, 7, 8]

    # get a sequence of units corresponding to the start subtokens of the words
    # size: [n_words, n_features]
    def gather_nd(params, indices):
        assert type(indices) == torch.Tensor
        return params[indices.transpose(0, 1).long().numpy().tolist()]

    elements = gather_nd(units, idxs)

    # prepare zeros for paddings
    # size: [batch_size * TOKEN_seq_length - n_words, n_features]
    sh = tuple(torch.stack([torch.sum(max_token_seq_len - token_seq_lengths), torch.tensor(nf)], 0).numpy())
    paddings = torch.zeros(sh, dtype=torch.float64)

    def dynamic_stitch(indices, data):
        # https://discuss.pytorch.org/t/equivalent-of-tf-dynamic-partition/53735/2
        n = sum(idx.numel() for idx in indices)
        res = [None] * n
        for i, data_ in enumerate(data):
            idx = indices[i].view(-1)
            if idx.numel() > 0:
                d = data_.view(idx.numel(), -1)
                k = 0
                for idx_ in idx:
                    res[idx_] = d[k].to(torch.float64)
                    k += 1
        return res

    tensor_flat = torch.stack(dynamic_stitch([word_indices_flat, nonword_indices_flat], [elements, paddings]))
    # tensor_flat -> [x, x, x, x, x, 0, x, 0, 0]

    tensor = torch.reshape(tensor_flat, (batch_size, max_token_seq_len.item(), nf_int))
    # tensor -> [[x, x, x],
    #            [x, x, 0],
    #            [x, 0, 0]]

    return tensor


def token_labels_to_subtoken_labels(labels, y_mask, input_mask):
    subtoken_labels = []
    labels_ind = 0
    n_tokens_with_special = int(np.sum(input_mask))

    for el in y_mask[1:n_tokens_with_special - 1]:
        if el == 1:
            subtoken_labels += [labels[labels_ind]]
            labels_ind += 1
        else:
            subtoken_labels += [labels[labels_ind - 1]]

    subtoken_labels = [0] + subtoken_labels + [0] * (len(input_mask) - n_tokens_with_special + 1)
    return subtoken_labels


def token_labels_to_subtoken_labels_spans(start_labels, end_labels, y_mask, input_mask):
    labels_ind = 0
    subtoken_labels_start = [0 for _ in input_mask]
    if any([el == 1 for el in start_labels]):
        for n, el in enumerate(y_mask):
            if el == 1:
                subtoken_labels_start[n] = start_labels[labels_ind]
                labels_ind += 1
    labels_ind = 0
    subtoken_labels_end = [0 for _ in input_mask]
    if any([el == 1 for el in end_labels]):
        for n, el in enumerate(y_mask):
            if el == 1:
                found_x = -1
                for x in range(n, len(y_mask)):
                    if x == len(y_mask) - 1 or y_mask[x + 1] == 1:
                        found_x = x
                        break
                subtoken_labels_end[found_x] = end_labels[labels_ind]
                labels_ind += 1
    
    return subtoken_labels_start, subtoken_labels_end


def token_from_subtoken_spans(start_logits, end_logits, y_masks, input_masks):
    start_logits = start_logits.detach().cpu().numpy().tolist()
    end_logits = end_logits.detach().cpu().numpy().tolist()
    y_masks = y_masks.tolist()
    input_masks = input_masks.tolist()
    token_start_preds, token_end_preds, subtok_to_tok = [], [], []
    for start_logits_elem, end_logits_elem, y_masks_elem, input_masks_elem in \
            zip(start_logits, end_logits, y_masks, input_masks):
        subtok_to_tok_dict = {}
        tok_ind = 0
        cur_length = int(sum(input_masks_elem))
        cur_y_masks_elem = y_masks_elem[1:cur_length]
        for n in range(len(cur_y_masks_elem)):
            subtok_to_tok_dict[n] = tok_ind
            if n == len(cur_y_masks_elem) - 1 or cur_y_masks_elem[n + 1] == 1:
                tok_ind += 1
        token_start_preds_elem = [0 for _ in range(cur_length)]
        token_end_preds_elem = [0 for _ in range(cur_length)]
        for n, el in enumerate(start_logits_elem[1:cur_length]):
            token_start_preds_elem[subtok_to_tok_dict[n]] = max(token_start_preds_elem[subtok_to_tok_dict[n]], el)
        for n, el in enumerate(end_logits_elem[1:cur_length]):
            token_end_preds_elem[subtok_to_tok_dict[n]] = max(token_end_preds_elem[subtok_to_tok_dict[n]], el)
        token_start_preds.append(token_start_preds_elem)
        token_end_preds.append(token_end_preds_elem)
        subtok_to_tok.append(subtok_to_tok_dict)
        
    return token_start_preds, token_end_preds, subtok_to_tok


class AutoModelForTokenClassificationSpans(nn.Module):
    def __init__(self, pretrained_bert: str, num_classes_tags: int, bert_config_file: Optional[str] = None,
                       dropout: float = 0.1, smooth: float = 1e-4, alpha: float = 0.0, use_o_tag: bool = False):
        super().__init__()
        self.pretrained_bert = pretrained_bert
        if self.pretrained_bert:
            config = AutoConfig.from_pretrained(self.pretrained_bert, output_attentions=False,
                                                output_hidden_states=False)
            self.encoder = AutoModel.from_pretrained(self.pretrained_bert, config=config)
        elif bert_config_file and Path(bert_config_file).is_file():
            self.bert_config = AutoConfig.from_json_file(str(expand_path(bert_config_file)))
            self.encoder = AutoModel(config=self.bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")
        
        self.dropout = nn.Dropout(dropout)
        self.use_o_tag = use_o_tag
        if self.use_o_tag:
            self.num_classes_tags = num_classes_tags
        else:
            self.num_classes_tags = num_classes_tags - 1
        self.classifier_tags = nn.Linear(config.hidden_size, self.num_classes_tags)
        self.classifier_spans = nn.Linear(config.hidden_size, 2)
        self.smooth = smooth
        self.alpha = alpha
    
    def forward(self, input_ids, attention_mask, labels_tags=None, start_labels=None, end_labels=None):
        transf_output = self.encoder(input_ids, attention_mask, output_attentions=True)
        output = transf_output.last_hidden_state
        output = self.dropout(output)
        bs, seq_len, emb = output.size()
        
        logits_tags = self.classifier_tags(output)
        logits_spans = self.classifier_spans(output)
        
        start_logits, end_logits = logits_spans.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        
        loss = None
        if labels_tags is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            active_logits_tags = logits_tags.view(-1, self.num_classes_tags)
            if self.use_o_tag:
                new_labels_tags = labels_tags
                active_loss_tags = attention_mask.view(-1) == 1
            else:
                new_labels_tags = labels_tags - 1
                active_loss_tags = labels_tags.view(-1) > 0
            active_labels_tags = torch.where(
                active_loss_tags, new_labels_tags.view(-1), torch.tensor(loss_fct.ignore_index).type_as(new_labels_tags)
            )
            loss_tags = loss_fct(active_logits_tags, active_labels_tags)
            
            loss_spans = self.compute_dice_loss_total(start_logits, end_logits, start_labels, end_labels, attention_mask)
            loss = loss_tags + loss_spans
            return loss
        else:
            start_logits = torch.sigmoid(start_logits)
            end_logits = torch.sigmoid(end_logits)
            return logits_tags, start_logits, end_logits
    
    def compute_dice_loss_total(self, start_logits, end_logits, start_labels, end_labels, attention_mask):
        start_loss = self.compute_dice_loss(start_logits, start_labels, attention_mask)
        end_loss = self.compute_dice_loss(end_logits, end_labels, attention_mask)
        total_loss = start_loss + end_loss
        return total_loss
    
    def compute_dice_loss(self, inp, target, mask):
        flat_input = inp.view(-1)
        flat_target = target.view(-1).float()
        flat_input = torch.sigmoid(flat_input)
        mask = mask.view(-1)

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        loss = self.compute_dice_loss_elem(flat_input, flat_target)
        return loss
    
    def compute_dice_loss_elem(self, flat_input, flat_target):
        flat_input = ((1 - flat_input) ** self.alpha) * flat_input
        interection = torch.sum(flat_input * flat_target, -1)
        loss = 1 - ((2 * interection + self.smooth) /
                    (flat_input.sum() + flat_target.sum() + self.smooth))
        return loss


@register('torch_transformers_sequence_tagger_spans')
class TorchTransformersSequenceTaggerSpans(TorchModel):
    """Transformer-based model on PyTorch for text tagging. It predicts a label for every token (not subtoken)
    in the text. You can use it for sequence labeling tasks, such as morphological tagging or named entity recognition.

    Args:
        n_tags: number of distinct tags
        pretrained_bert: pretrained Bert checkpoint path or key title (e.g. "bert-base-uncased")
        return_probas: set this to `True` if you need the probabilities instead of raw answers
        bert_config_file: path to Bert configuration file, or None, if `pretrained_bert` is a string name
        attention_probs_keep_prob: keep_prob for Bert self-attention layers
        hidden_keep_prob: keep_prob for Bert hidden layers
        optimizer: optimizer name from `torch.optim`
        optimizer_parameters: dictionary with optimizer's parameters,
                              e.g. {'lr': 0.1, 'weight_decay': 0.001, 'momentum': 0.9}
        learning_rate_drop_patience: how many validations with no improvements to wait
        learning_rate_drop_div: the divider of the learning rate after `learning_rate_drop_patience` unsuccessful
            validations
        load_before_drop: whether to load best model before dropping learning rate or not
        clip_norm: clip gradients by norm
        min_learning_rate: min value of learning rate if learning rate decay is used
    """

    def __init__(self,
                 num_classes_tags: int,
                 pretrained_bert: str,
                 bert_config_file: Optional[str] = None,
                 return_probas_tag: bool = False,
                 return_probas_span: bool = False,
                 attention_probs_keep_prob: Optional[float] = None,
                 hidden_keep_prob: Optional[float] = None,
                 optimizer: str = "AdamW",
                 optimizer_parameters: dict = {"lr": 1e-3, "weight_decay": 1e-6},
                 head_lr: float = None,
                 init_lr: float = None,
                 num_warmup_steps: int = 50,
                 learning_rate_drop_patience: int = 20,
                 learning_rate_drop_div: float = 2.0,
                 load_before_drop: bool = True,
                 clip_norm: Optional[float] = None,
                 min_learning_rate: float = 1e-07,
                 use_o_tag: bool = False,
                 **kwargs) -> None:

        self.num_classes_tags = num_classes_tags
        self.return_probas_tag = return_probas_tag
        self.return_probas_span = return_probas_span
        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob
        self.clip_norm = clip_norm

        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file
        self.min_learning_rate = min_learning_rate
        self.head_lr = head_lr
        self.init_lr = init_lr
        self.num_warmup_steps = num_warmup_steps
        self.use_o_tag = use_o_tag

        super().__init__(optimizer=optimizer,
                         optimizer_parameters=optimizer_parameters,
                         learning_rate_drop_patience=learning_rate_drop_patience,
                         learning_rate_drop_div=learning_rate_drop_div,
                         load_before_drop=load_before_drop,
                         min_learning_rate=min_learning_rate,
                         **kwargs)

    def train_on_batch(self,
                       input_ids: Union[List[List[int]], np.ndarray],
                       input_masks: Union[List[List[int]], np.ndarray],
                       y_masks: Union[List[List[int]], np.ndarray],
                       y_tags: List[List[int]],
                       start_labels: List[List[int]],
                       end_labels: List[List[int]],
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
        subtoken_labels_tags = [token_labels_to_subtoken_labels(y_el, y_mask, input_mask)
                                for y_el, y_mask, input_mask in zip(y_tags, y_masks, input_masks)]
        b_labels_tags = torch.from_numpy(np.array(subtoken_labels_tags)).to(torch.int64).to(self.device)
        b_labels_start, b_labels_end = [], []
        for start_labels_el, end_labels_el, y_mask, input_mask in zip(start_labels, end_labels, y_masks, input_masks):
            subtoken_labels_start, subtoken_labels_end = \
                token_labels_to_subtoken_labels_spans(start_labels_el, end_labels_el, y_mask, input_mask)
            b_labels_start.append(subtoken_labels_start)
            b_labels_end.append(subtoken_labels_end)
        
        b_labels_start = torch.Tensor(b_labels_start).to(self.device)
        b_labels_end = torch.Tensor(b_labels_end).to(self.device)
        
        self.optimizer.zero_grad()

        loss = self.model(input_ids=b_input_ids, attention_mask=b_input_masks, labels_tags=b_labels_tags,
                          start_labels=b_labels_start, end_labels=b_labels_end)
        if isinstance(self.model, torch.nn.DataParallel):
            loss = loss.mean()
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
            logits_tags, start_logits, end_logits = self.model(b_input_ids, attention_mask=b_input_masks)

            # Move logits and labels to CPU and to numpy arrays
            logits_tags = token_from_subtoken(logits_tags.detach().cpu(), torch.from_numpy(y_masks))
            token_start_preds, token_end_preds, subtok_to_tok = token_from_subtoken_spans(start_logits, end_logits, y_masks, input_masks)

        seq_lengths = np.sum(y_masks, axis=1)
        if self.return_probas_tag:
            pred_tags = torch.nn.functional.softmax(logits_tags, dim=-1)
            pred_tags = pred_tags.detach().cpu().numpy()
        else:
            logits_tags = logits_tags.detach().cpu().numpy()
            pred_tags = np.argmax(logits_tags, axis=-1).tolist()
        pred_tags = [p[:l] for l, p in zip(seq_lengths, pred_tags)]
        
        if self.return_probas_span:
            pred_start = start_logits
            pred_end = end_logits
        else:
            pred_start = token_start_preds
            pred_end = token_end_preds
        pred_start = [p[:l] for l, p in zip(seq_lengths, pred_start)]
        pred_end = [p[:l] for l, p in zip(seq_lengths, pred_end)]

        return pred_tags, pred_start, pred_end

    @overrides
    def load(self, fname=None):
        if fname is not None:
            self.load_path = fname

        if self.pretrained_bert:
            self.model = AutoModelForTokenClassificationSpans(self.pretrained_bert, self.num_classes_tags,
                                                              self.bert_config_file, use_o_tag=self.use_o_tag)
        
        if self.head_lr is not None:
            opt_parameters = []
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            lr = self.init_lr
            named_parameters = list(self.model.named_parameters())
            params_0 = [p for n,p in named_parameters if "classifier" in n and any(nd in n for nd in no_decay)]
            params_1 = [p for n,p in named_parameters if "classifier" in n and not any(nd in n for nd in no_decay)]
            head_params = {"params": params_0, "lr": self.head_lr, "weight_decay": 0.0}    
            opt_parameters.append(head_params)
            head_params = {"params": params_1, "lr": self.head_lr, "weight_decay": 0.01}    
            opt_parameters.append(head_params)
            for layer in range(11, -1, -1):        
                params_0 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n 
                            and any(nd in n for nd in no_decay)]
                params_1 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n 
                            and not any(nd in n for nd in no_decay)]
                layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
                opt_parameters.append(layer_params)       
                layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
                opt_parameters.append(layer_params)       
                lr *= 0.9
    
            params_0 = [p for n, p in named_parameters if "embeddings" in n 
                        and any(nd in n for nd in no_decay)]
            params_1 = [p for n, p in named_parameters if "embeddings" in n
                        and not any(nd in n for nd in no_decay)]
            embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
            opt_parameters.append(embed_params)
            embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01} 
            opt_parameters.append(embed_params)
            
            self.optimizer = getattr(torch.optim, self.optimizer_name)(
                opt_parameters, **self.optimizer_parameters)  
            self.lr_scheduler = transformers.get_constant_schedule_with_warmup(self.optimizer, 
                                                         num_warmup_steps=50)
        else:
            self.optimizer = getattr(torch.optim, self.optimizer_name)(
                self.model.parameters(), **self.optimizer_parameters)
        if self.lr_scheduler_name is not None:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                self.optimizer, **self.lr_scheduler_parameters)
        
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

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
                if isinstance(self.model, torch.nn.DataParallel):
                    self.model.module.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.epochs_done = checkpoint.get("epochs_done", 0)
            else:
                log.info(f"Init from scratch. Load path {weights_path} does not exist.")
