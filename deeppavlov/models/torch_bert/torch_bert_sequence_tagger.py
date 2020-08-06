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
from overrides import overrides
from transformers import BertForTokenClassification, BertConfig

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.commands.utils import expand_path
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


@register('torch_bert_sequence_tagger')
class TorchBertSequenceTagger(TorchModel):
    """BERT-based model on PyTorch for text tagging. It predicts a label for every token (not subtoken) in the text.
    You can use it for sequence labeling tasks, such as morphological tagging or named entity recognition.

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
                 n_tags: int,
                 pretrained_bert: str,
                 bert_config_file: Optional[str] = None,
                 return_probas: bool = False,
                 attention_probs_keep_prob: Optional[float] = None,
                 hidden_keep_prob: Optional[float] = None,
                 optimizer: str = "AdamW",
                 optimizer_parameters: dict = {"lr": 1e-3, "weight_decay": 1e-6},
                 learning_rate_drop_patience: int = 20,
                 learning_rate_drop_div: float = 2.0,
                 load_before_drop: bool = True,
                 clip_norm: Optional[float] = None,
                 min_learning_rate: float = 1e-07,
                 **kwargs) -> None:

        self.n_classes = n_tags
        self.return_probas = return_probas
        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob
        self.clip_norm = clip_norm

        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file

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
        subtoken_labels = [token_labels_to_subtoken_labels(y_el, y_mask, input_mask)
                           for y_el, y_mask, input_mask in zip(y, y_masks, input_masks)]
        b_labels = torch.from_numpy(np.array(subtoken_labels)).to(torch.int64).to(self.device)
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
            logits = token_from_subtoken(logits[0].detach().cpu(), torch.from_numpy(y_masks))

        if self.return_probas:
            pred = torch.nn.functional.softmax(logits, dim=-1)
            pred = pred.detach().cpu().numpy()
        else:
            logits = logits.detach().cpu().numpy()
            pred = np.argmax(logits, axis=-1)
            seq_lengths = np.sum(y_masks, axis=1)
            pred = [p[:l] for l, p in zip(seq_lengths, pred)]

        return pred

    @overrides
    def load(self, fname=None):
        if fname is not None:
            self.load_path = fname

        if self.pretrained_bert and not Path(self.pretrained_bert).is_file():
            self.model = BertForTokenClassification.from_pretrained(
                self.pretrained_bert, num_labels=self.n_classes,
                output_attentions=False, output_hidden_states=False)
        elif self.bert_config_file and Path(self.bert_config_file).is_file():
            self.bert_config = BertConfig.from_json_file(str(expand_path(self.bert_config_file)))

            if self.attention_probs_keep_prob is not None:
                self.bert_config.attention_probs_dropout_prob = 1.0 - self.attention_probs_keep_prob
            if self.hidden_keep_prob is not None:
                self.bert_config.hidden_dropout_prob = 1.0 - self.hidden_keep_prob
            self.model = BertForTokenClassification(config=self.bert_config)
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
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.epochs_done = checkpoint.get("epochs_done", 0)
            else:
                log.info(f"Init from scratch. Load path {weights_path} does not exist.")
