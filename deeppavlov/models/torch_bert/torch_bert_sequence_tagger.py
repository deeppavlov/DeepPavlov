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


def token_from_subtoken(units: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """ Assemble token level units from subtoken level units

    Args:
        units: tf.Tensor of shape [batch_size, SUBTOKEN_seq_length, n_features]
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
    shape = tf.cast(tf.shape(units), tf.int64)
    batch_size = shape[0]
    nf = shape[2]
    nf_int = units.get_shape().as_list()[-1]

    # number of TOKENS in each sentence
    token_seq_lengths = tf.cast(tf.reduce_sum(mask, 1), tf.int64)
    # for a matrix m =
    # [[1, 1, 1],
    #  [0, 1, 1],
    #  [1, 0, 0]]
    # it will be
    # [3, 2, 1]

    n_words = tf.reduce_sum(token_seq_lengths)
    # n_words -> 6

    max_token_seq_len = tf.cast(tf.reduce_max(token_seq_lengths), tf.int64)
    # max_token_seq_len -> 3

    idxs = tf.where(mask)
    # for the matrix mentioned above
    # tf.where(mask) ->
    # [[0, 0],
    #  [0, 1]
    #  [0, 2],
    #  [1, 1],
    #  [1, 2]
    #  [2, 0]]

    sample_ids_in_batch = tf.pad(idxs[:, 0], [[1, 0]])
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

    a = tf.cast(tf.not_equal(sample_ids_in_batch[1:], sample_ids_in_batch[:-1]), tf.int64)
    # for the example above the result of this statement equals
    # [0, 0, 0, 1, 0, 1]
    # so data samples begin in 3rd and 5th positions (the indexes of ones)

    # transforming sample start masks to the sample starts themselves
    q = a * tf.cast(tf.range(n_words), tf.int64)
    # [0, 0, 0, 3, 0, 5]
    count_to_substract = tf.pad(tf.boolean_mask(q, q), [(1, 0)])
    # [0, 3, 5]

    new_word_indices = tf.cast(tf.range(n_words), tf.int64) - tf.gather(count_to_substract, tf.cumsum(a))
    # tf.range(n_words) -> [0, 1, 2, 3, 4, 5]
    # tf.cumsum(a) -> [0, 0, 0, 1, 1, 2]
    # tf.gather(count_to_substract, tf.cumsum(a)) -> [0, 0, 0, 3, 3, 5]
    # new_word_indices -> [0, 1, 2, 3, 4, 5] - [0, 0, 0, 3, 3, 5] = [0, 1, 2, 0, 1, 0]
    # new_word_indices is the concatenation of range(word_len(sentence))
    # for all sentences in units

    n_total_word_elements = tf.cast(batch_size * max_token_seq_len, tf.int32)
    word_indices_flat = tf.cast(idxs[:, 0] * max_token_seq_len + new_word_indices, tf.int32)
    x_mask = tf.reduce_sum(tf.one_hot(word_indices_flat, n_total_word_elements), 0)
    x_mask = tf.cast(x_mask, tf.bool)
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

    full_range = tf.cast(tf.range(batch_size * max_token_seq_len), tf.int32)
    # full_range -> [0, 1, 2, 3, 4, 5, 6, 7, 8]
    nonword_indices_flat = tf.boolean_mask(full_range, tf.math.logical_not(x_mask))
    # # y_idxs -> [5, 7, 8]

    # get a sequence of units corresponding to the start subtokens of the words
    # size: [n_words, n_features]
    elements = tf.gather_nd(units, idxs)

    # prepare zeros for paddings
    # size: [batch_size * TOKEN_seq_length - n_words, n_features]
    paddings = tf.zeros(tf.stack([tf.reduce_sum(max_token_seq_len - token_seq_lengths),
                                  nf], 0), tf.float32)

    tensor_flat = tf.dynamic_stitch([word_indices_flat, nonword_indices_flat],
                                    [elements, paddings])
    # tensor_flat -> [x, x, x, x, x, 0, x, 0, 0]

    tensor = tf.reshape(tensor_flat, tf.stack([batch_size, max_token_seq_len, nf_int], 0))
    # tensor -> [[x, x, x],
    #            [x, x, 0],
    #            [x, 0, 0]]

    return tensor


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
                 bert_config_file: str,
                 return_probas: bool = False,
                 pretrained_bert: str = None,
                 attention_probs_keep_prob: float = None,
                 hidden_keep_prob: float = None,
                 encoder_layer_ids: List[int] = (-1,),
                 encoder_dropout: float = 0.0,
                 optimizer: str = None,
                 weight_decay_rate: float = 1e-6,
                 freeze_embeddings: bool = False,
                 learning_rate: float = 1e-3,
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
        self.weight_decay_rate = weight_decay_rate
        self.freeze_embeddings = freeze_embeddings
        self.bert_learning_rate_multiplier = bert_learning_rate / learning_rate
        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob

        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file

        super().__init__(optimizer=optimizer,
                         optimizer_parameters={"lr": learning_rate},
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
        b_input_ids = torch.cat(input_ids, dim=0).to(self.device)
        b_input_masks = torch.cat(input_masks, dim=0).to(self.device)
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
        b_input_ids = torch.cat(input_ids, dim=0).to(self.device)
        b_input_masks = torch.cat(input_masks, dim=0).to(self.device)

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
