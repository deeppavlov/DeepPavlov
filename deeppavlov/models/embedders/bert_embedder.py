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
from typing import List, Union, Tuple, Dict

import numpy as np
import tensorflow.compat.v1 as tf
from bert_dp.modeling import BertConfig, BertModel

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import TFModel

log = getLogger(__name__)


@register('bert_embedder')
class BertEmbedder(TFModel):
    """BERT model for embedding tokens, subtokens and sentences.

    For each token a tag is predicted. Can be used for any tagging.

    Args:
        bert_config_path: path to Bert configuration file
        load_path: pretrained Bert checkpoint
        encoder_layer_ids: list of averaged layers from Bert encoder
            (layer indeces)
    """

    def __init__(self,
                 bert_config_path: str,
                 load_path: str,
                 level: str = 'all',
                 encoder_layer_ids: List[int] = (-1,),
                 **kwargs) -> None:
        super().__init__(load_path=load_path, save_path=None, **kwargs)

        self.encoder_layer_ids = encoder_layer_ids
        assert level in ('word', 'subword', 'text', 'all'),\
            f"`level` argument should have value of 'word', 'subword', 'text'"\
            f" or 'all', but has value of {level}."
        self.level = level

        bert_config_path = str(expand_path(bert_config_path))
        self.bert_config = BertConfig.from_json_file(bert_config_path)

        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)

        self._init_graph()

        self.sess.run(tf.global_variables_initializer())

        if self.load_path is not None and \
                tf.train.checkpoint_exists(str(self.load_path)):
            log.info(f"[initializing Bert embedder from {self.load_path}]")
            self.load()
        else:
            log.info(f"[initializing Bert embedder from scratch]")

    def _init_graph(self) -> None:
        self._init_placeholders()
        self.bert = BertModel(config=self.bert_config,
                              is_training=self.is_train_ph,
                              input_ids=self.input_ids_ph,
                              input_mask=self.input_masks_ph,
                              token_type_ids=self.token_types_ph,
                              use_one_hot_embeddings=False)
        encoder_layer = tf.reduce_mean([self.bert.all_encoder_layers[i]
                                        for i in self.encoder_layer_ids],
                                       axis=0)

        self.subword_seq_lengths = tf.reduce_sum(self.input_masks_ph, axis=1)
        self.subword_embs = encoder_layer
        self.word_seq_lengths = \
            tf.reduce_sum(self.startofword_markers_ph, axis=1)
        self.word_embs = \
            self.token_from_subtoken(encoder_layer, self.startofword_markers_ph)

        self.cls_emb = self.subword_embs[:, 0]

        idxs = tf.expand_dims(self.subword_seq_lengths - 1, 1)
        self.sep_emb = tf.gather_nd(self.subword_embs, idxs, batch_dims=1)

        mask_expanded = \
            tf.expand_dims(tf.cast(self.input_masks_ph, tf.float32), axis=2)
        self.max_pool_emb = \
            tf.reduce_max(encoder_layer - 1e9 * (1 - mask_expanded), axis=1)
        self.mean_pool_emb = \
            tf.reduce_sum(encoder_layer * mask_expanded, axis=1) /\
            tf.reduce_sum(mask_expanded, axis=1)

    def _init_placeholders(self) -> None:
        self.input_ids_ph = tf.placeholder(shape=(None, None),
                                           dtype=tf.int32,
                                           name='token_indices_ph')
        self.input_masks_ph = tf.placeholder(shape=(None, None),
                                             dtype=tf.int32,
                                             name='token_mask_ph')
        self.token_types_ph = \
            tf.placeholder_with_default(tf.zeros_like(self.input_ids_ph,
                                                      dtype=tf.int32),
                                        shape=self.input_ids_ph.shape,
                                        name='token_types_ph')

        self.startofword_markers_ph = tf.placeholder(shape=(None, None),
                                                     dtype=tf.int32,
                                                     name='y_mask_ph')

        self.is_train_ph = \
            tf.placeholder_with_default(False, shape=[], name='is_train_ph')

    @staticmethod
    def token_from_subtoken(units: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """ Assemble token level units from subtoken level units

        Args:
            units: tf.Tensor of shape [batch_size, SUBTOKEN_seq_length, n_features]
            mask: mask of startings of new tokens. Example: for tokens

                    [[`[CLS]` `My`, `capybara`, `[SEP]`],
                    [`[CLS]` `Your`, `aar`, `##dvark`, `is`, `awesome`, `[SEP]`]]

                the mask will be

                    [[0, 1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 1, 1, 0]]

        Returns:
            word_level_units: Units assembled from ones in the mask. For the
                example above this units will correspond to the following

                    [[`My`, `capybara`],
                    [`Your`, `aar`, `is`, `awesome`,]]

                the shape of this thesor will be [batch_size, TOKEN_seq_length, n_features]
        """
        shape = tf.cast(tf.shape(units), tf.int64)
        bs = shape[0]
        nf = shape[2]
        nf_int = units.get_shape().as_list()[-1]

        # numer of TOKENS in each sentence
        token_seq_lenghs = tf.cast(tf.reduce_sum(mask, 1), tf.int64)
        # for a matrix m =
        # [[1, 1, 1],
        #  [0, 1, 1],
        #  [1, 0, 0]]
        # it will be
        # [3, 2, 1]

        n_words = tf.reduce_sum(token_seq_lenghs)
        # n_words -> 6

        max_token_seq_len = tf.reduce_max(token_seq_lenghs)
        max_token_seq_len = tf.cast(max_token_seq_len, tf.int64)
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

        sample_id_in_batch = tf.pad(idxs[:, 0], [[1, 0]])
        # for indices
        # [[0, 0],
        #  [0, 1]
        #  [0, 2],
        #  [1, 1],
        #  [1, 2],
        #  [2, 0]]
        # it will be
        # [0, 0, 0, 0, 1, 1, 2]
        # padding is for computing change from one sample to another in the batch

        a = tf.cast(tf.not_equal(sample_id_in_batch[1:], sample_id_in_batch[:-1]), tf.int64)
        # for the example above the result of this line will be
        # [0, 0, 0, 1, 0, 1]
        # so the number of the sample in batch changes only in the last word element

        q = a * tf.cast(tf.range(n_words), tf.int64)
        # [0, 0, 0, 3, 0, 5]

        count_to_substract = tf.pad(tf.boolean_mask(q, q), [(1, 0)])
        # [0, 3, 5]

        new_word_indices = tf.cast(tf.range(n_words), tf.int64) - tf.gather(count_to_substract, tf.cumsum(a))
        # tf.range(n_words) -> [0, 1, 2, 3, 4, 5]
        # tf.cumsum(a) -> [0, 0, 0, 1, 1, 2]
        # tf.gather(count_to_substract, tf.cumsum(a)) -> [0, 0, 0, 3, 3, 5]
        # new_word_indices -> [0, 1, 2, 3, 4, 5] - [0, 0, 0, 3, 3, 5] = [0, 1, 2, 0, 1, 0]
        # this is new indices token dimension

        n_total_word_elements = tf.cast(bs * max_token_seq_len, tf.int32)
        x_mask = tf.reduce_sum(tf.one_hot(idxs[:, 0] * max_token_seq_len + new_word_indices, n_total_word_elements), 0)
        x_mask = tf.cast(x_mask, tf.bool)
        # to get absolute indices we add max_token_seq_len:
        # idxs[:, 0] * max_token_seq_len -> [0, 0, 0, 1, 1, 2] * 2 = [0, 0, 0, 3, 3, 6]
        # idxs[:, 0] * max_token_seq_len + new_word_indices ->
        # [0, 0, 0, 3, 3, 6] + [0, 1, 2, 0, 1, 0] = [0, 1, 2, 3, 4, 6]
        # total number of words in the batch (including paddings)
        # bs * max_token_seq_len -> 3 * 2 = 6
        # tf.one_hot(...) ->
        # [[1. 0. 0. 0. 0. 0. 0. 0. 0.]
        #  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
        #  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
        #  [0. 0. 0. 1. 0. 0. 0. 0. 0.]
        #  [0. 0. 0. 0. 1. 0. 0. 0. 0.]
        #  [0. 0. 0. 0. 0. 0. 1. 0. 0.]]
        #  x_mask -> [1, 1, 1, 1, 1, 0, 1, 0, 0]

        # full_range -> [0, 1, 2, 3, 4, 5, 6, 7, 8]
        full_range = tf.cast(tf.range(bs * max_token_seq_len), tf.int32)

        x_idxs = tf.boolean_mask(full_range, x_mask)
        # x_idxs -> [0, 1, 2, 3, 4, 6]

        y_mask = tf.math.logical_not(x_mask)
        y_idxs = tf.boolean_mask(full_range, y_mask)
        # y_idxs -> [5, 7, 8]

        # get a sequence of units corresponding to the start subtokens of the words
        # size: [n_words, n_features]
        els = tf.gather_nd(units, idxs)

        # prepare zeros for paddings
        # size: [batch_size * TOKEN_seq_length - n_words, n_features]
        paddings = tf.zeros(tf.stack([tf.reduce_sum(max_token_seq_len - token_seq_lenghs),
                                      nf], 0), tf.float32)

        tensor_flat = tf.dynamic_stitch([x_idxs, y_idxs], [els, paddings])
        # tensor_flat -> [x, x, x, x, x, 0, x, 0, 0]

        tensor = tf.reshape(tensor_flat, tf.stack([bs, max_token_seq_len, nf_int], 0))
        # tensor_flat -> [[x, x, x],
        #                 [x, x, 0],
        #                 [x, 0, 0]]

        return tensor

    def _build_feed_dict(self, input_ids, input_masks, y_masks, token_types=None):
        feed_dict = {
            self.input_ids_ph: input_ids,
            self.input_masks_ph: input_masks,
            self.startofword_markers_ph: y_masks
        }
        if token_types is not None:
            feed_dict[self.token_types_ph] = token_types
        return feed_dict

    def train_on_batch(self, **kwargs):
        raise NotImplementedError()

    def __call__(self,
                 tokens: Union[List[List[str]], np.ndarray],
                 subword_tokens: Union[List[List[str]], np.ndarray],
                 input_ids: Union[List[List[int]], np.ndarray],
                 input_masks: Union[List[List[int]], np.ndarray],
                 y_masks: Union[List[List[int]], np.ndarray]) -> Union[Tuple[List[Dict[str, np.ndarray]],
                                                                             List[Dict[str, np.ndarray]],
                                                                             List[Dict[str, np.ndarray]]],
                                                                       List[Dict[str, np.ndarray]]]:
        """ Predicts tag indices for a given subword tokens batch

        Args:
            input_ids: indices of the subwords
            input_masks: mask that determines where to attend and where not to
            y_masks: mask which determines the first subword units in the the word

        Returns:
            Predictions indices or predicted probabilities fro each token (not subtoken)

        """
        feed_dict = self._build_feed_dict(input_ids, input_masks, y_masks)

        if self.level == 'word':
            return self.predict_word_embs(tokens, feed_dict)
        elif self.level == 'subword':
            return self.predict_subword_embs(subword_tokens, feed_dict)
        elif self.level == 'text':
            return self.predict_text_embs(feed_dict)

        return self.predict_all(tokens, subword_tokens, feed_dict)

    def predict_word_embs(self, tokens, feed_dict) -> List[dict]:
        pred, lengths = self.sess.run([self.word_embs,
                                       self.word_seq_lengths],
                                      feed_dict=feed_dict)
        return [{'words': ts,
                 'word_embeddings': np.array(p[:l])}
                for ts, p, l in zip(tokens, pred, lengths)]

    def predict_subword_embs(self, subword_tokens, feed_dict) -> List[Dict[str, np.ndarray]]:
        pred, lengths = self.sess.run([self.subword_embs,
                                       self.subword_seq_lengths],
                                      feed_dict=feed_dict)
        return [{'subwords': ts[1:-1],
                 'subword_embeddings': np.array(p[1:l-1])}
                for ts, p, l in zip(subword_tokens, pred, lengths)]

    def predict_text_embs(self, feed_dict) -> List[Dict[str, np.ndarray]]:
        cls_e, sep_e, max_e, mean_e = \
            self.sess.run([self.cls_emb, self.sep_emb,
                           self.max_pool_emb, self.mean_pool_emb],
                          feed_dict=feed_dict)
        text_embs = [{'[CLS]': cls,
                      '[SEP]': sep,
                      'MAX': mx,
                      'MEAN': mn}
                     for cls, sep, mx, mn in zip(cls_e, sep_e, max_e, mean_e)]
        return text_embs

    def predict_all(self, tokens, subword_tokens, feed_dict) -> Tuple[List[Dict[str, np.ndarray]],
                                                                      List[Dict[str, np.ndarray]],
                                                                      List[Dict[str, np.ndarray]]]:
        sw_emb, sw_lengths, w_emb, w_lengths, cls_e, sep_e, max_e, mean_e = \
            self.sess.run([self.subword_embs, self.subword_seq_lengths,
                           self.word_embs, self.word_seq_lengths,
                           self.cls_emb, self.sep_emb,
                           self.max_pool_emb, self.mean_pool_emb],
                          feed_dict=feed_dict)
        word_embs = [{'words': ts,
                      'word_embeddings': np.array(p[:l])}
                     for ts, p, l in zip(tokens, w_emb, w_lengths)]
        subword_embs = [{'subwords': ts[1:-1],
                         'subword_embeddings': np.array(p[1:l-1])}
                        for ts, p, l in zip(subword_tokens, sw_emb, sw_lengths)]
        text_embs = [{'[CLS]': cls,
                      '[SEP]': sep,
                      'MAX': mx,
                      'MEAN': mn}
                     for cls, sep, mx, mn in zip(cls_e, sep_e, max_e, mean_e)]
        return text_embs, word_embs, subword_embs

    def load(self,
             exclude_scopes=('Optimizer', 'learning_rate', 'momentum'),
             **kwargs) -> None:
        return super().load(exclude_scopes=exclude_scopes, **kwargs)
