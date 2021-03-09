from typing import List, Tuple, Union, Dict, Optional

import numpy as np
import tensorflow as tf

from deeppavlov.core.common.check_gpu import check_gpu_existence
from deeppavlov.core.common.registry import register
from deeppavlov.core.layers.tf_layers import variational_dropout, bi_rnn
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.tf_model import LRScheduledTFModel
from deeppavlov.models.embedders.abstract_embedder import Embedder
from deeppavlov.models.squad.utils import CudnnGRU, CudnnCompatibleGRU, softmax_mask


@register('two_sentences_emb')
class TwoSentencesEmbedder(Component):
    """This class is used for embedding of two sentences."""

    def __init__(self, embedder: Embedder, **kwargs):
        """

        Args:
            embedder: what embedder to use: Glove, Fasttext or other
            **kwargs:
        """
        self.embedder = embedder

    def __call__(self, sentence_tokens_1: List[List[str]], sentence_tokens_2: List[List[str]]) -> \
            Tuple[List[Union[list, np.ndarray]], List[Union[list, np.ndarray]]]:
        sentence_token_embs_1 = self.embedder(sentence_tokens_1)
        sentence_token_embs_2 = self.embedder(sentence_tokens_2)
        return sentence_token_embs_1, sentence_token_embs_2


@register('fact_ranker')
class FactRanker(LRScheduledTFModel):
    """
        This class determines whether the relation appropriate for the question or not.
    """

    def __init__(self, n_classes: int = 2,
                 dropout_keep_prob: float = 0.5,
                 input_dim: int = 300,
                 bigru_num_units: int = 75,
                 return_probas: bool = False, **kwargs):
        """

        Args:
            n_classes: number of classes for classification
            dropout_keep_prob: Probability of keeping the hidden state, values from 0 to 1. 0.5 works well
                in most cases.
            return_probas: whether to return confidences of the relation to be appropriate or not
            **kwargs:
        """
        kwargs.setdefault('learning_rate_drop_div', 10.0)
        kwargs.setdefault('learning_rate_drop_patience', 5.0)
        kwargs.setdefault('clip_norm', 5.0)

        super().__init__(**kwargs)

        self.n_classes = n_classes
        self.dropout_keep_prob = dropout_keep_prob
        self.input_dim = input_dim
        self.bigru_num_units = bigru_num_units
        self.return_probas = return_probas
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        if check_gpu_existence():
            self.GRU = CudnnGRU
        else:
            self.GRU = CudnnCompatibleGRU

        self.sentence1_ph = tf.placeholder(tf.float32, [None, None, self.input_dim])
        self.sentence2_ph = tf.placeholder(tf.float32, [None, None, self.input_dim])
        self.y_ph = tf.placeholder(tf.int32, shape=(None,))
        
        self.one_hot_labels = tf.one_hot(self.y_ph, depth=self.n_classes, dtype=tf.float32)
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph')

        sent1_mask_2 = tf.cast(self.sentence1_ph, tf.bool)
        sent1_len_2 = tf.reduce_sum(tf.cast(sent1_mask_2, tf.int32), axis=2)
        sent1_mask = tf.cast(sent1_len_2, tf.bool)
        sent1_len = tf.reduce_sum(tf.cast(sent1_mask, tf.int32), axis=1)
        
        sent2_mask_2 = tf.cast(self.sentence2_ph, tf.bool)
        sent2_len_2 = tf.reduce_sum(tf.cast(sent2_mask_2, tf.int32), axis=2)
        sent2_mask = tf.cast(sent2_len_2, tf.bool)
        sent2_len = tf.reduce_sum(tf.cast(sent2_mask, tf.int32), axis=1)

        sent1_dr = variational_dropout(self.sentence1_ph, keep_prob=self.keep_prob_ph)
        sent2_dr = variational_dropout(self.sentence2_ph, keep_prob=self.keep_prob_ph)
        b_size = tf.shape(self.sentence1_ph)[0]

        with tf.variable_scope("question_encode"):
            rnn1 = self.GRU(num_layers=2, num_units=self.bigru_num_units, batch_size=b_size,
                            input_size=self.input_dim, keep_prob=self.keep_prob_ph)
            rnn2 = self.GRU(num_layers=2, num_units=self.bigru_num_units, batch_size=b_size,
                            input_size=self.input_dim, keep_prob=self.keep_prob_ph)
            s1 = rnn1(sent1_dr, seq_len=sent1_len)
            s2 = rnn2(sent2_dr, seq_len=sent2_len)

        with tf.variable_scope("attention"):
            s12_att = dot_attention(s1, s2, mask=self.sent1_mask, att_size=100,
                                   keep_prob=self.keep_prob_ph, concat=True, gate=True)
                                   
            _, (fw_st, bw_st) = bi_rnn(s12_att, n_hidden=100, seq_lengths=sent1_len)
            
            v = tf.concat([fw_st, bw_st], axis=2)

            self.logits = tf.layers.dense(v, 2, activation=None, use_bias=False)
            self.y_pred = tf.argmax(self.logits, axis=-1)

            loss_tensor = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.one_hot_labels, logits=self.logits)

            self.loss = tf.reduce_mean(loss_tensor)
            self.train_op = self.get_train_op(self.loss)

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.load()

    def fill_feed_dict(self, sentence1_embs: List[np.ndarray],
                             sentence2_embs: List[np.ndarray], y=None, train=False) -> \
            Dict[tf.placeholder, List[np.ndarray]]:
        sentence1_embs = np.array(sentence1_embs)
        sentence2_embs = np.array(sentence2_embs)
        feed_dict = {self.sentence1_ph: sentence1_embs, self.sentence2_ph: sentence2_embs}
        if y is not None:
            feed_dict[self.y_ph] = y
        if train:
            feed_dict[self.keep_prob_ph] = self.dropout_keep_prob
        else:
            feed_dict[self.keep_prob_ph] = 1.0

        return feed_dict

    def __call__(self, sentence1_embs: List[np.ndarray], sentence2_embs: List[np.ndarray]) -> \
            List[np.ndarray]:
        feed_dict = self.fill_feed_dict(sentence1_embs, sentence2_embs)
        if self.return_probas:
            pred = self.sess.run(self.logits, feed_dict)
        else:
            pred = self.sess.run(self.y_pred, feed_dict)
        return pred

    def train_on_batch(self, sentence1_embs: List[np.ndarray], 
                             sentence2_embs: List[np.ndarray],
                             y: List[int]) -> Dict[str, float]:
        feed_dict = self.fill_feed_dict(sentence1_embs, sentence2_embs, y, train=True)
        _, loss_value = self.sess.run([self.train_op, self.loss], feed_dict)

        return {'loss': loss_value,
                'learning_rate': self.get_learning_rate(),
                'momentum': self.get_momentum()}
