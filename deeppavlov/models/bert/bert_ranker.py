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

import tensorflow as tf
from deeppavlov.core.common.registry import register
from deeppavlov.models.bert.bert_classifier import BertClassifierModel
from logging import getLogger

from bert_dp.modeling import BertModel

logger = getLogger(__name__)


@register('bert_ranker')
class BertRankerModel(BertClassifierModel):
    def _init_graph(self):
        self._init_placeholders()

        with tf.variable_scope("model"):
            model_a = BertModel(
                config=self.bert_config,
                is_training=self.is_train_ph,
                input_ids=self.input_masks_ph_a,
                input_mask=self.input_masks_ph_a,
                token_type_ids=self.token_types_ph_a,
                use_one_hot_embeddings=False)

        with tf.variable_scope("model", reuse=True):
            model_b = BertModel(
                config=self.bert_config,
                is_training=self.is_train_ph,
                input_ids=self.input_masks_ph_b,
                input_mask=self.input_masks_ph_b,
                token_type_ids=self.token_types_ph_b,
                use_one_hot_embeddings=False)

        output_layer_a = model_a.get_sequence_output()
        output_layer_b = model_b.get_sequence_output()
        output_layer_a = tf.reduce_max(output_layer_a, axis=1)
        output_layer_b = tf.reduce_max(output_layer_b, axis=1)
        hidden_size = output_layer_a.shape[-1].value

        with tf.variable_scope("W"):
            output_layer_a = tf.layers.dense(
                output_layer_a,
                hidden_size,
                activation=tf.tanh,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

        with tf.variable_scope("W", reuse=True):
            output_layer_b = tf.layers.dense(
                output_layer_b,
                hidden_size,
                activation=tf.tanh,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

        with tf.variable_scope("loss"):

            output_layer_a = tf.nn.dropout(output_layer_a, keep_prob=0.9)
            output_layer_b = tf.nn.dropout(output_layer_b, keep_prob=0.9)
            loss = tf.contrib.losses.metric_learning.npairs_loss(self.y_ph, output_layer_a, output_layer_b)
            logits = tf.multiply(output_layer_a, output_layer_b)
            logits = tf.reduce_sum(logits, 1)


    def _init_placeholders(self):
        self.input_ids_ph_a = tf.placeholder(shape=(None, None), dtype=tf.int32, name='ids_ph_a')
        self.input_masks_ph_a = tf.placeholder(shape=(None, None), dtype=tf.int32, name='masks_ph_a')
        self.token_types_ph_a = tf.placeholder(shape=(None, None), dtype=tf.int32, name='token_types_ph_a')
        self.input_ids_ph_b = tf.placeholder(shape=(None, None), dtype=tf.int32, name='ids_ph_b')
        self.input_masks_ph_b = tf.placeholder(shape=(None, None), dtype=tf.int32, name='masks_ph_b')
        self.token_types_ph_b = tf.placeholder(shape=(None, None), dtype=tf.int32, name='token_types_ph_b')

        if not self.one_hot_labels:
            self.y_ph = tf.placeholder(shape=(None, ), dtype=tf.int32, name='y_ph')
        else:
            self.y_ph = tf.placeholder(shape=(None, self.n_classes), dtype=tf.float32, name='y_ph')

        self.learning_rate_ph = tf.placeholder_with_default(0.0, shape=[], name='learning_rate_ph')
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph')
        self.is_train_ph = tf.placeholder_with_default(False, shape=[], name='is_train_ph')

    def _build_feed_dict(self, input_ids_a, input_masks_a, token_types_a,
                         input_ids_b, input_masks_b, token_types_b, y=None):
        feed_dict = {
            self.input_ids_ph_a: input_ids_a,
            self.input_masks_ph_a: input_masks_a,
            self.token_types_ph_a: token_types_a,
            self.input_ids_ph_b: input_ids_b,
            self.input_masks_ph_b: input_masks_b,
            self.token_types_ph_b: token_types_b,

        }
        if y is not None:
            feed_dict.update({
                self.y_ph: y,
                self.learning_rate_ph: max(self.get_learning_rate(), self.min_learning_rate),
                self.keep_prob_ph: self.keep_prob,
                self.is_train_ph: True,
            })

        return feed_dict

    def __call__(self, features):

        input_ids_a = [f.input_ids_a for f in features]
        input_masks_a = [f.input_mask_a for f in features]
        input_type_ids_a = [f.input_type_ids_a for f in features]
        input_ids_b = [f.input_ids_b for f in features]
        input_masks_b = [f.input_mask_b for f in features]
        input_type_ids_b = [f.input_type_ids_b for f in features]

        feed_dict = self._build_feed_dict(input_ids_a, input_masks_a, input_type_ids_a,
                                          input_ids_b, input_masks_b, input_type_ids_b)
        if not self.return_probas:
            pred = self.sess.run(self.y_predictions, feed_dict=feed_dict)
        else:
            pred = self.sess.run(self.y_probas, feed_dict=feed_dict)
        return pred
