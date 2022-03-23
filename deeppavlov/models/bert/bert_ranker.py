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

import re
from collections import OrderedDict
from logging import getLogger
from operator import itemgetter
from typing import List, Dict, Union

import numpy as np
import tensorflow as tf
from bert_dp.modeling import BertConfig, BertModel
from bert_dp.optimization import AdamWeightDecayOptimizer
from bert_dp.preprocessing import InputFeatures

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import LRScheduledTFModel
from deeppavlov.models.bert.bert_classifier import BertClassifierModel

logger = getLogger(__name__)


@register('bert_ranker')
class BertRankerModel(BertClassifierModel):
    """BERT-based model for interaction-based text ranking.

    Linear transformation is trained over the BERT pooled output from [CLS] token.
    Predicted probabilities of classes are used as a similarity measure for ranking.

    Args:
        bert_config_file: path to Bert configuration file
        n_classes: number of classes
        keep_prob: dropout keep_prob for non-Bert layers
        return_probas: set True if class probabilities are returned instead of the most probable label
    """

    def __init__(self, bert_config_file, n_classes=2, keep_prob=0.9, return_probas=True, **kwargs) -> None:
        super().__init__(bert_config_file=bert_config_file, n_classes=n_classes,
                         keep_prob=keep_prob, return_probas=return_probas, **kwargs)

    def train_on_batch(self, features_li: List[List[InputFeatures]], y: Union[List[int], List[List[int]]]) -> Dict:
        """Train the model on the given batch.

        Args:
            features_li: list with the single element containing the batch of InputFeatures
            y: batch of labels (class id or one-hot encoding)

        Returns:
            dict with loss and learning rate values
        """

        features = features_li[0]
        input_ids = [f.input_ids for f in features]
        input_masks = [f.input_mask for f in features]
        input_type_ids = [f.input_type_ids for f in features]

        feed_dict = self._build_feed_dict(input_ids, input_masks, input_type_ids, y)

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return {'loss': loss, 'learning_rate': feed_dict[self.learning_rate_ph]}

    def __call__(self, features_li: List[List[InputFeatures]]) -> Union[List[int], List[List[float]]]:
        """Calculate scores for the given context over candidate responses.

        Args:
            features_li: list of elements where each element contains the batch of features
             for contexts with particular response candidates

        Returns:
            predicted scores for contexts over response candidates
        """

        if len(features_li) == 1 and len(features_li[0]) == 1:
            msg = "It is not intended to use the {} in the interact mode.".format(self.__class__)
            logger.error(msg)
            return [msg]

        predictions = []
        for features in features_li:
            input_ids = [f.input_ids for f in features]
            input_masks = [f.input_mask for f in features]
            input_type_ids = [f.input_type_ids for f in features]

            feed_dict = self._build_feed_dict(input_ids, input_masks, input_type_ids)
            if not self.return_probas:
                pred = self.sess.run(self.y_predictions, feed_dict=feed_dict)
            else:
                pred = self.sess.run(self.y_probas, feed_dict=feed_dict)
            predictions.append(pred[:, 1])
        if len(features_li) == 1:
            predictions = predictions[0]
        else:
            predictions = np.hstack([np.expand_dims(el, 1) for el in predictions])
        return predictions


@register('bert_sep_ranker')
class BertSepRankerModel(LRScheduledTFModel):
    """BERT-based model for representation-based text ranking.

     BERT pooled output from [CLS] token is used to get a separate representation of a context and a response.
     Similarity measure is calculated as cosine similarity between these representations.

    Args:
        bert_config_file: path to Bert configuration file
        keep_prob: dropout keep_prob for non-Bert layers
        attention_probs_keep_prob: keep_prob for Bert self-attention layers
        hidden_keep_prob: keep_prob for Bert hidden layers
        optimizer: name of tf.train.* optimizer or None for ``AdamWeightDecayOptimizer``
        weight_decay_rate: L2 weight decay for ``AdamWeightDecayOptimizer``
        pretrained_bert: pretrained Bert checkpoint
        min_learning_rate: min value of learning rate if learning rate decay is used
    """

    def __init__(self, bert_config_file, keep_prob=0.9,
                 attention_probs_keep_prob=None, hidden_keep_prob=None,
                 optimizer=None, weight_decay_rate=0.01,
                 pretrained_bert=None, min_learning_rate=1e-06, **kwargs) -> None:
        super().__init__(**kwargs)

        self.min_learning_rate = min_learning_rate
        self.keep_prob = keep_prob
        self.optimizer = optimizer
        self.weight_decay_rate = weight_decay_rate

        self.bert_config = BertConfig.from_json_file(str(expand_path(bert_config_file)))

        if attention_probs_keep_prob is not None:
            self.bert_config.attention_probs_dropout_prob = 1.0 - attention_probs_keep_prob
        if hidden_keep_prob is not None:
            self.bert_config.hidden_dropout_prob = 1.0 - hidden_keep_prob

        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)

        self._init_graph()

        self._init_optimizer()

        if pretrained_bert is not None:
            pretrained_bert = str(expand_path(pretrained_bert))

            if tf.train.checkpoint_exists(pretrained_bert) \
                    and not (self.load_path and tf.train.checkpoint_exists(str(self.load_path.resolve()))):
                logger.info('[initializing model with Bert from {}]'.format(pretrained_bert))
                # Exclude optimizer and classification variables from saved variables
                var_list = self._get_saveable_variables(
                    exclude_scopes=('Optimizer', 'learning_rate', 'momentum', 'output_weights', 'output_bias'))
                assignment_map = self.get_variables_to_restore(var_list, pretrained_bert)
                tf.train.init_from_checkpoint(pretrained_bert, assignment_map)

        self.sess.run(tf.global_variables_initializer())

        if self.load_path is not None:
            self.load()

    @classmethod
    def get_variables_to_restore(cls, tvars, init_checkpoint):
        """Determine correspondence of checkpoint variables to current variables."""

        assignment_map = OrderedDict()
        graph_names = []
        for var in tvars:
            name = var.name
            m = re.match("^(.*):\\d+$", name)
            if m is not None:
                name = m.group(1)
                graph_names.append(name)
        ckpt_names = [el[0] for el in tf.train.list_variables(init_checkpoint)]
        for u in ckpt_names:
            for v in graph_names:
                if u in v:
                    assignment_map[u] = v
        return assignment_map

    def _init_graph(self):
        self._init_placeholders()

        with tf.variable_scope("model"):
            model_a = BertModel(
                config=self.bert_config,
                is_training=self.is_train_ph,
                input_ids=self.input_ids_a_ph,
                input_mask=self.input_masks_a_ph,
                token_type_ids=self.token_types_a_ph,
                use_one_hot_embeddings=False)

        with tf.variable_scope("model", reuse=True):
            model_b = BertModel(
                config=self.bert_config,
                is_training=self.is_train_ph,
                input_ids=self.input_ids_b_ph,
                input_mask=self.input_masks_b_ph,
                token_type_ids=self.token_types_b_ph,
                use_one_hot_embeddings=False)

        output_layer_a = model_a.get_pooled_output()
        output_layer_b = model_b.get_pooled_output()

        with tf.variable_scope("loss"):
            output_layer_a = tf.nn.dropout(output_layer_a, keep_prob=self.keep_prob_ph)
            output_layer_b = tf.nn.dropout(output_layer_b, keep_prob=self.keep_prob_ph)
            output_layer_a = tf.nn.l2_normalize(output_layer_a, axis=1)
            output_layer_b = tf.nn.l2_normalize(output_layer_b, axis=1)
            embeddings = tf.concat([output_layer_a, output_layer_b], axis=0)
            labels = tf.concat([self.y_ph, self.y_ph], axis=0)
            self.loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, embeddings)
            logits = tf.multiply(output_layer_a, output_layer_b)
            self.y_probas = tf.reduce_sum(logits, 1)
            self.pooled_out = output_layer_a

    def _init_placeholders(self):
        self.input_ids_a_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='ids_a_ph')
        self.input_masks_a_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='masks_a_ph')
        self.token_types_a_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='token_a_types_ph')
        self.input_ids_b_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='ids_b_ph')
        self.input_masks_b_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='masks_b_ph')
        self.token_types_b_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='token_types_b_ph')
        self.y_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='y_ph')
        self.learning_rate_ph = tf.placeholder_with_default(0.0, shape=[], name='learning_rate_ph')
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph')
        self.is_train_ph = tf.placeholder_with_default(False, shape=[], name='is_train_ph')

    def _init_optimizer(self):
        with tf.variable_scope('Optimizer'):
            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            # default optimizer for Bert is Adam with fixed L2 regularization
            if self.optimizer is None:

                self.train_op = self.get_train_op(self.loss, learning_rate=self.learning_rate_ph,
                                                  optimizer=AdamWeightDecayOptimizer,
                                                  weight_decay_rate=self.weight_decay_rate,
                                                  beta_1=0.9,
                                                  beta_2=0.999,
                                                  epsilon=1e-6,
                                                  exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
                                                  )
            else:
                self.train_op = self.get_train_op(self.loss, learning_rate=self.learning_rate_ph)

            if self.optimizer is None:
                new_global_step = self.global_step + 1
                self.train_op = tf.group(self.train_op, [self.global_step.assign(new_global_step)])

    def _build_feed_dict(self, input_ids_a, input_masks_a, token_types_a,
                         input_ids_b, input_masks_b, token_types_b, y=None):
        feed_dict = {
            self.input_ids_a_ph: input_ids_a,
            self.input_masks_a_ph: input_masks_a,
            self.token_types_a_ph: token_types_a,
            self.input_ids_b_ph: input_ids_b,
            self.input_masks_b_ph: input_masks_b,
            self.token_types_b_ph: token_types_b,
        }
        if y is not None:
            feed_dict.update({
                self.y_ph: y,
                self.learning_rate_ph: max(self.get_learning_rate(), self.min_learning_rate),
                self.keep_prob_ph: self.keep_prob,
                self.is_train_ph: True,
            })

        return feed_dict

    def train_on_batch(self, features_li: List[List[InputFeatures]], y: Union[List[int], List[List[int]]]) -> Dict:
        """Train the model on the given batch.

        Args:
            features_li: list with two elements, one containing the batch of context features
             and the other containing the batch of response features
            y: batch of labels (class id or one-hot encoding)

        Returns:
            dict with loss and learning rate values
        """

        input_ids_a = [f.input_ids for f in features_li[0]]
        input_masks_a = [f.input_mask for f in features_li[0]]
        input_type_ids_a = [f.input_type_ids for f in features_li[0]]
        input_ids_b = [f.input_ids for f in features_li[1]]
        input_masks_b = [f.input_mask for f in features_li[1]]
        input_type_ids_b = [f.input_type_ids for f in features_li[1]]

        feed_dict = self._build_feed_dict(input_ids_a, input_masks_a, input_type_ids_a,
                                          input_ids_b, input_masks_b, input_type_ids_b, y)

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return {'loss': loss, 'learning_rate': feed_dict[self.learning_rate_ph]}

    def __call__(self, features_li: List[List[InputFeatures]]) -> Union[List[int], List[List[float]]]:
        """Calculate scores for the given context over candidate responses.

        Args:
            features_li: list of elements where the first element represents the context batch of features
                and the rest of elements represent response candidates batches of features

        Returns:
            predicted scores for contexts over response candidates
        """

        if len(features_li) == 1 and len(features_li[0]) == 1:
            msg = "It is not intended to use the {} in the interact mode.".format(self.__class__)
            logger.error(msg)
            return [msg]

        predictions = []
        input_ids_a = [f.input_ids for f in features_li[0]]
        input_masks_a = [f.input_mask for f in features_li[0]]
        input_type_ids_a = [f.input_type_ids for f in features_li[0]]
        for features in features_li[1:]:
            input_ids_b = [f.input_ids for f in features]
            input_masks_b = [f.input_mask for f in features]
            input_type_ids_b = [f.input_type_ids for f in features]

            feed_dict = self._build_feed_dict(input_ids_a, input_masks_a, input_type_ids_a,
                                              input_ids_b, input_masks_b, input_type_ids_b)
            pred = self.sess.run(self.y_probas, feed_dict=feed_dict)
            predictions.append(pred)
        if len(features_li) == 1:
            predictions = predictions[0]
        else:
            predictions = np.hstack([np.expand_dims(el, 1) for el in predictions])
        return predictions
