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
from typing import List, Dict, Union, Optional
import math

import tensorflow as tf
from bert_dp.modeling import BertConfig, BertModel
from bert_dp.optimization import AdamWeightDecayOptimizer
from bert_dp.preprocessing import InputFeatures

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import LRScheduledTFModel

logger = getLogger(__name__)
tf.enable_resource_variables()


@register('bert_classifier')
class BertClassifierModel(LRScheduledTFModel):
    """Bert-based model for text classification.

    It uses output from [CLS] token and predicts labels using linear transformation.

    Args:
        bert_config_file: path to Bert configuration file
        n_classes: number of classes
        keep_prob: dropout keep_prob for non-Bert layers
        one_hot_labels: set True if one-hot encoding for labels is used
        multilabel: set True if it is multi-label classification
        return_probas: set True if return class probabilites instead of most probable label needed
        attention_probs_keep_prob: keep_prob for Bert self-attention layers
        hidden_keep_prob: keep_prob for Bert hidden layers
        optimizer: name of tf.train.* optimizer or None for `AdamWeightDecayOptimizer`
        num_warmup_steps: number of warmup steps before training
        gradient_accumulation_steps: number of steps we do on each batch
        ( if it is >1, we split the batch on the gradient_accumulation_step parts
         and update params at the end of the batch, default: 1)
        weight_decay_rate: L2 weight decay for `AdamWeightDecayOptimizer`
        pretrained_bert: pretrained Bert checkpoint
        min_learning_rate: min value of learning rate if learning rate decay is used
        clip_norm: clip gradients by clip_norm (default: None, can be equal to 1, 2 etc)
    """

    # TODO: add warmup
    # TODO: add head-only pre-training
    def __init__(self, bert_config_file, n_classes, keep_prob,
                 one_hot_labels=False, multilabel=False, return_probas=False,
                 attention_probs_keep_prob=None, hidden_keep_prob=None,
                 optimizer=None, gradient_accumulation_steps=1,
                 num_warmup_steps=None, weight_decay_rate=0.01,
                 pretrained_bert=None, min_learning_rate=1e-06,
                 clip_norm=None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.return_probas = return_probas
        self.n_classes = n_classes
        self.min_learning_rate = min_learning_rate
        self.keep_prob = keep_prob
        self.one_hot_labels = one_hot_labels
        self.multilabel = multilabel
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        assert self.gradient_accumulation_steps > 0
        self.num_warmup_steps = num_warmup_steps
        self.weight_decay_rate = weight_decay_rate
        self.clip_norm = clip_norm
        if self.multilabel and not self.one_hot_labels:
            raise RuntimeError('Use one-hot encoded labels for multilabel classification!')

        if self.multilabel and not self.return_probas:
            raise RuntimeError('Set return_probas to True for multilabel classification!')

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

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())

        if pretrained_bert is not None:
            pretrained_bert = str(expand_path(pretrained_bert))

            if tf.train.checkpoint_exists(pretrained_bert) \
                and not (self.load_path and tf.train.checkpoint_exists(str(self.load_path.resolve()))):
                logger.info('[initializing model with Bert from {}]'.format(pretrained_bert))
                # Exclude optimizer and classification variables from saved variables
                var_list = self._get_saveable_variables(
                    exclude_scopes=('Optimizer', 'learning_rate', 'momentum', 'output_weights', 'output_bias'))
                saver = tf.train.Saver(var_list)
                saver.restore(self.sess, pretrained_bert)

        if self.load_path is not None:
            self.load()

    def _init_graph(self):
        self._init_placeholders()

        self.bert = BertModel(config=self.bert_config,
                              is_training=self.is_train_ph,
                              input_ids=self.input_ids_ph,
                              input_mask=self.input_masks_ph,
                              token_type_ids=self.token_types_ph,
                              use_one_hot_embeddings=False,
                              )

        output_layer = self.bert.get_pooled_output()
        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [self.n_classes, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [self.n_classes], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            output_layer = tf.nn.dropout(output_layer, keep_prob=self.keep_prob_ph)
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            if self.one_hot_labels:
                one_hot_labels = self.y_ph
            else:
                one_hot_labels = tf.one_hot(self.y_ph, depth=self.n_classes, dtype=tf.float32)

            self.y_predictions = tf.argmax(logits, axis=-1)
            if not self.multilabel:
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                self.y_probas = tf.nn.softmax(logits, axis=-1)
                per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
                self.loss = tf.reduce_mean(per_example_loss)
            else:
                self.y_probas = tf.nn.sigmoid(logits)
                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_labels, logits=logits))

    def _init_placeholders(self):
        self.input_ids_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='ids_ph')
        self.input_masks_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='masks_ph')
        self.token_types_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='token_types_ph')

        if not self.one_hot_labels:
            self.y_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='y_ph')
        else:
            self.y_ph = tf.placeholder(shape=(None, self.n_classes), dtype=tf.float32, name='y_ph')

        self.learning_rate_ph = tf.placeholder_with_default(0.0, shape=[], name='learning_rate_ph')
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph')
        self.is_train_ph = tf.placeholder_with_default(False, shape=[], name='is_train_ph')

    def _init_optimizer(self, learnable_scopes=None, clip_norm=None):
        with tf.variable_scope('Optimizer'):
            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            if learnable_scopes is None:
                variables_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            else:
                variables_to_train = []
                for scope_name in learnable_scopes:
                    variables_to_train.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name))
            if self.optimizer is None:
                self.optimizer = AdamWeightDecayOptimizer(
                    learning_rate=self.learning_rate_ph,
                    weight_decay_rate=self.weight_decay_rate,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-6,
                    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
                )
                new_global_step = self.global_step + 1
                grads_and_vars = self.optimizer.compute_gradients(self.loss, var_list=variables_to_train)

                def clip_if_not_none(grad):
                    if grad is not None:
                        return tf.clip_by_norm(grad, clip_norm)

                if clip_norm is not None:
                    grads_and_vars = [(clip_if_not_none(grad), var)
                                      for grad, var in grads_and_vars]
                self.train_op = self.optimizer.apply_gradients(grads_and_vars)
            else:
                self.train_op = self.get_train_op(self.loss, learning_rate=self.learning_rate_ph)

    def _build_feed_dict(self, input_ids, input_masks, token_types, y=None):
        feed_dict = {
            self.input_ids_ph: input_ids,
            self.input_masks_ph: input_masks,
            self.token_types_ph: token_types,
        }
        if y is not None:
            feed_dict.update({
                self.y_ph: y,
                self.learning_rate_ph: max(self.get_learning_rate(), self.min_learning_rate),
                self.keep_prob_ph: self.keep_prob,
                self.is_train_ph: True,
            })

        return feed_dict

    def split(self, features: List[InputFeatures]):
        """
        Splits features: batch of InputFeatures
         on the number of parts equal to the number of gradient accumulation steps

        Args:
            features: batch of InputFeatures

        Returns:
            list of input feature batches, with the size equal to the gradient_accumulation_steps

        """

        num_parts = self.gradient_accumulation_steps
        assert num_parts <= len(features)
        num_features = math.ceil(len(features) / num_parts)
        feature_batches = [features[i:i + num_features] for i in range(0, len(features), num_features)]
        return feature_batches

    def train_on_batch(self, features: List[InputFeatures], y: Union[List[int], List[List[int]]] = None) -> Dict:
        """Train model on given batch.
        This method calls train_op using features and y (labels).

        Args:
            features: batch of InputFeatures
            y: batch of labels (class id or one-hot encoding)

        Returns:
            dict with loss and learning_rate values

        """
        # Define operations
        #raise Exception('Func')
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            if self.gradient_accumulation_steps == 1:  # Don't use accumulation
                input_ids = [f.input_ids for f in features]
                input_masks = [f.input_mask for f in features]
                input_type_ids = [f.input_type_ids for f in features]
                feed_dict = self._build_feed_dict(input_ids, input_masks, input_type_ids, y)
                loss = self.sess.run(self.loss, feed_dict=feed_dict)
                return {'loss': loss, 'learning_rate': feed_dict[self.learning_rate_ph]}

            trainable_variables = tf.trainable_variables()
            trainable_variables = [j for j in trainable_variables
                                   if 'learning_rate' not in j.name and 'momentum' not in j.name]
            accumulated_gradients = [tf.Variable(tf.zeros_like(this_var), trainable=False)
                                     for this_var in trainable_variables]
            gradients_vars = self.optimizer.compute_gradients(self.loss, trainable_variables)

            def clip_if_not_none(grad):
                if grad is not None:
                    return tf.clip_by_norm(grad, self.clip_norm)

            if self.clip_norm is not None:
                gradients_vars = [(clip_if_not_none(grad), var)
                                  for grad, var in gradients_vars]
            apply_gradients = self.optimizer.apply_gradients([
                (accumulated_gradient, variable)
                for accumulated_gradient, (gradient, variable)
                in zip(accumulated_gradients, gradients_vars)])
            evaluate_batch = [
                accumulated_gradient.assign_add(tf.div(gradient, self.gradient_accumulation_steps))
                for accumulated_gradient, (gradient, variable)
                in zip(accumulated_gradients, gradients_vars)]
            average_loss = tf.Variable(0., trainable=False)
            update_loss = average_loss.assign_add(tf.div(self.loss, self.gradient_accumulation_steps))
            reset_gradients = [gradient.assign(tf.zeros_like(gradient)) for gradient in
                               accumulated_gradients]
            reset_loss = average_loss.assign(0.)
            self.sess.run([reset_gradients, reset_loss])

            # Extract data
            input_ids = [f.input_ids for f in features]
            input_ids_batches = self.split(input_ids)
            input_masks = [f.input_mask for f in features]
            input_masks_batches = self.split(input_masks)
            token_type_ids = [f.input_type_ids for f in features]
            token_types_batches = self.split(token_type_ids)
            y_batches = self.split(y)
            #raise Exception('Splitted')
            # https://stackoverflow.com/questions/50000263/how-to-feed-the-list-of-gradients-or-grad-variable-name-pairs-to-my-model
            for (input_ids_batch, input_masks_batch, token_types_batch, y_batch) in zip(
                input_ids_batches, input_masks_batches, token_types_batches, y_batches):
                feed_dict = self._build_feed_dict(input_ids=input_ids_batch, input_masks=input_masks_batch,
                                                  token_types=token_types_batch, y=y_batch)
                self.sess.run([evaluate_batch, update_loss],
                              feed_dict=feed_dict)
            loss = self.sess.run(average_loss)
            #self.sess.run(apply_gradients)
            learning_rate = max(self.get_learning_rate(), self.min_learning_rate)
            answer = {'loss': loss, 'learning_rate': learning_rate}

            return answer

    def __call__(self, features: List[InputFeatures]) -> Union[List[int], List[List[float]]]:
        """Make prediction for given features (texts).

        Args:
            features: batch of InputFeatures

        Returns:
            predicted classes or probabilities of each class

        """
        input_ids = [f.input_ids for f in features]
        input_masks = [f.input_mask for f in features]
        input_type_ids = [f.input_type_ids for f in features]

        feed_dict = self._build_feed_dict(input_ids, input_masks, input_type_ids)
        if not self.return_probas:
            pred = self.sess.run(self.y_predictions, feed_dict=feed_dict)
        else:
            pred = self.sess.run(self.y_probas, feed_dict=feed_dict)
        return pred
