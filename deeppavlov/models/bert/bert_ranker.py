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

from operator import itemgetter


import tensorflow as tf
import numpy as np
from logging import getLogger
from bert_dp.modeling import BertConfig, BertModel

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.lr_scheduled_tf_model import LRScheduledTFModel
from deeppavlov.core.commands.utils import expand_path

logger = getLogger(__name__)


@register('bert_ranker')
class BertRankerModel(LRScheduledTFModel):
    # TODO: docs
    # TODO: add head-only pre-training
    def __init__(self, bert_config_file, n_classes, keep_prob,
                 batch_size, num_ranking_samples,
                 one_hot_labels=False,
                 attention_probs_keep_prob=None, hidden_keep_prob=None,
                 pretrained_bert=None,
                 resps=None, resp_vecs=None, resp_features=None, resp_eval=True,
                 conts=None, cont_vecs=None, cont_features=None, cont_eval=True,
                 bot_mode=0, min_learning_rate=1e-06, **kwargs) -> None:
        super().__init__(**kwargs)

        self.batch_size = batch_size
        self.num_ranking_samples = num_ranking_samples
        self.n_classes = n_classes
        self.min_learning_rate = min_learning_rate
        self.keep_prob = keep_prob
        self.one_hot_labels = one_hot_labels
        self.batch_size = batch_size
        self.resp_eval = resp_eval
        self.resps = resps
        self.resp_vecs = resp_vecs
        self.cont_eval = cont_eval
        self.conts = conts
        self.cont_vecs = cont_vecs
        self.bot_mode = bot_mode

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

        self.sess.run(tf.global_variables_initializer())

        if pretrained_bert is not None:
            pretrained_bert = str(expand_path(pretrained_bert))

        if tf.train.checkpoint_exists(pretrained_bert) \
                and not tf.train.checkpoint_exists(str(self.load_path.resolve())):
            logger.info('[initializing model with Bert from {}]'.format(pretrained_bert))
            # Exclude optimizer and classification variables from saved variables
            var_list = self._get_saveable_variables(
                exclude_scopes=('Optimizer', 'learning_rate', 'momentum', 'classification'))
            saver = tf.train.Saver(var_list)
            saver.restore(self.sess, pretrained_bert)

        if self.load_path is not None:
            self.load()

        if self.resp_eval:
            assert(self.resps is not None)
        if self.cont_eval:
            assert(self.conts is not None)
        if self.resp_eval and self.cont_eval:
            assert(len(self.resps) == len(self.conts))

        if self.resps is not None and self.resp_vecs is None:
            self.resp_features = [resp_features[0][i * self.batch_size: (i + 1) * self.batch_size]
                                  for i in range(len(resp_features[0]) // batch_size + 1)]
            self.resp_vecs = self(self.resp_features)
            np.save(self.save_path / "resp_vecs", self.resp_vecs)

        if self.conts is not None and self.cont_vecs is None:
            self.cont_features = [cont_features[0][i * self.batch_size: (i + 1) * self.batch_size]
                                  for i in range(len(cont_features[0]) // batch_size + 1)]
            self.cont_vecs = self(self.cont_features)
            np.save(self.save_path / "cont_vecs", self.cont_vecs)

    def _init_graph(self):
        self._init_placeholders()
        with tf.variable_scope("model"):
            self.bert = BertModel(config=self.bert_config,
                                  is_training=self.is_train_ph,
                                  input_ids=self.input_ids_ph,
                                  input_mask=self.input_masks_ph,
                                  token_type_ids=self.token_types_ph,
                                  use_one_hot_embeddings=False,
                                  )

        output_layer_a = self.bert.get_pooled_output()

        with tf.variable_scope("loss"):
            with tf.variable_scope("loss"):
                self.loss = tf.contrib.losses.metric_learning.npairs_loss(self.y_ph, output_layer_a, output_layer_a)
                self.y_probas = output_layer_a

    def _init_placeholders(self):
        self.input_ids_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='ids_ph')
        self.input_masks_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='masks_ph')
        self.token_types_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='token_types_ph')

        if not self.one_hot_labels:
            self.y_ph = tf.placeholder(shape=(None, ), dtype=tf.int32, name='y_ph')
        else:
            self.y_ph = tf.placeholder(shape=(None, self.n_classes), dtype=tf.float32, name='y_ph')

        self.learning_rate_ph = tf.placeholder_with_default(0.0, shape=[], name='learning_rate_ph')
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph')
        self.is_train_ph = tf.placeholder_with_default(False, shape=[], name='is_train_ph')

    def _init_optimizer(self):
        # TODO: use AdamWeightDecay optimizer
        with tf.variable_scope('Optimizer'):
            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
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

    def train_on_batch(self, features, y):
        pass

    def __call__(self, features_list):
        pred = []
        for features in features_list:
            input_ids = [f.input_ids for f in features]
            input_masks = [f.input_mask for f in features]
            input_type_ids = [f.input_type_ids for f in features]
            feed_dict = self._build_feed_dict(input_ids, input_masks, input_type_ids)
            p = self.sess.run(self.y_probas, feed_dict=feed_dict)
            if len(p.shape) == 1:
                p = np.expand_dims(p, 0)
            pred.append(p)
        pred = np.vstack(pred)
        bs = pred.shape[0]
        if self.bot_mode == 0:
            s = pred @ self.resp_vecs.T
            ids = np.argmax(s, 1)
            ans = [[self.resps[el] for el in ids]]
        if self.bot_mode == 1:
            sr = pred @ self.resp_vecs.T
            sc = pred @ self.cont_vecs.T
            ids = np.argsort(sr, 1)[:, -10:]
            sc = [sc[i, ids[i]] for i in range(bs)]
            ids = [sorted(zip(ids[i], sc[i]), key=itemgetter(1), reverse=True) for i in range(bs)]
            ids = [list(map(lambda x: x[0], ids[i])) for i in range(bs)]
            ans = [[self.resps[ids[i][0]] for i in range(bs)]]
        if self.bot_mode == 2:
            sr = pred @ self.resp_vecs.T
            sc = pred @ self.cont_vecs.T
            ids = np.argsort(sc, 1)[:, -10:]
            sr = [sr[i, ids[i]] for i in range(bs)]
            ids = [sorted(zip(ids[i], sr[i]), key=itemgetter(1), reverse=True) for i in range(bs)]
            ids = [list(map(lambda x: x[0], ids[i])) for i in range(bs)]
            ans = [[self.resps[ids[i][0]] for i in range(bs)]]
        if self.bot_mode == 3:
            sr = pred @ self.resp_vecs.T
            sc = pred @ self.cont_vecs.T
            s = sr + sc
            ids = np.argmax(s, 1)
            ans = [[self.resps[el] for el in ids]]
        return ans[0]


