# originally based on https://github.com/allenai/bilm-tf/blob/master/bilm/training.py

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

from typing import  Optional

import tensorflow as tf
import numpy as np
import json

from deeppavlov.core.models.tf_model import TFModel
# from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.commands.utils import expand_path

from deeppavlov.models.elmo.bilm_model import LanguageModel
from deeppavlov.models.elmo.train_utils import print_variable_summary, average_gradients, clip_grads

log = get_logger(__name__)


# class ELMo(TFModel): # TODO: Add TFModel inheritance
@register('elmo_model')
class ELMo(TFModel):
    """>>
    The :class:`~deeppavlov.models.elmo.elmo.ELMo` is for Neural Named Entity Recognition and Slot Filling.

    Parameters:
        options_json_path: Path to the json configure.
        char_cnn: Options of char_cnn. For example {"activation":"relu","embedding":{"dim":16},"filters":[[1,32],[2,32],[3,64],[4,128],[5,256],[6,512],[7,1024]],"max_characters_per_token":50,"n_characters":261,"n_highway":2}
        bidirectional: Whether to use bidirectional or not.
        unroll_steps: Number of unrolling steps.
        n_tokens_vocab: A size of a vocabulary.
        lstm: Options of lstm. It is a dict of "cell_clip":int ,"dim":int ,"n_layers":int ,"proj_clip":int ,"projection_dim":int ,"use_skip_connections":bool
        dropout: Probability of keeping the network state, values from 0 to 1. 
        n_negative_samples_batch: Whether to use negative samples batch or not. Number of batch samples.
        all_clip_norm_val: Clip the gradients.
        initial_accumulator_value: Whether to use dropout between layers or not.
        learning_rate: Learning rate to use during the training (usually from 0.1 to 0.0001)
        n_gpus: Number of gpu to use.
        seed: Random seed.
        batch_size: A size of a train batch.
    """

    def __init__(self,
                 options_json_path: Optional[str] = None, # Configure by json file
                 char_cnn: Optional[dict] = None, # Net architecture by direct params, it may be used for overwrite a json file architecture
                 bidirectional: Optional[bool] = None,
                 unroll_steps: Optional[int] = None,
                 n_tokens_vocab: Optional[int] = None,
                 lstm: Optional[dict] = None,
                 dropout: Optional[float] = None,  # Regularization
                 n_negative_samples_batch: Optional[int] = None, # Train options
                 all_clip_norm_val: Optional[float] = None,
                 initial_accumulator_value: float = 1.0,
                 learning_rate: float = 2e-1, # For AdagradOptimizer
                 n_gpus: int = 1, # TODO: Add cpu supporting
                 seed: int = None, # Other
                 batch_size: int = 128, # Data params
                 **kwargs) -> None:
        
        # ================ Checking input args =================
        if not(options_json_path or (char_cnn and bidirectional and unroll_steps\
                                     and n_tokens_vocab and lstm and dropout and\
                                     n_negative_samples_batch and all_clip_norm_val\
                                     )):
            raise Warning('Use options_json_path or/and direct params to set net architecture.')
        self._options = self._load_options(options_json_path)
        self._update_arch_options(char_cnn, bidirectional, unroll_steps, n_tokens_vocab, lstm)
        self._update_other_options(dropout, n_negative_samples_batch, all_clip_norm_val)

        # Special options
        self._options['learning_rate'] = learning_rate
        self._options['initial_accumulator_value'] = initial_accumulator_value
        self._options['seed'] = seed
        self._options['n_gpus'] = n_gpus
        self._options['batch_size'] = batch_size
        tf.set_random_seed(seed)
        np.random.seed(seed)

        # ==================== Suply vars =====================

        self._train_last_loss = np.inf
        
        # ================== Building the network ==================

        self.models, self.train_op, _, _, self.train_loss, self.valid_loss = self._build_graph()

        # ================= Initialize the session =================
        self.init_state_values, self.init_state_tensors, self.final_state_tensors =\
            self._init_session()


        super().__init__(**kwargs)
        self.load()

    def _load_options(self, options_json_path):
        if options_json_path:
            options_json_path = expand_path(options_json_path)
            with open(options_json_path, 'r') as fin:
                options = json.load(fin)
        else:
            options = {}
        return options

    def _update_arch_options(self, char_cnn, bidirectional, unroll_steps, n_tokens_vocab, lstm):
        if char_cnn is not None:
            self._options['char_cnn'] = char_cnn
        if bidirectional is not None:
            self._options['bidirectional'] = bidirectional
        if unroll_steps is not None:
            self._options['unroll_steps'] = unroll_steps
        if n_tokens_vocab is not None:
            self._options['n_tokens_vocab'] = n_tokens_vocab
        if lstm is not None:
            self._options['lstm'] = lstm

    def _update_other_options(self, dropout, n_negative_samples_batch, all_clip_norm_val):
        if dropout is not None:
            self._options['dropout'] = dropout
        if n_negative_samples_batch is not None:
            self._options['n_negative_samples_batch'] = n_negative_samples_batch
        if all_clip_norm_val is not None:
            self._options['all_clip_norm_val'] = all_clip_norm_val

    def _build_graph(self):
        init_step = 0 # TODO: Add resolution of init_step
        with tf.device('/cpu:0'):
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(init_step), trainable=False)

            # set up the optimizer
            opt = tf.train.AdagradOptimizer(learning_rate=self._options['learning_rate'],
                                            initial_accumulator_value=1.0)

            # calculate the gradients on each GPU
            tower_grads = []
            models = []
            train_loss = tf.get_variable(
                'train_loss', [],
                initializer=tf.constant_initializer(0.0), trainable=False)
            valid_loss = tf.get_variable(
                'valid_loss', [],
                initializer=tf.constant_initializer(0.0), trainable=False)
            for k in range(self._options['n_gpus']):
                with tf.device('/gpu:%d' % k):
                    with tf.variable_scope('lm', reuse=k > 0):
                        # calculate the loss for one model replica and get
                        #   lstm states
                        model = LanguageModel(self._options, True)
                        total_train_loss = model.total_train_loss
                        total_valid_loss = model.total_valid_loss
                        models.append(model)
                        # get gradients
                        grads = opt.compute_gradients(
                            total_train_loss * self._options['unroll_steps'],
                            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                        )
                        tower_grads.append(grads)
                        # # keep track of loss across all GPUs
                        # train_perplexity += total_train_loss
                        train_loss += total_train_loss
                        valid_loss += total_valid_loss
                print_variable_summary()

            # calculate the mean of each gradient across all GPUs
            grads = average_gradients(tower_grads, self._options['batch_size'], self._options)
            grads, _ = clip_grads(grads, self._options, True, global_step)
            train_loss = train_loss / self._options['n_gpus']
            valid_loss = valid_loss / self._options['n_gpus']
            train_op = opt.apply_gradients(grads, global_step=global_step)
            hist_summary_op = None
            summary_op = None

        return models, train_op, summary_op, hist_summary_op, train_loss, valid_loss

    def _init_session(self):
        restart_ckpt_file = None # TODO: It is one too
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.initialize_all_variables())

        # load the checkpoint data if needed
        if restart_ckpt_file is not None:
            loader = tf.train.Saver()
            loader.restore(self.sess, restart_ckpt_file)

        batch_size = self._options['batch_size']
        unroll_steps = self._options['unroll_steps']

        # get the initial lstm states
        init_state_tensors = []
        final_state_tensors = []
        for model in self.models:
            init_state_tensors.extend(model.init_lstm_state)
            final_state_tensors.extend(model.final_lstm_state)

        char_inputs = 'char_cnn' in self._options
        if char_inputs:
            max_chars = self._options['char_cnn']['max_characters_per_token']

        if not char_inputs:
            feed_dict = {
                model.token_ids:
                    np.zeros([batch_size, unroll_steps], dtype=np.int64)
                for model in self.models
            }
        else:
            feed_dict = {
                model.tokens_characters:
                    np.zeros([batch_size, unroll_steps, max_chars],
                            dtype=np.int32)
                for model in self.models
            }

        if self._options['bidirectional']:
            if not char_inputs:
                feed_dict.update({
                    model.token_ids_reverse:
                        np.zeros([batch_size, unroll_steps], dtype=np.int64)
                    for model in self.models
                })
            else:
                feed_dict.update({
                    model.tokens_characters_reverse:
                        np.zeros([batch_size, unroll_steps, max_chars],
                                dtype=np.int32)
                    for model in self.models
                })

        init_state_values = self.sess.run(init_state_tensors, feed_dict=feed_dict)
        return init_state_values, init_state_tensors, final_state_tensors

    def _fill_feed_dict(self, 
                        char_ids_batches, 
                        reversed_char_ids_batches, 
                        token_ids_batches = None, 
                        reversed_token_ids_batches = None):
        # init state tensors
        feed_dict = {t: v for t, v in zip(
                                    self.init_state_tensors, self.init_state_values)}


        for k, model in enumerate(self.models):
            start = k * self._options['batch_size']
            end = (k + 1) * self._options['batch_size']


            # character inputs
            char_ids = char_ids_batches[start:end] # get char_ids
            
            feed_dict[model.tokens_characters] = char_ids

            if self._options['bidirectional']:
                feed_dict[model.tokens_characters_reverse] = \
                    reversed_char_ids_batches[start:end] # get tokens_characters_reverse
                
            if token_ids_batches is not None:
                feed_dict[model.next_token_id] = token_ids_batches[start:end] # get next_token_id
                if self._options['bidirectional']:
                    feed_dict[model.next_token_id_reverse] = reversed_token_ids_batches[start:end] # get next_token_id_reverse

        return feed_dict


    def __call__(self, *args, **kwargs):
        if len(args) != 4:
            return []
        char_ids_batches, reversed_char_ids_batches, token_ids_batches, reversed_token_ids_batches =\
            args

        feed_dict = self._fill_feed_dict(char_ids_batches, reversed_char_ids_batches, token_ids_batches, reversed_token_ids_batches)
        # TODO: Do right ppl
        ret = self.sess.run([self.train_loss] + self.final_state_tensors, feed_dict)

        self.init_state_values = ret[1:]
        valid_loss = ret[0]


        return [valid_loss]

    def train_on_batch(self, 
                       char_ids_batches:list, 
                       reversed_char_ids_batches:list, 
                       token_ids_batches:list, 
                       reversed_token_ids_batches:list):
        """
        This method is called by trainer to make one training step on one batch.

        Args:
            char_ids_batches: batches of char_ids
            reversed_char_ids_batches: batches of reversed_char_ids
            token_ids_batches: batches of token_ids
            reversed_token_ids_batches: batches of reversed_token_ids

        Returns:
            value of loss function on batch
        """
        
        feed_dict = self._fill_feed_dict(char_ids_batches, reversed_char_ids_batches, token_ids_batches, reversed_token_ids_batches)
        ret = self.sess.run([self.train_loss, self.train_op] + self.final_state_tensors, feed_dict)

        self.init_state_values = ret[2:]
        train_loss = ret[0]

        return train_loss

