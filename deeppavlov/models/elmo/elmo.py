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

from typing import Optional, List

import tensorflow as tf
import numpy as np
import json
from overrides import overrides

# from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.commands.utils import expand_path

from deeppavlov.models.elmo.bilm_model import LanguageModel
from deeppavlov.models.elmo.train_utils import print_variable_summary, average_gradients, clip_grads

log = get_logger(__name__)


@register('elmo_model')
class ELMo(NNModel):
    """>>
    The :class:`~deeppavlov.models.elmo.elmo.ELMo` is a deep contextualized word representation that models both
    complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic
    contexts (i.e., to model polysemy)

    Parameters:
        options_json_path: Path to the json configure.
        char_cnn: Options of char_cnn. For example {"activation":"relu","embedding":{"dim":16},
            "filters":[[1,32],[2,32],[3,64],[4,128],[5,256],[6,512],[7,1024]],"max_characters_per_token":50,
            "n_characters":261,"n_highway":2}
        bidirectional: Whether to use bidirectional or not.
        unroll_steps: Number of unrolling steps.
        n_tokens_vocab: A size of a vocabulary.
        lstm: Options of lstm. It is a dict of "cell_clip":int, "dim":int, "n_layers":int, "proj_clip":int, 
            "projection_dim":int, "use_skip_connections":bool
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
                 options_json_path: Optional[str] = None,  # Configure by json file
                 char_cnn: Optional[dict] = None,  # Net architecture by direct params, use for overwrite a json arch.
                 bidirectional: Optional[bool] = None,
                 unroll_steps: Optional[int] = None,
                 n_tokens_vocab: Optional[int] = None,
                 lstm: Optional[dict] = None,
                 dropout: Optional[float] = None,   # Regularization
                 n_negative_samples_batch: Optional[int] = None,  # Train options
                 all_clip_norm_val: Optional[float] = None,
                 initial_accumulator_value: float = 1.0,
                 learning_rate: float = 2e-1,  # For AdagradOptimizer
                 n_gpus: int = 1,  # TODO: Add cpu supporting
                 seed: int = None,  # Other
                 batch_size: int = 128,  # Data params
                 **kwargs) -> None:
        
        # ================ Checking input args =================
        if not(options_json_path or (char_cnn and bidirectional and unroll_steps
                                     and n_tokens_vocab and lstm and dropout and
                                     n_negative_samples_batch and all_clip_norm_val
                                     )):
            raise Warning('Use options_json_path or/and direct params to set net architecture.')
        self.options = self._load_options(options_json_path)
        self._update_arch_options(char_cnn, bidirectional, unroll_steps, n_tokens_vocab, lstm)
        self._update_other_options(dropout, n_negative_samples_batch, all_clip_norm_val)

        # Special options
        self.options['learning_rate'] = learning_rate
        self.options['initial_accumulator_value'] = initial_accumulator_value
        self.options['seed'] = seed
        self.options['n_gpus'] = n_gpus
        self.options['batch_size'] = batch_size
        self.train_options = {}
        self.valid_options = {'batch_size': 256, 'unroll_steps': 1}
        self._options = self.options.copy()
        self.init_step = 0
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self._build_graphs()

        self._build_model(train = False, rebuild = False, batch_size = 256, unroll_steps = 1, n_gpus = 1)
        self.sess = tf.Session()
        super().__init__(**kwargs)
        # self.load()
        # print(repr(vars(self)))
        # input()

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
            self.options['char_cnn'] = char_cnn
        if bidirectional is not None:
            self.options['bidirectional'] = bidirectional
        if unroll_steps is not None:
            self.options['unroll_steps'] = unroll_steps
        if n_tokens_vocab is not None:
            self.options['n_tokens_vocab'] = n_tokens_vocab
        if lstm is not None:
            self.options['lstm'] = lstm

    def _update_other_options(self, dropout, n_negative_samples_batch, all_clip_norm_val):
        if dropout is not None:
            self.options['dropout'] = dropout
        if n_negative_samples_batch is not None:
            self.options['n_negative_samples_batch'] = n_negative_samples_batch
        if all_clip_norm_val is not None:
            self.options['all_clip_norm_val'] = all_clip_norm_val

    def _build_graphs(self):
        if not hasattr(self, 'train_graph'):
            self.options = self._options.copy()
            self.options.update(self.train_options)
            models, train_op, train_loss, eval_loss, graph =\
                self._build_graph(tf.Graph())
            self.train_model_graph = {"models": models,
                                      "train_op": train_op,
                                      "train_loss": train_loss,
                                      "eval_loss": eval_loss,
                                      "graph": graph
                                      }

        if not hasattr(self, 'valid_graph'):
            self.options = self._options.copy()
            self.options.update(self.valid_options)
            models, train_op, train_loss, eval_loss, graph =\
                self._build_graph(tf.Graph())
            self.valid_model_graph = {"models": models,
                                      "train_op": train_op,
                                      "train_loss": train_loss,
                                      "eval_loss": eval_loss,
                                      "graph": graph
                                      }
                                            
    def _build_graph(self, graph):
        with graph.as_default():
            with tf.device('/cpu:0'):
                global_step = tf.get_variable(
                    'global_step', [],
                    initializer=tf.constant_initializer(self.init_step), trainable=False)

                # set up the optimizer
                opt = tf.train.AdagradOptimizer(learning_rate=self.options['learning_rate'],
                                                initial_accumulator_value=1.0)

                # calculate the gradients on each GPU
                tower_grads = []
                models = []
                train_loss = tf.get_variable(
                    'train_loss', [],
                    initializer=tf.constant_initializer(0.0), trainable=False)
                eval_loss = tf.get_variable(
                    'eval_loss', [],
                    initializer=tf.constant_initializer(0.0), trainable=False)
                for k in range(self.options['n_gpus']):
                    with tf.device('/gpu:%d' % k):
                        with tf.variable_scope('lm', reuse=k > 0):
                            # calculate the loss for one model replica and get
                            #   lstm states
                            model = LanguageModel(self.options, True)
                            total_train_loss = model.total_train_loss
                            total_eval_loss = model.total_eval_loss
                            models.append(model)
                            # get gradients
                            grads = opt.compute_gradients(
                                total_train_loss * self.options['unroll_steps'],
                                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                            )
                            tower_grads.append(grads)
                            # # keep track of loss across all GPUs
                            # train_perplexity += total_train_loss
                            train_loss += total_train_loss
                            eval_loss += total_eval_loss
                    print_variable_summary()

                # calculate the mean of each gradient across all GPUs
                grads = average_gradients(tower_grads, self.options['batch_size'], self.options)
                grads, _ = clip_grads(grads, self.options, True, global_step)
                train_loss = train_loss / self.options['n_gpus']
                eval_loss = eval_loss / self.options['n_gpus']
                train_op = opt.apply_gradients(grads, global_step=global_step)

        return models, train_op, train_loss, eval_loss, graph

    def _init_session(self):
        restart_ckpt_file = None  # TODO: It is one too
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())

        # load the checkpoint data if needed
        if restart_ckpt_file is not None:
            loader = tf.train.Saver()
            loader.restore(self.sess, restart_ckpt_file)

        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']

        # get the initial lstm states
        init_state_tensors = []
        final_state_tensors = []
        for model in self.models:
            init_state_tensors.extend(model.init_lstm_state)
            final_state_tensors.extend(model.final_lstm_state)

        char_inputs = 'char_cnn' in self.options
        if char_inputs:
            max_chars = self.options['char_cnn']['max_characters_per_token']

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

        if self.options['bidirectional']:
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
        feed_dict = {t: v for t, v in zip(self.init_state_tensors, self.init_state_values)}

        for k, model in enumerate(self.models):
            start = k * self.options['batch_size']
            end = (k + 1) * self.options['batch_size']

            # character inputs
            char_ids = char_ids_batches[start:end]  # get char_ids
            
            feed_dict[model.tokens_characters] = char_ids

            if self.options['bidirectional']:
                feed_dict[model.tokens_characters_reverse] = \
                    reversed_char_ids_batches[start:end]  # get tokens_characters_reverse
                
            if token_ids_batches is not None:
                feed_dict[model.next_token_id] = token_ids_batches[start:end]  # get next_token_id
                if self.options['bidirectional']:
                    feed_dict[model.next_token_id_reverse] = \
                        reversed_token_ids_batches[start:end]  # get next_token_id_reverse

        return feed_dict

    def __call__(self, *args, **kwargs) -> List[float]:
        if len(args) != 2:
            return []
        char_ids_batches, reversed_char_ids_batches = args[0]
        token_ids_batches, reversed_token_ids_batches = args[1]

        feed_dict = self._fill_feed_dict(char_ids_batches, reversed_char_ids_batches, token_ids_batches, 
                                         reversed_token_ids_batches)

        with self.current_graph.as_default():
            ret = self.sess.run([self.loss] + self.final_state_tensors, feed_dict)

        self.init_state_values = ret[1:]
        loss = ret[0]
        return [loss]

    @overrides
    def load(self) -> None:
        pass

    @overrides
    def save(self) -> None:
        pass
    # @overrides
    # def load(self) -> None:
    #     """Load model parameters from self.load_path"""
    #     with self.current_graph.as_default():
    #         path = str(self.load_path.resolve())
    #         # Check presence of the model files
    #         if tf.train.checkpoint_exists(path):
    #             log.info('[loading model from {}]'.format(path))
    #             saver = tf.train.Saver()
    #             saver.restore(self.sess, path)

    def train_on_batch(self,
                       x_char_ids: list,
                       y_token_ids: list):
        """
        This method is called by trainer to make one training step on one batch.

        Args:
            x_char_ids:  a batch of char_ids
            y_token_ids: a batch of token_ids

        Returns:
            value of loss function on batch
        """

        char_ids_batches, reversed_char_ids_batches = x_char_ids
        token_ids_batches, reversed_token_ids_batches = y_token_ids

        feed_dict = self._fill_feed_dict(char_ids_batches, reversed_char_ids_batches,
                                         token_ids_batches, reversed_token_ids_batches)
        
        with self.current_graph.as_default():
            ret = self.sess.run([self.train_loss, self.train_op] + self.final_state_tensors, feed_dict)

        self.init_state_values = ret[2:]
        train_loss = ret[0]

        return train_loss

    def _build_model(self, train: bool, rebuild: bool, **kwargs):

        if rebuild:
            self.sess.close()
        if train:
            self.options = self._options.copy()
            self.options.update(self.train_options)

            self.models, self.train_op, self.train_loss, self.loss, self.current_graph =\
                self.train_model_graph['models'],\
                self.train_model_graph['train_op'],\
                self.train_model_graph['train_loss'],\
                self.train_model_graph['train_loss'],\
                self.train_model_graph['graph']
        else:
            self.options = self._options.copy()
            self.options.update(self.valid_options)

            self.models, self.train_op, self.train_loss, self.loss, self.current_graph =\
                self.valid_model_graph['models'],\
                self.valid_model_graph['train_op'],\
                self.valid_model_graph['train_loss'],\
                self.valid_model_graph['eval_loss'],\
                self.valid_model_graph['graph']
                
        with self.current_graph.as_default():
            self.init_state_values, self.init_state_tensors, self.final_state_tensors =\
                self._init_session()

    def process_event(self, event_name, data):
        if event_name == 'after_validation':
            self._build_model(train = True, rebuild = True, batch_size = 256, unroll_steps = 1)
        elif event_name == 'after_epoch':
            self._build_model(train = False, rebuild = True)
