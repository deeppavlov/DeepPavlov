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

from typing import List, Tuple, Optional

import tensorflow as tf
import numpy as np
import json
import time

from functools import partial

from deeppavlov.core.layers.tf_layers import embedding_layer, character_embedding_network, variational_dropout
from deeppavlov.core.layers.tf_layers import cudnn_bi_lstm, cudnn_bi_gru, bi_rnn, stacked_cnn, INITIALIZER
# from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.commands.utils import expand_path

from deeppavlov.models.elmo.bilm_model import LanguageModel
from deeppavlov.models.elmo.train_utils import print_variable_summary, average_gradients, clip_grads, summary_gradient_updates

log = get_logger(__name__)


# class ELMo(TFModel): # TODO: Add TFModel inheritance
@register('elmo_model')
class ELMo(NNModel):
    """>>
    The :class:`~deeppavlov.models.ner.network.NerNetwork` is for Neural Named Entity Recognition and Slot Filling.

    Parameters:
        n_tags: Number of tags in the tag vocabulary.
        token_emb_dim: Dimensionality of token embeddings, needed if embedding matrix is not provided.
        char_emb_dim: Dimensionality of token embeddings.
        capitalization_dim : Dimensionality of capitalization features, if they are provided.
        pos_features_dim: Dimensionality of POS features, if they are provided.
        additional_features: Some other features.
        net_type: Type of the network, either ``'rnn'`` or ``'cnn'``.
        cell_type: Type of the cell in RNN, either ``'lstm'`` or ``'gru'``.
        use_cudnn_rnn: Whether to use CUDNN implementation of RNN.
        two_dense_on_top: Additional dense layer before predictions.
        n_hidden_list: A list of output feature dimensionality for each layer. A value (100, 200) means that there will
            be two layers with 100 and 200 units, respectively.
        cnn_filter_width: The width of the convolutional kernel for Convolutional Neural Networks.
        use_crf: Whether to use Conditional Random Fields on top of the network (recommended).
        token_emb_mat: Token embeddings matrix.
        char_emb_mat: Character embeddings matrix.
        use_batch_norm: Whether to use Batch Normalization or not. Affects only CNN networks.
        dropout_keep_prob: Probability of keeping the hidden state, values from 0 to 1. 0.5 works well in most cases.
        embeddings_dropout: Whether to use dropout on embeddings or not.
        top_dropout: Whether to use dropout on output units of the network or not.
        intra_layer_dropout: Whether to use dropout between layers or not.
        l2_reg: L2 norm regularization for all kernels.
        clip_grad_norm: Clip the gradients by norm.
        learning_rate: Learning rate to use during the training (usually from 0.1 to 0.0001)
        gpu: Number of gpu to use.
        seed: Random seed.
        lr_drop_patience: How many epochs to wait until drop the learning rate.
        lr_drop_value: Amount of learning rate drop.
    """
    GRAPH_PARAMS = ["n_tags",  # TODO: add check
                    "char_emb_dim",
                    "capitalization_dim",
                    "additional_features",
                    "use_char_embeddings",
                    "additional_features",
                    "net_type",
                    "cell_type",
                    "char_filter_width",
                    "cell_type"]

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
                 learning_rate: float = 2e-1, # For AdagradOptimizer
                 initial_accumulator_value: float = 1.0,
                 seed: int = None, # Other
                 n_gpus: int = 1, # TODO: Add cpu supporting
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

        # ================== Building the network ==================

        self.models, self.train_op, _, _ = self._build_graph()
        # self.models, self.train_op, self.summary_op, self.hist_summary_op = self._build_graph()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        
        self.batch_no = 0
        self.batch_time = time.time()

        # ================= Initialize the session =================
        self.summary_writer, self.init_state_values, self.init_state_tensors, self.final_state_tensors =\
            self._init_session()
        # data_gen = data.iter_batches(batch_size * self._options['n_gpus'], self._options['unroll_steps'])

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
            # train_perplexity = tf.get_variable(
            #     'train_perplexity', [],
            #     initializer=tf.constant_initializer(0.0), trainable=False)
            for k in range(self._options['n_gpus']):
                with tf.device('/gpu:%d' % k):
                    with tf.variable_scope('lm', reuse=k > 0):
                        # calculate the loss for one model replica and get
                        #   lstm states
                        model = LanguageModel(self._options, True)
                        loss = model.total_loss
                        models.append(model)
                        # get gradients
                        grads = opt.compute_gradients(
                            loss * self._options['unroll_steps'],
                            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                        )
                        tower_grads.append(grads)
                        # # keep track of loss across all GPUs
                        # train_perplexity += loss
                print_variable_summary()

            # calculate the mean of each gradient across all GPUs
            grads = average_gradients(tower_grads, self._options['batch_size'], self._options)
            grads, _ = clip_grads(grads, self._options, True, global_step)
            # grads, norm_summary_ops = clip_grads(grads, self._options, True, global_step)
            # norm_summaries = []
            # norm_summaries.extend(norm_summary_ops)

            # log the training perplexity
            # train_perplexity = tf.exp(train_perplexity / self._options['n_gpus'])
            # perplexity_summmary = tf.summary.scalar(
            #     'train_perplexity', train_perplexity)

            # # some histogram summaries.  all models use the same parameters
            # # so only need to summarize one
            # histogram_summaries = [
            #     tf.summary.histogram('token_embedding', models[0].embedding)
            # ]
            # # tensors of the output from the LSTM layer
            # lstm_out = tf.get_collection('lstm_output_embeddings')
            # histogram_summaries.append(
            #         tf.summary.histogram('lstm_embedding_0', lstm_out[0]))
            # if self._options.get('bidirectional', False):
            #     # also have the backward embedding
            #     histogram_summaries.append(
            #         tf.summary.histogram('lstm_embedding_1', lstm_out[1]))

            # apply the gradients to create the training operation
            train_op = opt.apply_gradients(grads, global_step=global_step)

            # # histograms of variables
            # for v in tf.global_variables():
            #     histogram_summaries.append(tf.summary.histogram(v.name.replace(":", "_"), v))

            # # get the gradient updates -- these aren't histograms, but we'll
            # # only update them when histograms are computed
            # histogram_summaries.extend(
            #     summary_gradient_updates(grads, opt, self._options['learning_rate']))

            # summary_op = tf.summary.merge(
            #     [perplexity_summmary] + norm_summaries
            # )
            # hist_summary_op = tf.summary.merge(histogram_summaries)
            hist_summary_op = None
            summary_op = None

        return models, train_op, summary_op, hist_summary_op

    def load(self):
        pass
    def save(self):
        pass

    def _init_session(self):
        restart_ckpt_file = None # TODO: It is one too
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        # if self._options['n_gpus'] is not None:
        #     sess_config.gpu_options.visible_device_list = str(self._options['n_gpus']-1)
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.initialize_all_variables())

        # load the checkpoint data if needed
        if restart_ckpt_file is not None:
            loader = tf.train.Saver()
            loader.restore(self.sess, restart_ckpt_file)

        # summary_writer = tf.summary.FileWriter(tf_log_dir, self.sess.graph)
        summary_writer = None

        # For each batch:
        # Get a batch of data from the generator. The generator will
        # yield batches of size batch_size * n_gpus that are sliced
        # and fed for each required placeholer.
        #
        # We also need to be careful with the LSTM states.  We will
        # collect the final LSTM states after each batch, then feed
        # them back in as the initial state for the next batch

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
        return summary_writer, init_state_values, init_state_tensors, final_state_tensors

    def _fill_feed_dict(self,char_ids_batches, reversed_char_ids_batches, token_ids_batches = None, reversed_token_ids_batches = None, learning_rate=None, train=False):
        # init state tensors
        feed_dict = {t: v for t, v in zip(
                                    self.init_state_tensors, self.init_state_values)}
        # # pprint.pprint(f"char_ids_batches {char_ids_batches.shape}")
        # # pprint.pprint(f"reversed_char_ids_batches {reversed_char_ids_batches.shape}")
        # # pprint.pprint(f"token_ids_batches {token_ids_batches.shape}")
        # # pprint.pprint(f"reversed_token_ids_batches {reversed_token_ids_batches.shape}")
        for k, model in enumerate(self.models):
            start = k * self._options['batch_size']
            end = (k + 1) * self._options['batch_size']


            # character inputs
            char_ids = char_ids_batches[start:end] # get char_ids
            # # pprint.pprint(f"char_ids_batches[start:end] {char_ids_batches[start:end].shape}")
            feed_dict[model.tokens_characters] = char_ids

            if self._options['bidirectional']:
                feed_dict[model.tokens_characters_reverse] = \
                    reversed_char_ids_batches[start:end] # get tokens_characters_reverse
                
                # # pprint.pprint(f"reversed_char_ids_batches[start:end] {reversed_char_ids_batches[start:end].shape}")

            if token_ids_batches is not None:
                # # pprint.pprint(f"token_ids_batches[start:end] {token_ids_batches[start:end].shape}")
                # pprint.pprint(f"reversed_token_ids_batches[start:end] {reversed_token_ids_batches[start:end].shape}")
                # now the targets with weights
                feed_dict[model.next_token_id] = token_ids_batches[start:end] # get next_token_id
                if self._options['bidirectional']:
                    feed_dict[model.next_token_id_reverse] = reversed_token_ids_batches[start:end] # get next_token_id_reverse

            # if learning_rate is not None:
            #     feed_dict[self.learning_rate_ph] = learning_rate
            # feed_dict[self.training_ph] = train
            #     if not train:
            #         feed_dict[self._dropout_ph] = 1.0
        return feed_dict


    def __call__(self, *args, **kwargs):
        # if len(args[0]) == 0 or (len(args[0]) == 1 and len(args[0][0]) == 0):
        #     return []
        # return self.predict(args)
        return None

    def train_on_batch(self, char_ids_batches:list, reversed_char_ids_batches:list, token_ids_batches:list, reversed_token_ids_batches:list):
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
        self.batch_no +=1
        if char_ids_batches:
            print(f'batch_no {self.batch_no}')
            print(f'share time {time.time() - self.batch_time }')
            self.batch_time = time.time()

            print(f'len(char_ids_batches) {len(char_ids_batches)}')
        for c_ids, rc_ids, t_ids, rt_ids in zip(char_ids_batches, reversed_char_ids_batches, token_ids_batches, reversed_token_ids_batches):
            feed_dict = self._fill_feed_dict(c_ids, rc_ids, t_ids, rt_ids, train=True)
            # import pdb; pdb.set_trace()
            ret = self.sess.run([self.train_op] + self.final_state_tensors, feed_dict)

            self.init_state_values = ret[1:]


    # def process_event(self, event_name, data):
    #     if event_name == 'after_validation':
    #         if not hasattr(self, '_best_f1'):
    #             self._best_f1 = 0
    #         if not hasattr(self, '_impatience'):
    #             self._impatience = 0
    #         if data['metrics']['ner_f1'] > self._best_f1:
    #             self._best_f1 = data['metrics']['ner_f1']
    #             self._impatience = 0
    #         else:
    #             self._impatience += 1

    #         if self._impatience >= self._lr_drop_patience:
    #             self._impatience = 0
    #             log.info('Dropping learning rate from {:.1e} to {:.1e}'.format(self._learning_rate,
    #                                                                            self._learning_rate * self._lr_drop_value))
    #             self.load()
    #             self._learning_rate *= self._lr_drop_value
