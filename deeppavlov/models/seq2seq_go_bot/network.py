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

import json
import tensorflow as tf

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register("seq2seq_go_bot_nn")
class Seq2SeqGoalOrientedBotNetwork(TFModel):
    """
    The :class:`~deeppavlov.models.seq2seq_go_bot.bot.GoalOrientedBotNetwork` is a recurrent network that encodes user utterance and generates response in a sequence-to-sequence manner.

    For network architecture is similar to https://arxiv.org/abs/1705.05414 .

    Parameters:
        hidden_size: RNN hidden layer size.
        target_start_of_sequence_index: index of a start of sequence token during decoding.
        target_end_of_sequence_index: index of an end of sequence token during decoding.
        source_vocab_size: size of a vocabulary of encoder tokens.
        target_vocab_size: size of a vocabulary of decoder tokens.
        learning_rate: training learning rate.
        **kwargs: parameters passed to a parent :class:`~deeppavlov.core.models.tf_model.TFModel` class.
    """

    GRAPH_PARAMS = ['source_vocab_size', 'target_vocab_size', 'hidden_size']

    def __init__(self,
                 hidden_size: int,
                 source_vocab_size: int,
                 target_vocab_size: int,
                 target_start_of_sequence_index: int,
                 target_end_of_sequence_index: int,
                 learning_rate: float,
                 **kwargs) -> None:
        # specify model options
        self.opt = {
            'hidden_size': hidden_size,
            'source_vocab_size': source_vocab_size,
            'target_vocab_size': target_vocab_size,
            'target_start_of_sequence_index': target_start_of_sequence_index,
            'target_end_of_sequence_index': target_end_of_sequence_index,
            'learning_rate': learning_rate
        }
        # initialize parameters
        self._init_params()
        # build computational graph
        self._build_graph()
        # initialize session
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        super().__init__(**kwargs)

        if tf.train.checkpoint_exists(str(self.save_path.resolve())):
            log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
            self.load()
        else:
            log.info("[initializing `{}` from scratch]".format(self.__class__.__name__))

    def _init_params(self):
        self.hidden_size = self.opt['hidden_size']
        self.src_vocab_size = self.opt['source_vocab_size']
        self.tgt_vocab_size = self.opt['target_vocab_size']
        self.tgt_sos_id = self.opt['target_start_of_sequence_index']
        self.tgt_eos_id = self.opt['target_end_of_sequence_index']
        self.learning_rate = self.opt['learning_rate']

    def _build_graph(self):

        self._add_placeholders()

        _logits, self._predictions = self._build_body()

        _weights = tf.expand_dims(self._tgt_weights, -1)
        _loss_tensor = \
            tf.losses.sparse_softmax_cross_entropy(logits=_logits,
                                                   labels=self._decoder_outputs,
                                                   weights=_weights,
                                                   reduction=tf.losses.Reduction.NONE)
        # normalize loss by batch_size
        self._loss = tf.reduce_sum(_loss_tensor) / tf.cast(self._batch_size, tf.float32)
        #self._loss = tf.reduce_mean(_loss_tensor, name='loss')
# TODO: tune clip_norm
        self._train_op = \
            self.get_train_op(self._loss, self.learning_rate, clip_norm=10.) 

    def _add_placeholders(self):
        # _encoder_inputs: [batch_size, max_input_time]
        self._encoder_inputs = tf.placeholder(tf.int32, 
                                              [None, None],
                                              name='encoder_inputs')
        self._batch_size = tf.shape(self._encoder_inputs)[0]
        # _decoder_inputs: [batch_size, max_output_time]
        self._decoder_inputs = tf.placeholder(tf.int32, 
                                              [None, None],
                                              name='decoder_inputs')
        # _decoder_outputs: [batch_size, max_output_time]
        self._decoder_outputs = tf.placeholder(tf.int32, 
                                               [None, None],
                                               name='decoder_inputs')
#TODO: compute sequence lengths on the go
        # _src_sequence_lengths, _tgt_sequence_lengths: [batch_size]
        self._src_sequence_lengths = tf.placeholder(tf.int32,
                                                   [None],
                                                   name='input_sequence_lengths')
        self._tgt_sequence_lengths = tf.placeholder(tf.int32,
                                                   [None],
                                                   name='output_sequence_lengths')
        # _tgt_weights: [batch_size, max_output_time]
        self._tgt_weights = tf.placeholder(tf.int32,
                                           [None, None],
                                           name='target_weights')

    def _build_body(self):
#TODO: try learning embeddings
        # Encoder embedding
        #_encoder_embedding = tf.get_variable(
        #    "encoder_embedding", [self.src_vocab_size, self.embedding_size])
        #_encoder_emb_inp = tf.nn.embedding_lookup(_encoder_embedding,
        #                                          self._encoder_inputs)
        _encoder_emb_inp = tf.one_hot(self._encoder_inputs, self.src_vocab_size)
    
        # Decoder embedding
        #_decoder_embedding = tf.get_variable(
        #    "decoder_embedding", [self.tgt_vocab_size, self.embedding_size])
        #_decoder_emb_inp = tf.nn.embedding_lookup(_decoder_embedding,
        #                                          self._decoder_inputs)
        _decoder_emb_inp = tf.one_hot(self._decoder_inputs, self.tgt_vocab_size)

        with tf.variable_scope("Encoder"):
            _encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            # Run Dynamic RNN
            #   _encoder_outputs: [max_time, batch_size, num_units]
            #   _encoder_state: [batch_size, num_units]
# input_states?
            _encoder_outputs, _encoder_state = tf.nn.dynamic_rnn(
                _encoder_cell, _encoder_emb_inp, dtype=tf.float32,
                sequence_length=self._src_sequence_lengths, time_major=False)

        with tf.variable_scope("Decoder"):
            _decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            # Helper
            _helper = tf.contrib.seq2seq.TrainingHelper(
                _decoder_emb_inp, self._tgt_sequence_lengths, time_major=False)
            # Output dense layer
            _projection_layer = \
                tf.layers.Dense(self.tgt_vocab_size, use_bias=False)
            # Decoder
            _decoder = tf.contrib.seq2seq.BasicDecoder(
                _decoder_cell, _helper, _encoder_state,
                output_layer=_projection_layer)
            # Dynamic decoding
# NOTE: pass extra arguments to dynamic_decode?
# TRY: impute_finished = True,
            _outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(_decoder,
                                                            output_time_major=False)
            _logits = _outputs.rnn_output

        with tf.variable_scope("DecoderOnInfer"):
            _maximum_iterations = \
                tf.round(tf.reduce_max(self._src_sequence_lengths) * 2)
            # Helper
            _helper_infer = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                lambda d: tf.one_hot(d, self.tgt_vocab_size),
                tf.fill([self._batch_size], self.tgt_sos_id), self.tgt_eos_id)

            # Decoder
            _decoder_infer = tf.contrib.seq2seq.BasicDecoder(
                _decoder_cell, _helper_infer, _encoder_state,
                output_layer=_projection_layer)
            # Dynamic decoding
            _outputs_infer, _, _ = tf.contrib.seq2seq.dynamic_decode(
                _decoder_infer, maximum_iterations=_maximum_iterations)
            _predictions = _outputs_infer.sample_id
        return _logits, _predictions

    def __call__(self, enc_inputs, src_seq_lengths, prob=False):
        predictions = self.sess.run(
            self._predictions,
            feed_dict={
                self._encoder_inputs: enc_inputs,
                self._src_sequence_lengths: src_seq_lengths
            }
        )
# TODO: implement infer probabilities
        if prob:
            raise NotImplementedError("Probs not available for now.")
        return predictions
    
    def train_on_batch(self, enc_inputs, dec_inputs, dec_outputs, 
                       src_seq_lengths, tgt_seq_lengths, tgt_weights):
        _, loss_value = self.sess.run(
            [ self._train_op, self._loss ],
            feed_dict={
                self._encoder_inputs: enc_inputs,
                self._decoder_inputs: dec_inputs,
                self._decoder_outputs: dec_outputs,
                self._src_sequence_lengths: src_seq_lengths,
                self._tgt_sequence_lengths: tgt_seq_lengths,
                self._tgt_weights: tgt_weights
            }
        )
        return loss_value

    def load(self, *args, **kwargs):
        self.load_params()
        super().load(*args, **kwargs)

    def load_params(self):
        path = str(self.load_path.with_suffix('.json').resolve())
        log.info('[loading parameters from {}]'.format(path))
        with open(path, 'r', encoding='utf8') as fp:
            params = json.load(fp)
        for p in self.GRAPH_PARAMS:
            if self.opt.get(p) != params.get(p):
                raise ConfigError("`{}` parameter must be equal to saved model"
                                  " parameter value `{}`, but is equal to `{}`"
                                  .format(p, params.get(p), self.opt.get(p)))

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        self.save_params()

    def save_params(self):
        path = str(self.save_path.with_suffix('.json').resolve())
        log.info('[saving parameters to {}]'.format(path))
        with open(path, 'w', encoding='utf8') as fp:
            json.dump(self.opt, fp)

    def shutdown(self):
        self.sess.close()
