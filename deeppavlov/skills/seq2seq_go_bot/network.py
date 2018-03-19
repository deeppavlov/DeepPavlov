"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register("seq2seq_go_bot_nn")
class Seq2SeqGoalOrientedBotNetwork(TFModel):

    GRAPH_PARAMS = ['source_vocab_size', 'target_vocab_size', 'hidden_size']
    
    def __init__(self,
                 #embedder=None,
                 **params):
        # initialize parameters
        #self.embedder = embedder
        self._init_params(params)

        super().__init__(**params)

        # build computational graph
        self._build_graph()
        # initialize session
        self.sess = tf.Session()

        self.sess.run(tf.global_varibales_initializer())

        if tf.train.checkpoint_exists(str(self.save_path.resolve())):
            log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
            self.load()
        else:
            log.info("[initializing `{}` from scratch]".format(self.__class__.__name__))

    def _init_params(self, params):
        self.opt = params
        
        self.learning_rate = self.opt['learning_rate']
        self.src_vocab_size = self.opt['source_vocab_size']
        self.tgt_vocab_size = self.opt['target_vocab_size']
        #self.embedding_size = self.opt['embedding_size']
        self.hidden_size = self.opt['hidden_size']

    def _build_graph(self):

        self._add_placeholders()

        logits = self._build_body()
       
        _weights = tf.expand_dims(self._tgt_weights, -1)
        _loss_tensor = \
            tf.losses.sparse_softmax_cross_entropy(logits=_logits,
                                                   labels=self._decoder_outputs,
                                                   weights=_weights,
                                                   reduction=tf.losses.Reduction.NONE)
        # normalize loss by batch_size
        self._loss = tf.reduce_sum(_loss_tensor) / self._batch_size
        #self._loss = tf.reduce_mean(_loss_tensor, name='loss')
# TODO: tune clip_norm
        self._train_op = \
            self.get_train_op(self._loss, self.learning_rate, clip_norm=5.) 

    def _add_placeholders(self):
        # _encoder_inputs: [batch_size, max_input_time]
        self._encoder_inputs = tf.placeholder(tf.int32, 
                                              [None, None],
                                              name='encoder inputs')
        self._batch_size = tf.shape(self._encoder_inputs)[0]
        # _decoder_inputs: [batch_size, max_output_time]
        self._decoder_inputs = tf.placeholder(tf.int32, 
                                              [None, None],
                                              name='decoder inputs')
        # _decoder_outputs: [batch_size, max_output_time]
        self._decoder_outputs = tf.placeholder(tf.int32, 
                                               [None, None],
                                               name='decoder inputs')
#TODO: compute sequence lengths on the go
        # _src_sequence_length, _tgt_sequence_length: [batch_size]
        self._src_sequence_length = tf.placeholder(tf.int32,
                                                   [None],
                                                   name='input sequence lengths')
        self._tgt_sequence_length = tf.placeholder(tf.int32,
                                                   [None],
                                                   name='output sequence lengths')
        # _tgt_weights: [batch_size, max_output_time]
        self._tgt_weights = tf.placeholder(tf.int32,
                                           [None, None],
                                           name='target weights')

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
            _encoder_outputs, _encoder_state = tf.nn.dynamic_rnn(
                _encoder_cell, _encoder_emb_inp,
                sequence_length=self._src_sequence_length, time_major=False)

        with tf.variable_scope("Decoder"):
            _decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            # Helper
            _helper = tf.contrib.seq2seq.TrainingHelper(
                _decoder_emb_inp, self._tgt_sequence_lengths, time_major=False)
            # Output dense layer
            _projection_layer = \
                tf.python.layers.core.Dense(self.tgt_vocab_size, use_bias=False)
            # Decoder
            _decoder = tf.contrib.seq2seq.BasicDecoder(
                _decoder_cell, _helper, _encoder_state,
                output_layer=_projection_layer)
            # Dynamic decoding
# NOTE: pass extra arguments to dynamic_decode?
            _outputs, _ = tf.contrib.seq2seq.dynamic_decode(_decoder,
                                                            output_time_major=False)
            _logits = _outputs.rnn_output
        return _logits

    def __call__(self, features, prob=False):
        prediction ,probs = None, None
        if prob:
            return probs
        return prediction

    def train_on_batch(self, x: list, y: list):
        pass

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)
        self.load_params()

    def load_params(self):
        path = str(self.load_path.with_suffix('.json').resolve())
        log.info('[loading parameters from {}]'.format(path))
        with open(path, 'r') as fp:
            params = json.load(fp)
        for p in self.GRAPH_PARAMS:
            if self.opt[p] != params[p]:
                raise ConfigError("`{}` parameter must be equal to"
                                  " saved model parameter value `{}`"\
                                  .format(p, params[p]))

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        self.save_params()

    def save_params(self):
        path = str(self.save_path.with_suffix('.json').resolve())
        log.info('[saving parameters to {}]'.format(path))
        with open(path, 'w') as fp:
            json.dump(self.opt, fp)

    def shutdown(self):
        self.sess.close()
