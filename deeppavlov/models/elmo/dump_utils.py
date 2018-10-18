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


import os
import json
import h5py
import re

import tensorflow as tf
import numpy as np

from deeppavlov.models.elmo.bilm_model import LanguageModel
from deeppavlov.models.elmo.elmo_model import BidirectionalLanguageModel

from deeppavlov.models.elmo.data import UnicodeCharsVocabulary, Batcher

DTYPE = 'float32'

tf.logging.set_verbosity(tf.logging.INFO)


def load_options_latest_checkpoint(tf_save_dir):
    options_file = os.path.join(tf_save_dir, 'options.json')
    ckpt_file = tf.train.latest_checkpoint(tf_save_dir)

    with open(options_file, 'r') as fin:
        options = json.load(fin)

    return options, ckpt_file


def dump_weights(tf_save_dir, outfile):
    '''
    Dump the trained weights from a model to a HDF5 file.
    '''

    def _get_outname(tf_name):
        outname = re.sub(':0$', '', tf_name)
        outname = outname.lstrip('lm/')
        outname = re.sub('/rnn/', '/RNN/', outname)
        outname = re.sub('/multi_rnn_cell/', '/MultiRNNCell/', outname)
        outname = re.sub('/cell_', '/Cell', outname)
        outname = re.sub('/lstm_cell/', '/LSTMCell/', outname)
        if '/RNN/' in outname:
            if 'projection' in outname:
                outname = re.sub('projection/kernel', 'W_P_0', outname)
            else:
                outname = re.sub('/kernel', '/W_0', outname)
                outname = re.sub('/bias', '/B', outname)
        return outname

    options, ckpt_file = load_options_latest_checkpoint(tf_save_dir)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.variable_scope('lm'):
            model = LanguageModel(options, False)
            # we use the "Saver" class to load the variables
            loader = tf.train.Saver()
            loader.restore(sess, ckpt_file)

        with h5py.File(outfile, 'w') as fout:   
            for v in tf.trainable_variables():
                if v.name.find('softmax') >= 0:
                    # don't dump these
                    continue
                outname = _get_outname(v.name)
                print("Saving variable {0} with name {1}".format(
                    v.name, outname))
                shape = v.get_shape().as_list()
                dset = fout.create_dataset(outname, shape, dtype='float32')
                values = sess.run([v])[0]
                dset[...] = values


def dump_token_embeddings(vocab_file, options_file, weight_file, outfile):
    '''
    Given an input vocabulary file, dump all the token embeddings to the
    outfile.  The result can be used as the embedding_weight_file when
    constructing a BidirectionalLanguageModel.
    '''
    with open(options_file, 'r') as fin:
        options = json.load(fin)
    max_word_length = options['char_cnn']['max_characters_per_token']

    vocab = UnicodeCharsVocabulary(vocab_file, max_word_length)
    batcher = Batcher(vocab_file, max_word_length)

    ids_placeholder = tf.placeholder('int32',
                                     shape=(None, None, max_word_length)
    )
    model = BidirectionalLanguageModel(options_file, weight_file)
    embedding_op = model(ids_placeholder)['token_embeddings']

    n_tokens = vocab.size
    embed_dim = int(embedding_op.shape[2])

    embeddings = np.zeros((n_tokens, embed_dim), dtype=DTYPE)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for k in range(n_tokens):
            token = vocab.id_to_word(k)
            char_ids = batcher.batch_sentences([[token]])[0, 1, :].reshape(
                1, 1, -1)
            embeddings[k, :] = sess.run(
                embedding_op, feed_dict={ids_placeholder: char_ids}
            )

    with h5py.File(outfile, 'w') as fout:
        ds = fout.create_dataset(
            'embedding', embeddings.shape, dtype='float32', data=embeddings
        )

def dump_bilm_embeddings(vocab_file, dataset_file, options_file,
                         weight_file, outfile):
    with open(options_file, 'r') as fin:
        options = json.load(fin)
    max_word_length = options['char_cnn']['max_characters_per_token']

    vocab = UnicodeCharsVocabulary(vocab_file, max_word_length)
    batcher = Batcher(vocab_file, max_word_length)

    ids_placeholder = tf.placeholder('int32',
                                     shape=(None, None, max_word_length)
    )
    model = BidirectionalLanguageModel(options_file, weight_file)
    ops = model(ids_placeholder)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sentence_id = 0
        with open(dataset_file, 'r') as fin, h5py.File(outfile, 'w') as fout:
            for line in fin:
                sentence = line.strip().split()
                char_ids = batcher.batch_sentences([sentence])
                embeddings = sess.run(
                    ops['lm_embeddings'], feed_dict={ids_placeholder: char_ids}
                )
                ds = fout.create_dataset(
                    '{}'.format(sentence_id),
                    embeddings.shape[1:], dtype='float32',
                    data=embeddings[0, :, :, :]
                )

                sentence_id += 1