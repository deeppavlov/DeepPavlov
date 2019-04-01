'''
Train and test bidirectional language models.
'''

from typing import List
import os
import time
import json
import re
import glob

import tensorflow as tf
import numpy as np

# from bilm.training import LanguageModel 


from deeppavlov.models.bidirectional_lms.elmo.model import InferLanguageModel
from deeppavlov.models.bidirectional_lms.elmo.data import InferBatcher


DTYPE = 'float32'
DTYPE_INT = 'int64'

tf.logging.set_verbosity(tf.logging.INFO)


def load_model(options, ckpt_file, batch_size=256):
    '''
    Get the test set perplexity!
    '''

    bidirectional = options.get('bidirectional', False)
    char_inputs = 'char_cnn' in options
    if char_inputs:
        max_chars = options['char_cnn']['max_characters_per_token']

    unroll_steps = 1

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # with tf.Session(config=config) as sess:
    with tf.device('/gpu:0'), tf.variable_scope('lm'):
        test_options = dict(options)
        # NOTE: the number of tokens we skip in the last incomplete
        # batch is bounded above batch_size * unroll_steps
        test_options['batch_size'] = batch_size
        test_options['unroll_steps'] = 1
        # model = InferLanguageModel(test_options, False)
        model = InferLanguageModel(test_options, False)
        # we use the "Saver" class to load the variables
        loader = tf.train.Saver()
        loader.restore(sess, ckpt_file)

    # model.total_loss is the op to compute the loss
    # perplexity is exp(loss)
    init_state_tensors = model.init_lstm_state
    final_state_tensors = model.final_lstm_state
    if not char_inputs:
        feed_dict = {
            model.token_ids:
                    np.zeros([batch_size, unroll_steps], dtype=np.int64)
        }
        if bidirectional:
            feed_dict.update({
                model.token_ids_reverse:
                    np.zeros([batch_size, unroll_steps], dtype=np.int64)
            })
    else:
        feed_dict = {
            model.tokens_characters:
               np.zeros([batch_size, unroll_steps, max_chars],
                             dtype=np.int32)
        }
        if bidirectional:
            feed_dict.update({
                model.tokens_characters_reverse:
                    np.zeros([batch_size, unroll_steps, max_chars],
                        dtype=np.int32)
            })

    init_state_values = sess.run(
        init_state_tensors,
        feed_dict=feed_dict)

    return model, sess, init_state_tensors, init_state_values, final_state_tensors


def load_options_latest_checkpoint(tf_save_dir):
    ckpt_file = glob.glob(tf_save_dir+'/model.ckpt*.meta')[0][:-5]
    vocab_file = os.path.join(tf_save_dir, 'tokens_set.txt')
    options_file = os.path.join(tf_save_dir, 'options.json')
    # ckpt_file = tf.train.latest_checkpoint(tf_save_dir)

    with open(options_file, 'r') as fin:
        options = json.load(fin)

    # addition of options
    options['dropout'] = 0.0
    options['char_cnn']['n_characters'] = 261
    options['n_tokens_vocab'] = len(open(vocab_file).readlines())
    

    return options, ckpt_file, vocab_file
