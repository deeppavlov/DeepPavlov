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


import time

import tensorflow as tf
import numpy as np

from deeppavlov.models.elmo.bilm_model import LanguageModel
from deeppavlov.models.elmo.utils import get_feed_dict_from_X

DTYPE = 'float32'

tf.logging.set_verbosity(tf.logging.INFO)


def test(options, ckpt_file, data, batch_size=256):
    '''
    Get the test set perplexity!
    '''

    bidirectional = options.get('bidirectional', False)
    char_inputs = 'char_cnn' in options
    if char_inputs:
        max_chars = options['char_cnn']['max_characters_per_token']

    unroll_steps = 1

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.device('/gpu:0'), tf.variable_scope('lm'):
            test_options = dict(options)
            # NOTE: the number of tokens we skip in the last incomplete
            # batch is bounded above batch_size * unroll_steps
            test_options['batch_size'] = batch_size
            test_options['unroll_steps'] = 1
            model = LanguageModel(test_options, False)
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

        t1 = time.time()
        batch_losses = []
        total_loss = 0.0
        for batch_no, batch in enumerate(
                                data.iter_batches(batch_size, 1), start=1):
            # slice the input in the batch for the feed_dict
            X = batch

            feed_dict = {t: v for t, v in zip(
                                        init_state_tensors, init_state_values)}

            feed_dict.update(
                get_feed_dict_from_X(X, 0, X['token_ids'].shape[0], model,
                                          char_inputs, bidirectional)
            )

            ret = sess.run(
                [model.total_loss, final_state_tensors],
                feed_dict=feed_dict
            )

            loss, init_state_values = ret
            batch_losses.append(loss)
            batch_perplexity = np.exp(loss)
            total_loss += loss
            avg_perplexity = np.exp(total_loss / batch_no)

            print("batch=%s, batch_perplexity=%s, avg_perplexity=%s, time=%s" %
                (batch_no, batch_perplexity, avg_perplexity, time.time() - t1))

    avg_loss = np.mean(batch_losses)
    print("FINSIHED!  AVERAGE PERPLEXITY = %s" % np.exp(avg_loss))

    return np.exp(avg_loss)