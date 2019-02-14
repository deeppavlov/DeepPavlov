# originally based on https://github.com/allenai/bilm-tf/blob/master/bilm/training.py

# Modifications copyright 2017 Neural Networks and Deep Learning lab, MIPT
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

import re

import h5py
import tensorflow as tf

from deeppavlov.models.elmo.bilm_model import LanguageModel

tf.logging.set_verbosity(tf.logging.INFO)


def average_gradients(tower_grads, batch_size, options):
    # calculate average gradient for each shared variable across all GPUs
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        # We need to average the gradients across each GPU.

        g0, v0 = grad_and_vars[0]

        if g0 is None:
            # no gradient for this variable, skip it
            average_grads.append((g0, v0))
            continue

        if isinstance(g0, tf.IndexedSlices):
            # If the gradient is type IndexedSlices then this is a sparse
            #   gradient with attributes indices and values.
            # To average, need to concat them individually then create
            #   a new IndexedSlices object.
            indices = []
            values = []
            for g, v in grad_and_vars:
                indices.append(g.indices)
                values.append(g.values)
            all_indices = tf.concat(indices, 0)
            avg_values = tf.concat(values, 0) / len(grad_and_vars)
            # deduplicate across indices
            av, ai = _deduplicate_indexed_slices(avg_values, all_indices)
            grad = tf.IndexedSlices(av, ai, dense_shape=g0.dense_shape)

        else:
            # a normal tensor can just do a simple average
            grads = []
            for g, v in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

        # the Variables are redundant because they are shared
        # across towers. So.. just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)

    assert len(average_grads) == len(list(zip(*tower_grads)))

    return average_grads


def summary_gradient_updates(grads, opt, lr):
    """get summary ops for the magnitude of gradient updates"""

    # strategy:
    # make a dict of variable name -> [variable, grad, adagrad slot]
    vars_grads = {}
    for v in tf.trainable_variables():
        vars_grads[v.name] = [v, None, None]
    for g, v in grads:
        vars_grads[v.name][1] = g
        vars_grads[v.name][2] = opt.get_slot(v, 'accumulator')

    # now make summaries
    ret = []
    for vname, (v, g, a) in vars_grads.items():

        if g is None:
            continue

        if isinstance(g, tf.IndexedSlices):
            # a sparse gradient - only take norm of params that are updated
            updates = lr * g.values
            if a is not None:
                updates /= tf.sqrt(tf.gather(a, g.indices))
        else:
            updates = lr * g
            if a is not None:
                updates /= tf.sqrt(a)

        values_norm = tf.sqrt(tf.reduce_sum(v * v)) + 1.0e-7
        updates_norm = tf.sqrt(tf.reduce_sum(updates * updates))
        ret.append(tf.summary.scalar('UPDATE/' + vname.replace(":", "_"), updates_norm / values_norm))

    return ret


def _deduplicate_indexed_slices(values, indices):
    """Sums `values` associated with any non-unique `indices`.
    Args:
      values: A `Tensor` with rank >= 1.
      indices: A one-dimensional integer `Tensor`, indexing into the first
      dimension of `values` (as in an IndexedSlices object).
    Returns:
      A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
      de-duplicated version of `indices` and `summed_values` contains the sum of
      `values` slices associated with each unique index.
    """
    unique_indices, new_index_positions = tf.unique(indices)
    summed_values = tf.unsorted_segment_sum(values,
                                            new_index_positions,
                                            tf.shape(unique_indices)[0])
    return (summed_values, unique_indices)


def clip_by_global_norm_summary(t_list, clip_norm, norm_name, variables):
    # wrapper around tf.clip_by_global_norm that also does summary ops of norms

    # compute norms
    # use global_norm with one element to handle IndexedSlices vs dense
    norms = [tf.global_norm([t]) for t in t_list]

    # summary ops before clipping
    summary_ops = []
    for ns, v in zip(norms, variables):
        name = 'norm_pre_clip/' + v.name.replace(":", "_")
        summary_ops.append(tf.summary.scalar(name, ns))

    # clip
    clipped_t_list, tf_norm = tf.clip_by_global_norm(t_list, clip_norm)

    # summary ops after clipping
    norms_post = [tf.global_norm([t]) for t in clipped_t_list]
    for ns, v in zip(norms_post, variables):
        name = 'norm_post_clip/' + v.name.replace(":", "_")
        summary_ops.append(tf.summary.scalar(name, ns))

    summary_ops.append(tf.summary.scalar(norm_name, tf_norm))

    return clipped_t_list, tf_norm, summary_ops


def clip_grads(grads, options, do_summaries, global_step):
    # grads = [(grad1, var1), (grad2, var2), ...]
    def _clip_norms(grad_and_vars, val, name):
        # grad_and_vars is a list of (g, v) pairs
        grad_tensors = [g for g, v in grad_and_vars]
        vv = [v for g, v in grad_and_vars]
        scaled_val = val
        if do_summaries:
            clipped_tensors, g_norm, so = clip_by_global_norm_summary(
                grad_tensors, scaled_val, name, vv)
        else:
            so = []
            clipped_tensors, g_norm = tf.clip_by_global_norm(
                grad_tensors, scaled_val)

        ret = []
        for t, (g, v) in zip(clipped_tensors, grad_and_vars):
            ret.append((t, v))

        return ret, so

    all_clip_norm_val = options['all_clip_norm_val']
    ret, summary_ops = _clip_norms(grads, all_clip_norm_val, 'norm_grad')

    assert len(ret) == len(grads)

    return ret, summary_ops


def safely_str2int(in_str: str):
    try:
        i = int(in_str)
    except ValueError:
        i = None
    return i


def dump_weights(tf_save_dir, outfile, options):
    """
    Dump the trained weights from a model to a HDF5 file.
    """

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

    ckpt_file = tf.train.latest_checkpoint(tf_save_dir)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            with tf.variable_scope('lm'):
                LanguageModel(options, False)  # Create graph
                # we use the "Saver" class to load the variables
                loader = tf.train.Saver()
                loader.restore(sess, ckpt_file)

            with h5py.File(outfile, 'w') as fout:
                for v in tf.trainable_variables():
                    if v.name.find('softmax') >= 0:
                        # don't dump these
                        continue
                    outname = _get_outname(v.name)
                    # print("Saving variable {0} with name {1}".format(
                    #     v.name, outname))
                    shape = v.get_shape().as_list()
                    dset = fout.create_dataset(outname, shape, dtype='float32')
                    values = sess.run([v])[0]
                    dset[...] = values
