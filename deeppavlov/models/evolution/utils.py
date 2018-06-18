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

import numpy as np
import sys
import hashlib

from keras.engine.topology import Layer
from deeppavlov.core.common.log import get_logger
from keras import initializers, regularizers, constraints
from keras import backend as K
from keras.layers import Reshape, Lambda, Dense, Flatten
from keras.layers import Concatenate, Multiply, Activation, Dot

log = get_logger(__name__)


def labels2onehot(labels, classes):
    """
    Convert labels to one-hot vectors for multi-class multi-label classification
    Args:
        labels: list of samples where each sample is a list of classes which sample belongs with
        classes: array of classes' names

    Returns:
        2d array with one-hot representation of given samples
    """
    n_classes = len(classes)
    eye = np.eye(n_classes)
    y = []
    for sample in labels:
        curr = np.zeros(n_classes)
        for intent in sample:
            if intent not in classes:
                log.warning('Unknown intent {} detected'.format(intent))
                curr += eye[np.where(np.array(classes) == 'unknown')[0]].reshape(-1)
            else:
                curr += eye[np.where(np.array(classes) == intent)[0]].reshape(-1)
        y.append(curr)
    y = np.asarray(y)
    return y


def proba2labels(proba, confident_threshold, classes):
    """
    Convert vectors of probabilities to labels using confident threshold
    (if probability to belong with the class is bigger than confident_threshold, sample belongs with the class;
    if no probabilities bigger than confident threshold, sample belongs with the class with the biggest probability)
    Args:
        proba: list of samples where each sample is a vector of probabilities to belong with given classes
        confident_threshold (float): boundary of probability to belong with a class
        classes: array of classes' names

    Returns:
        array of lists of labels for each sample
    """
    y = []
    for sample in proba:
        to_add = np.where(sample > confident_threshold)[0]
        if len(to_add) > 0:
            y.append(np.array(classes)[to_add])
        else:
            y.append(np.array([np.array(classes)[np.argmax(sample)]]))
    y = np.asarray(y)
    return y


def proba2onehot(proba, confident_threshold, classes):
    """
    Convert vectors of probabilities to one-hot representations using confident threshold
    Args:
        proba: list of samples where each sample is a vector of probabilities to belong with given classes
        confident_threshold: boundary of probability to belong with a class
        classes: array of classes' names

    Returns:
        2d array with one-hot representation of given samples
    """
    return labels2onehot(proba2labels(proba, confident_threshold, classes), classes)


def log_metrics(names, values, updates=None, mode='train'):
    """
    Print training and validation data in the following view:
        `mode -->	updates: 0   	names[0]: 0.0	names[1]: 0.0	names[2]: 0.0`
    Args:
        names: list of names of considered metrics
        values: list of values of considered metrics
        updates: number of updates
        mode: dataset field on which calculation is being doing (i.e "train")

    Returns:
        None
    """
    sys.stdout.write("\r")  # back to previous line
    log.info("{} -->\t".format(mode))
    if updates is not None:
        log.info("updates: {}\t".format(updates))

    for id in range(len(names)):
        log.info("{}: {}\t".format(names[id], values[id]))
    return


def md5_hashsum(file_names):
    """
    Calculate md5 hash sum of files listed
    Args:
        file_names: list of file names

    Returns:
        hashsum string
    """
    hash_md5 = hashlib.md5()
    for file_name in file_names:
        with open(file_name, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()


class Attention(Layer):
    def __init__(self, context_length=None,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 use_bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.use_bias = use_bias
        self.context_length = context_length

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        if self.context_length is None:
            self.context_length = input_shape[-1]

        self.context = self.add_weight(tuple((self.context_length, input_shape[-1])),
                                       name="context",
                                       initializer=self.init)

        self.W = self.add_weight((2 * input_shape[-1], 1, ),
                                 name="w",
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        if self.use_bias:
            self.b = self.add_weight((1, ),
                                     name="b",
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def call(self, x, mask=None):

        expanded_context_3d = expand_tile_batch_size(memory=x, context=self.context)
        expanded_context_4d = expand_tile(expanded_context_3d, axis=1, n_repetitions=K.int_shape(x)[1])
        expanded_x = expand_tile(x, axis=2, n_repetitions=K.int_shape(expanded_context_3d)[1])

        # now expanded_context_4d and expanded_x are of
        # shape (bs, time_steps, context_size, n_features)

        x_full = Concatenate(axis=-1)([expanded_x, expanded_context_4d])

        out = K.dot(x_full, self.W)
        if self.use_bias:
            out = K.bias_add(out, self.b)

        out = Activation('softmax')(out)
        out = Multiply()([out, expanded_x])

        out = Lambda(lambda x: K.sum(x, axis=1))(out)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape

def expand_tile(units, axis, n_repetitions=None):
    """Expand and tile tensor along given axis
    Args:
        units: tf tensor with dimensions [batch_size, time_steps, n_input_features]
        axis: axis along which expand and tile. Must be 1 or 2

    """
    assert axis in (1, 2)
    repetitions = [1] * (len(K.int_shape(units)) + 1)

    if n_repetitions is None:
        repetitions[axis] = K.int_shape(units)[1]
    else:
        repetitions[axis] = n_repetitions

    if axis == 1:
        expanded = Reshape(target_shape=( (1,) + K.int_shape(units)[1:] ))(units)
    else: # axis=2
        expanded = Reshape(target_shape=(K.int_shape(units)[1:2] + (1,) + K.int_shape(units)[2:]))(units)
    return K.tile(expanded, repetitions)


def expand_tile_batch_size(memory, context):
    """Expand and tile tensor context along 0 axis up to 0-shape of memory
    Args:
        memory: tf tensor with dimensions [batch_size, time_steps, n_input_features]
        context: tf tensor with dimensions [new_time_steps, n_input_features]

    """
    axis = 0
    # batch_size = K.int_shape(memory)[0]
    batch_size = K.shape(memory)[0]
    repetitions = [1] * len(K.int_shape(memory))
    repetitions[axis] = batch_size
    if axis == 0:
        expanded = K.reshape(context, shape=((1,) + K.int_shape(context)))
    return K.tile(expanded, repetitions)

