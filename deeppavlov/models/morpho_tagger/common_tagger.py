"""
File containing common operation with keras.backend objects
"""

import keras.backend as kb
import numpy as np

EPS = 1e-15

# AUXILIARY = ['PAD', 'BEGIN', 'END', 'UNKNOWN']
# AUXILIARY_CODES = PAD, BEGIN, END, UNKNOWN = 0, 1, 2, 3


def to_one_hot(x, k):
    """
    Takes an array of integers and transforms it
    to an array of one-hot encoded vectors
    """
    unit = np.eye(k, dtype=int)
    return unit[x]


def repeat_(x, k):
    tile_factor = [1, k] + [1] * (kb.ndim(x) - 1)
    return kb.tile(x[:, None, :], tile_factor)


def make_pos_and_tag(tag):
    if "," in tag:
        pos, tag = tag.split(",", maxsplit=1)
    else:
        pos, tag = tag, "_"
    return pos, tag
