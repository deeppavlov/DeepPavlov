"""
File containing common operation with keras.backend objects
"""

import numpy as np

import keras.backend as kb

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


def make_pos_and_tag(tag, sep=" ", return_mode=None):
    if tag.endswith(" _"):
        tag = tag[:-2]
    if sep in tag:
        pos, tag = tag.split(sep, maxsplit=1)
    else:
        pos, tag = tag, ("_" if return_mode is None else "")
    if return_mode in ["dict", "list", "sorted_dict"]:
        tag = tag.split("|") if tag != "" else []
        if "dict" in return_mode:
            tag = dict(tuple(elem.split("=")) for elem in tag)
            if return_mode == "sorted_dict":
                tag = tuple(sorted(tag.items()))
    return pos, tag


def make_full_UD_tag(pos, tag, mode=None):
    if tag == "_" or len(tag) == 0:
        return pos
    if mode == "dict":
        tag, mode = sorted(tag.items()), "sorted_dict"
    if mode == "sorted_dict":
        tag, mode = ["{}={}".format(*elem) for elem in tag], "list"
    if mode == "list":
        tag = "|".join(tag)
    return "{},{}".format(pos, tag)


def _are_equal_pos(first, second):
    NOUNS, VERBS, CONJ = ["NOUN", "PROPN"], ["AUX", "VERB"], ["CCONJ", "SCONJ"]
    return (first == second or any((first in parts) and (second in parts)
                                   for parts in [NOUNS, VERBS, CONJ]))


IDLE_FEATURES = ["Voice", "Animacy", "Degree", "Mood", "VerbForm"]

def get_tag_distance(first, second, first_sep=",", second_sep=" "):
    first_pos, first_feats = make_pos_and_tag(first, sep=first_sep, return_mode="dict")
    second_pos, second_feats = make_pos_and_tag(second, sep=second_sep, return_mode="dict")
    dist = int(not _are_equal_pos(first_pos, second_pos))
    for key, value in first_feats.items():
        other = second_feats.get(key)
        if other is None:
            dist += int(key not in IDLE_FEATURES)
        else:
            dist += int(value != other)
    for key in second_feats:
        dist += int(key not in first_feats and key not in IDLE_FEATURES)
    return dist
