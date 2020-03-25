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

"""File containing common operation with keras.backend objects"""

from typing import Union, Optional, Tuple

from tensorflow.keras import backend as K
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
    tile_factor = [1, k] + [1] * (K.ndim(x) - 1)
    return K.tile(x[:, None, :], tile_factor)


def make_pos_and_tag(tag: str, sep: str = ",",
                     return_mode: Optional[str] = None) -> Tuple[str, Union[str, list, dict, tuple]]:
    """
    Args:
        tag: the part-of-speech tag
        sep: the separator between part-of-speech tag and grammatical features
        return_mode: the type of return value, can be None, list, dict or sorted_items

    Returns:
        the part-of-speech label and grammatical features in required format
    """
    if tag.endswith(" _"):
        tag = tag[:-2]
    if sep in tag:
        pos, tag = tag.split(sep, maxsplit=1)
    else:
        pos, tag = tag, ("_" if return_mode is None else "")
    if return_mode in ["dict", "list", "sorted_items"]:
        tag = tag.split("|") if tag != "" else []
        if return_mode in ["dict", "sorted_items"]:
            tag = dict(tuple(elem.split("=")) for elem in tag)
            if return_mode == "sorted_items":
                tag = tuple(sorted(tag.items()))
    return pos, tag


def make_full_UD_tag(pos: str, tag: Union[str, list, dict, tuple],
                     sep: str = ",", mode: Optional[str] = None) -> str:
    """
    Args:
        pos: the part-of-speech label
        tag: grammatical features in the format, specified by 'mode'
        sep: the separator between part of speech and features in output tag
        mode: the input format of tag, can be None, list, dict or sorted_items

    Returns:
        the string representation of morphological tag
    """
    if tag == "_" or len(tag) == 0:
        return pos
    if mode == "dict":
        tag, mode = sorted(tag.items()), "sorted_items"
    if mode == "sorted_items":
        tag, mode = ["{}={}".format(*elem) for elem in tag], "list"
    if mode == "list":
        tag = "|".join(tag)
    return pos + sep + tag


def _are_equal_pos(first, second):
    NOUNS, VERBS, CONJ = ["NOUN", "PROPN"], ["AUX", "VERB"], ["CCONJ", "SCONJ"]
    return (first == second or any((first in parts) and (second in parts)
                                   for parts in [NOUNS, VERBS, CONJ]))


IDLE_FEATURES = {"Voice", "Animacy", "Degree", "Mood", "VerbForm"}


def get_tag_distance(first, second, first_sep=",", second_sep=" "):
    """
    Measures the distance between two (Russian) morphological tags in UD Format.
    The first tag is usually the one predicted by our model (therefore it uses comma
    as separator), while the second is usually the result of automatical conversion,
    where the separator is space.

    Args:
        first: UD morphological tag
        second: UD morphological tag (usually the output of 'russian_tagsets' converter)
        first_sep: separator between two parts of the first tag
        second_sep: separator between two parts of the second tag

    Returns:
        the number of mismatched feature values
    """
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
