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


import numpy as np
import sys
import hashlib
from typing import List

from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


def labels2onehot(labels: [list, np.ndarray], classes:  [list, np.ndarray]) -> np.ndarray:
    """
    Convert labels to one-hot vectors for multi-class multi-label classification

    Args:
        labels: list of samples where each sample is a list of classes which sample belongs with
        classes: array of classes' names

    Returns:
        2d array with one-hot representation of given samples
    """
    n_classes = len(classes)
    y = []
    for sample in labels:
        curr = np.zeros(n_classes)
        for intent in sample:
            if intent not in classes:
                log.warning('Unknown intent {} detected. Assigning no class'.format(intent))
            else:
                curr[np.where(np.array(classes) == intent)[0]] = 1
        y.append(curr)
    y = np.asarray(y)
    return y


def proba2labels(proba: [list, np.ndarray], confident_threshold: float, classes:  [list, np.ndarray]) -> np.ndarray:
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


def proba2onehot(proba: [list, np.ndarray], confident_threshold: float, classes:  [list, np.ndarray]) -> np.ndarray:
    """
    Convert vectors of probabilities to one-hot representations using confident threshold

    Args:
        proba: samples where each sample is a vector of probabilities to belong with given classes
        confident_threshold: boundary of probability to belong with a class
        classes: array of classes' names

    Returns:
        2d array with one-hot representation of given samples
    """
    return labels2onehot(proba2labels(proba, confident_threshold, classes), classes)


def log_metrics(names: [list, np.ndarray], values: [list, np.ndarray],
                updates: int = None, mode: str = 'train') -> None:
    """
    Print training and validation data in the following view:
        `mode -->	updates: 0   	names[0]: 0.0	names[1]: 0.0	names[2]: 0.0`

    Args:
        names: names of considered metrics
        values: values of considered metrics
        updates: number of updates
        mode: dataset field on which calculation is being doing (i.e "train")

    Returns:
        None
    """
    log.info("\r{} -->\t".format(mode))
    if updates is not None:
        log.info("updates: {}\t".format(updates))

    for id in range(len(names)):
        log.info("{}: {}\t".format(names[id], values[id]))
    return


def md5_hashsum(file_names: List[str]) -> str:
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
