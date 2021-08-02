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


from logging import getLogger
from typing import List

import numpy as np

log = getLogger(__name__)


def labels2onehot(labels: [List[str], List[List[str]], np.ndarray], classes: [list, np.ndarray]) -> np.ndarray:
    """
    Convert labels to one-hot vectors for multi-class multi-label classification

    Args:
        labels: list of samples where each sample is a class or a list of classes which sample belongs with
        classes: array of classes' names

    Returns:
        2d array with one-hot representation of given samples
    """
    n_classes = len(classes)
    y = []
    for sample in labels:
        curr = np.zeros(n_classes)
        if isinstance(sample, list):
            for intent in sample:
                if intent not in classes:
                    log.warning('Unknown label {} detected. Assigning no class'.format(intent))
                else:
                    curr[np.where(np.array(classes) == intent)[0]] = 1
        else:
            curr[np.where(np.array(classes) == sample)[0]] = 1
        y.append(curr)
    y = np.asarray(y)
    return y


def proba2labels(proba: [list, np.ndarray], confidence_threshold: float, classes: [list, np.ndarray]) -> List[List]:
    """
    Convert vectors of probabilities to labels using confident threshold
    (if probability to belong with the class is bigger than confidence_threshold, sample belongs with the class;
    if no probabilities bigger than confident threshold, sample belongs with the class with the biggest probability)

    Args:
        proba: list of samples where each sample is a vector of probabilities to belong with given classes
        confidence_threshold (float): boundary of probability to belong with a class
        classes: array of classes' names

    Returns:
        list of lists of labels for each sample
    """
    y = []
    for sample in proba:
        to_add = np.where(sample > confidence_threshold)[0]
        if len(to_add) > 0:
            y.append(np.array(classes)[to_add].tolist())
        else:
            y.append(np.array([np.array(classes)[np.argmax(sample)]]).tolist())

    return y


def proba2onehot(proba: [list, np.ndarray], confidence_threshold: float, classes: [list, np.ndarray]) -> np.ndarray:
    """
    Convert vectors of probabilities to one-hot representations using confident threshold

    Args:
        proba: samples where each sample is a vector of probabilities to belong with given classes
        confidence_threshold: boundary of probability to belong with a class
        classes: array of classes' names

    Returns:
        2d array with one-hot representation of given samples
    """
    return labels2onehot(proba2labels(proba, confidence_threshold, classes), classes)
