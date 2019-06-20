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


import math
from typing import List, Union

import numpy as np

from deeppavlov.core.common.metrics_registry import register_metric


def _dcg(relevance: Union[List[int], np.ndarray],
         order: Union[List[int], np.ndarray]) -> float:
    """
    Calculate discounted cumulative gain (DCG)

    Args:
        relevance: array of relevances
        order: predicted ranking

    Returns:
        A metric for the discounted cumulative gain
    """
    ordered_rel = [relevance[i] for i in order]
    dcg = sum([(math.pow(2, rel)-1) / math.log(i+1) for i, rel in enumerate(ordered_rel, start=1)])
    return dcg


@register_metric('ndcg')
def ndcg(relevance: Union[List[List[int]], np.ndarray],
         predictions: Union[List[List[int]], np.ndarray]) -> float:
    """
    Calculate average normalized discounted cumulative gain (nDCG) for the set of samples

    Args:
        relevance: array of relevances
        predictions: predicted scores

    Returns:
        A metric for the normalized discounted cumulative gain
    """
    ideal_order = [np.flip(np.argsort(el), -1) for el in relevance]
    order = [np.flip(np.argsort(el), -1) for el in predictions]
    dcg = [_dcg(rel, el) for rel, el in zip(relevance, order)]
    idcg = [_dcg(rel, el) for rel, el in zip(relevance, ideal_order)]
    ndcg = [d / i for d, i in zip(dcg, idcg) if i != 0]
    return sum(ndcg) / len(ndcg) if len(ndcg) else 0
