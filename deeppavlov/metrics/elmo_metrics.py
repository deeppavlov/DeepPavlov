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

from typing import List

import numpy as np

from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('elmo_loss2ppl')
def elmo_loss2ppl(losses: List[np.ndarray]) -> float:
    """ Calculates perplexity by loss

    Args:
        losses: list of numpy arrays of model losses

    Returns:
        perplexity : float
    """
    avg_loss = np.mean(losses)
    return float(np.exp(avg_loss))
