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


from typing import List, Union

import numpy as np
from sklearn.metrics import log_loss

from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('log_loss')
def sk_log_loss(y_true: Union[List[List[float]], List[List[int]], np.ndarray],
                y_predicted: Union[List[List[float]], List[List[int]], np.ndarray]) -> float:
    """
    Calculates log loss.

    Args:
        y_true: list or array of true values
        y_predicted: list or array of predicted values

    Returns:
        Log loss
    """
    return log_loss(y_true, y_predicted)
