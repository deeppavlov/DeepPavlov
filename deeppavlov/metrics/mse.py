# Copyright 2020 Neural Networks and Deep Learning lab, MIPT
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
from sklearn.metrics import mean_squared_error
from typing import Union

from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('mean_squared_error')
def mse(y_true: Union[np.array, list],
        y_predicted: Union[np.array, list],
        *args,
        **kwargs) -> float:
    """
    Calculates mean squared error.
    Args:
        y_true: list of true values
        y_predicted: list of predicted values
    Returns:
        float: Mean squared error
    """
    for value in [y_true, y_predicted]:
        assert (np.isfinite(value).all())
    return mean_squared_error(y_true, y_predicted, *args, **kwargs)
