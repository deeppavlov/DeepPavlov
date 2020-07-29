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

from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('mean_squared_error')
def mse(y_true: Union[np.array, list],
        y_predicted: Union[np.array, list]):
    """
    Calculates mean squared error.
    Args:
        y_true: list of true probs
        y_predicted: list of predicted peobs
    Returns:
        F1 score
    """
    for value in [y_true, y_predicted]:
        assert (np.isfinite(value).all())
    y_true_np = np.array(y_true)
    y_predicted_np = np.array(y_predicted)
    assert y_true_np.ndim == y_predicted_np.ndim == 2
    err = ((y_true_np - y_predicted_np) ** 2).sum() ** 0.5
    return err
