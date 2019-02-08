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
import sklearn.metrics

from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('roc_auc')
def roc_auc_score(y_true: Union[List[List[float]], List[List[int]], np.ndarray],
                  y_pred: Union[List[List[float]], List[List[int]], np.ndarray]) -> float:
    """
    Compute Area Under the Curve (AUC) from prediction scores.

    Args:
        y_true: true binary labels
        y_pred: target scores, can either be probability estimates of the positive class

    Returns:
        Area Under the Curve (AUC) from prediction scores
    """
    try:
        return sklearn.metrics.roc_auc_score(np.squeeze(np.array(y_true)),
                                             np.squeeze(np.array(y_pred)), average="macro")
    except ValueError:
        return 0.
