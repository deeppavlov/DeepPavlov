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


@register_metric('classification_report')
def roc_auc_score(y_true: Union[List[List[float]], List[List[int]], np.ndarray],
                  y_pred: Union[List[List[float]], List[List[int]], np.ndarray],
                  target_names: Union[List[List[str]], List[List[int]], np.ndarray]) -> float:
    try:
        return sklearn.metrics.classification_report(np.squeeze(np.array(y_true)),
                                                     np.squeeze(np.array(y_pred)),
                                                     np.squeeze(np.array(target_names)))
    except ValueError:
        return 0.
