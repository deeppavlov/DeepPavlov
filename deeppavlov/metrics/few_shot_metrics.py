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


from typing import List, Union, Optional

import numpy as np
from deeppavlov.core.common.metrics_registry import register_metric
from sklearn.metrics import accuracy_score, \
                            balanced_accuracy_score, \
                            precision_recall_fscore_support, \
                            classification_report

def delete_oos(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ind_mask = np.where(y_true == 'oos')
    
    y_true = np.delete(y_true, ind_mask, 0)
    y_pred = np.delete(y_pred, ind_mask, 0)
    return y_true, y_pred

@register_metric('sklearn_accuracy')
def accuracy(y_true, y_pred, exclude_oos: bool = False) -> float:
    if exclude_oos:
        y_true, y_pred = delete_oos(y_true, y_pred)
    return accuracy_score(y_true, y_pred)


@register_metric('sklearn_balanced_accuracy')
def balanced_accuracy(y_true, y_pred, exclude_oos: bool = False) -> float:
    if exclude_oos:
        y_true, y_pred = delete_oos(y_true, y_pred)

    return balanced_accuracy_score(y_true, y_pred)


@register_metric('oos_scores')
def oos_scores(y_true, y_pred):
    y_true_binary = (np.array(y_true) == "oos")
    y_pred_binary = (np.array(y_pred) == "oos")
    scores = precision_recall_fscore_support(y_true_binary, y_pred_binary, average='binary')
    return dict(zip(["precision", "recall", "fbeta_score"], scores[:3]))
