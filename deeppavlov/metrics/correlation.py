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

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef

from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('pearson_correlation')
def pearson_correlation(y_true, y_predicted) -> float:
    return pearsonr(y_predicted, y_true)[0]


@register_metric('spearman_correlation')
def spearman_correlation(y_true, y_predicted) -> float:
    return spearmanr(y_predicted, y_true)[0]


@register_metric('matthews_correlation')
def matthews_correlation(y_true, y_predicted) -> float:
    return matthews_corrcoef(y_true, y_predicted)
