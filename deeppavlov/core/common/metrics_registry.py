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

import importlib
import json
from logging import getLogger
from pathlib import Path
from typing import Callable, Any

from deeppavlov.core.common.errors import ConfigError

log = getLogger(__name__)

_SKLEARN_METRICS = {"accuracy":"accuracy_score", 
                   "balanced_accuracy":"balanced_accuracy_score", 
                   "top_k_accuracy":"top_k_accuracy_score",
                   "average_precision":"average_precision_score",
                   "neg_brier_score":"brier_score_loss",
                   "f1":"f1_score",
                   "f1_micro":"f1_score",
                   "f1_macro":"f1_score",
                   "f1_weighted":"f1_score",
                   "f1_samples":"f1_score",
                   "neg_log_loss":"log_loss",                   
                   "precision":"precision_score",
                   "precision_micro":"precision_score",
                   "precision_macro":"precision_score",
                   "precision_weighted":"precision_score",
                   "precision_samples":"precision_score",
                   "recall":"recall_score",
                   "recall_micro":"recall_score",
                   "recall_macro":"recall_score",
                   "recall_weighted":"recall_score",
                   "recall_samples":"recall_score",
                   "jaccard":"jaccard_score",
                   "jaccard_micro":"jaccard_score",
                   "jaccard_macro":"jaccard_score",
                   "jaccard_weighted":"jaccard_score",
                   "jaccard_samples":"jaccard_score",                   
                   "roc_auc":"roc_auc_score",
                    "roc_auc_score":"roc_auc_score",
                   "roc_auc_ovr":"roc_auc_score",
                   "roc_auc_ovo":"roc_auc_score",
                   "roc_auc_ovr_weighted":"roc_auc_score",
                   "roc_auc_ovo_weighted":"roc_auc_score",
                   "adjusted_mutual_info_score":"adjusted_mutual_info_score",
                   "adjusted_rand_score":"adjusted_rand_score",
                   "completeness_score":"completeness_score",
                   "fowlkes_mallows_score":"fowlkes_mallows_score",
                   "homogeneity_score":"homogeneity_score",
                   "mutual_info_score":"mutual_info_score",                   
                   "normalized_mutual_info_score":"normalized_mutual_info_score",
                   "rand_score":"rand_score",
                   "v_measure_score":"v_measure_score",
                   "explained_variance":"explained_variance_score",
                   "max_error":"max_error",
                   "neg_mean_absolute_error":"mean_absolute_error",
                   "neg_mean_squared_error":"mean_squared_error",
                   "neg_root_mean_squared_error":"mean_squared_error",
                   "neg_mean_squared_log_error":"mean_squared_log_error",
                   "neg_median_absolute_error":"median_absolute_error",
                   "r2":"r2_score",
                   "neg_mean_poisson_deviance":"mean_poisson_deviance",
                   "neg_mean_gamma_deviance":"mean_gamma_deviance",
                   "neg_mean_absolute_percentage_error":"mean_absolute_percentage_error",                  
                   }

_registry_path = Path(__file__).parent / 'metrics_registry.json'
if _registry_path.exists():
    with _registry_path.open(encoding='utf-8') as f:
        _REGISTRY = json.load(f)
else:
    _REGISTRY = {}


def fn_from_str(name: str) -> Callable[..., Any]:
    """Returns a function object with the name given in string."""
    try:
        module_name, fn_name = name.split(':')
        if module_name == "sklearn":
            module_name = module_name + ".metrics"
    except ValueError:
        raise ConfigError('Expected function description in a `module.submodules:function_name` form, but got `{}`'
                          .format(name))

    return getattr(importlib.import_module(module_name), _SKLEARN_METRICS[fn_name])


def register_metric(metric_name: str) -> Callable[..., Any]:
    """Decorator for metric registration."""

    def decorate(fn):
        fn_name = fn.__module__ + ':' + fn.__name__
        if metric_name in _REGISTRY and _REGISTRY[metric_name] != fn_name:
            log.warning('"{}" is already registered as a metric name, the old function will be ignored'
                        .format(metric_name))
        _REGISTRY[metric_name] = fn_name
        return fn

    return decorate


def get_metric_by_name(name: str) -> Callable[..., Any]:
    """Returns a metric callable with a corresponding name."""
    name = _REGISTRY.get(name, name)
    return fn_from_str(name)
