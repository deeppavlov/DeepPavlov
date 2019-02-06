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

from logging import getLogger
import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component

logger = getLogger(__name__)


@register("search_predictor")
class SearchPredictor(Component):

    def __init__(self,
                 **kwargs):
        self.sample_size = 5


    def __call__(self, responses, preds):
        sorted_ids = np.flip(np.argsort(preds[0]), -1)
        return [responses[0][np.random.choice(sorted_ids[:self.sample_size])]]  # choose a random answer as the best one