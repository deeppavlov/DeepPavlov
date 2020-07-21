# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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
from typing import Tuple, Optional, Iterable, Collection, Any

from deeppavlov.core.trainers.utils import Metric
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.trainers.nn_trainer import NNTrainer

log = getLogger(__name__)


@register('torch_trainer')
class TorchTrainer(NNTrainer):

    def test(self, data: Iterable[Tuple[Collection[Any], Collection[Any]]],
             metrics: Optional[Collection[Metric]] = None, *,
             start_time: Optional[float] = None, show_examples: Optional[bool] = None) -> dict:
        self._chainer.get_main_component().model.eval()

        report = super(TorchTrainer, self).test(data=data, metrics=metrics, start_time=start_time,
                                                show_examples=show_examples)
        self._chainer.get_main_component().model.train()
        return report

    def train_on_batches(self, iterator: DataLearningIterator) -> None:
        self._chainer.get_main_component().model.train()
        super(TorchTrainer, self).train_on_batches(iterator=iterator)
        self._chainer.get_main_component().model.eval()
