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
from typing import Optional
from deeppavlov.core.commands.utils import expand_path

log = getLogger(__name__)


class TBWriter:
    def __init__(self, tensorboard_log_dir: str):
        # TODO: After adding wandb logger, create common parent class for both loggers
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_log_dir = expand_path(tensorboard_log_dir)
        self.tb_train_writer = SummaryWriter(str(tensorboard_log_dir / 'train_log'))
        self.tb_valid_writer = SummaryWriter(str(tensorboard_log_dir / 'valid_log'))

    # TODO: find how to write Summary
    def write_train(self, tag, scalar_value, global_step):
        self.tb_train_writer.add_scalar(tag, scalar_value, global_step)

    def write_valid(self, tag, scalar_value, global_step):
        self.tb_valid_writer.add_scalar(tag, scalar_value, global_step)

    def flush(self):
        self.tb_train_writer.flush()
        self.tb_valid_writer.flush()


def get_tb_writer(tensorboard_log_dir: Optional[str]) -> Optional[TBWriter]:
    try:
        if tensorboard_log_dir is not None:
            tb_writer = TBWriter(tensorboard_log_dir)
        else:
            tb_writer = None
    except ImportError:
        log.error('Failed to import SummaryWriter from torch.utils.tensorboard. Failed to initialize Tensorboard '
                  'logger. Install appropriate Pytorch version to use this logger or remove tensorboard_log_dir '
                  'parameter from the train parameters list in the configuration file.')
        tb_writer = None
    return tb_writer
