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

from pathlib import Path
from datetime import datetime

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.paths import get_configs_path
from deeppavlov.core.common.file import read_json

LOGGER_CONFIG_FILENAME = 'dialog_logger_config.json'
LOG_TIMESTAMP_FORMAT = '%Y-%m-%d_%H-%M-%S_%f'

log = get_logger(__name__)


class DialogLogger:
    def __init__(self, agent_name: str = 'dp_agent'):
        self.config: dict = read_json(get_configs_path() / LOGGER_CONFIG_FILENAME)
        self.agent_name: str = agent_name
        self.log_dir: Path = Path(self.config['log_path']).resolve() / agent_name
        self.log_max_size: int = self.config['logfile_max_size_kb']
        self.log_file = self._get_log_file(self.log_dir, self.agent_name)

    @staticmethod
    def _get_timestamp_utc_str() -> str:
        utc_timestamp_str = datetime.strftime(datetime.utcnow(), LOG_TIMESTAMP_FORMAT)
        return utc_timestamp_str

    def _get_log_file(self, log_dir: Path, agent_name: str):
        log_file_path = Path(log_dir, f'{self._get_timestamp_utc_str()}_{agent_name}.log')
        log_file = open(log_file_path, 'wa', buffering=1)
        return log_file

    def log(self, log_message: str, direction: str, dialog_id: str = 'no_id'):
        if self.log_file.tell() >= self.log_max_size:
            self.log_file.close()
            self.log_file = self._get_log_file(self.log_dir, self.agent_name)
        else:
            try:
                log_str = f'{self._get_timestamp_utc_str()} {log_message}\n'
                print(log_str, file=self.log_file, flush=True)
            except IOError:
                log.error('Failed to write dialog log.')
