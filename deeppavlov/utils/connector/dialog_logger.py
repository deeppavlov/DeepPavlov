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

import json
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Optional, Hashable

from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.core.data.utils import jsonify_data

LOGGER_CONFIG_FILENAME = 'dialog_logger_config.json'
LOG_TIMESTAMP_FORMAT = '%Y-%m-%d_%H-%M-%S_%f'

log = getLogger(__name__)


class DialogLogger:
    """DeepPavlov dialog logging facility.

    DialogLogger is an entity which provides tools for dialogs logging.

    Args:
        enabled: DialogLogger on/off flag.
        logger_name: Dialog logger name that is used for organising log files.

    Attributes:
        logger_name: Dialog logger name which is used for organising log files.
        log_max_size: Maximum size of log file, kb.
        self.log_file: Current log file object.
    """
    def __init__(self, enabled: bool = False, logger_name: Optional[str] = None) -> None:
        self.config: dict = read_json(get_settings_path() / LOGGER_CONFIG_FILENAME)
        self.enabled: bool = enabled or self.config['enabled']

        if self.enabled:
            self.logger_name: str = logger_name or self.config['logger_name']
            self.log_max_size: int = self.config['logfile_max_size_kb']
            self.log_file = self._get_log_file()
            self.log_file.writelines('"Dialog logger initiated"\n')

    @staticmethod
    def _get_timestamp_utc_str() -> str:
        """Returns str converted current UTC timestamp.

        Returns:
            utc_timestamp_str: str converted current UTC timestamp.
        """
        utc_timestamp_str = datetime.strftime(datetime.utcnow(), LOG_TIMESTAMP_FORMAT)
        return utc_timestamp_str

    def _get_log_file(self):
        """Returns opened file object for writing dialog logs.

        Returns:
            log_file: opened Python file object.
        """
        log_dir: Path = Path(self.config['log_path']).expanduser().resolve() / self.logger_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = Path(log_dir, f'{self._get_timestamp_utc_str()}_{self.logger_name}.log')
        log_file = open(log_file_path, 'a', buffering=1, encoding='utf8')
        return log_file

    def _log(self, utterance: Any, direction: str, dialog_id: Optional[Hashable]=None):
        """Logs single dialog utterance to current dialog log file.

        Args:
            utterance: Dialog utterance.
            direction: 'in' or 'out' utterance direction.
            dialog_id: Dialog ID.
        """
        if isinstance(utterance, str):
            pass
        elif isinstance(utterance, (list, dict)):
            utterance = jsonify_data(utterance)
        else:
            utterance = str(utterance)

        dialog_id = str(dialog_id) if not isinstance(dialog_id, str) else dialog_id

        if self.log_file.tell() >= self.log_max_size * 1024:
            self.log_file.close()
            self.log_file = self._get_log_file()
        else:
            try:
                log_msg = {}
                log_msg['timestamp'] = self._get_timestamp_utc_str()
                log_msg['dialog_id'] = dialog_id
                log_msg['direction'] = direction
                log_msg['message'] = utterance
                log_str = json.dumps(log_msg, ensure_ascii=self.config['ensure_ascii'])
                self.log_file.write(f'{log_str}\n')
            except IOError:
                log.error('Failed to write dialog log.')

    def log_in(self, utterance: Any, dialog_id: Optional[Hashable] = None) -> None:
        """Wraps _log method for all input utterances.
        Args:
            utterance: Dialog utterance.
            dialog_id: Dialog ID.
        """
        if self.enabled:
            self._log(utterance, 'in', dialog_id)

    def log_out(self, utterance: Any, dialog_id: Optional[Hashable] = None) -> None:
        """Wraps _log method for all output utterances.
        Args:
            utterance: Dialog utterance.
            dialog_id: Dialog ID.
        """
        if self.enabled:
            self._log(utterance, 'out', dialog_id)
