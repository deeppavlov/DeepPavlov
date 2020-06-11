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
import logging
import logging.config
from pathlib import Path

from .paths import get_settings_path

LOG_CONFIG_FILENAME = 'log_config.json'
TRACEBACK_LOGGER_ERRORS = True

root_path = Path(__file__).resolve().parents[3]

log_config_path = get_settings_path() / LOG_CONFIG_FILENAME

with log_config_path.open(encoding='utf8') as log_config_json:
    log_config = json.load(log_config_json)


class ProbeFilter(logging.Filter):
    """ProbeFilter class is used to filter POST requests to /probe endpoint from logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        """To log the record method should return True."""
        return 'POST /probe HTTP' not in record.getMessage()


def init_logger():
    configured_loggers = [log_config.get('root', {})] + [logger for logger in
                                                         log_config.get('loggers', {}).values()]

    used_handlers = {handler for log in configured_loggers for handler in log.get('handlers', [])}

    for handler_id, handler in list(log_config['handlers'].items()):
        if handler_id not in used_handlers:
            del log_config['handlers'][handler_id]
        elif 'filename' in handler.keys():
            filename = handler['filename']
            logfile_path = Path(filename).expanduser().resolve()
            handler['filename'] = str(logfile_path)

    logging.config.dictConfig(log_config)
