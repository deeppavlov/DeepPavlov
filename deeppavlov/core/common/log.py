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
import logging.config
import sys
from pathlib import Path

LOG_CONFIG_FILENAME = 'log_config.json'
TRACEBACK_LOGGER_ERRORS = True

root_path = Path(__file__).resolve().parent.parent.parent.parent


def get_configs_path() -> Path:
    with open(root_path / 'deeppavlov/paths.json', encoding='utf8') as fin:
        paths = json.load(fin)

    configs_paths = Path(paths['configs_path']).resolve() if paths['configs_path'][0] == '/' \
        else root_path / paths['configs_path']

    return configs_paths


def get_logger(logger_name):
    try:
        log_config_path = Path(get_configs_path(), LOG_CONFIG_FILENAME).resolve()

        with open(log_config_path, encoding='utf8') as log_config_json:
            log_config = json.load(log_config_json)

        configured_loggers = [log_config.get('root', {})] + log_config.get('loggers', [])
        used_handlers = {handler for log in configured_loggers for handler in log.get('handlers', [])}

        for handler_id, handler in list(log_config['handlers'].items()):
            if handler_id not in used_handlers:
                del log_config['handlers'][handler_id]
            elif 'filename' in handler.keys():
                filename = handler['filename']
                logfile_path = Path(filename).expanduser().resolve()
                handler['filename'] = str(logfile_path)

        logging.config.dictConfig(log_config)
        logger = logging.getLogger(logger_name)

    except Exception:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)

        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)d %(levelname)s in \'%(name)s\'[\'%(module)s\'] at line %(lineno)d: %(message)s',
            '%Y-%m-%d %H:%M:%S')

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        handler.setLevel(logging.WARNING)

        logger.addHandler(handler)

        logger.error(
            'LOGGER ERROR: Can not initialise {} logger, '
            'logging to the stderr. Error traceback:\n'.format(logger_name), exc_info=TRACEBACK_LOGGER_ERRORS)

    return logger
