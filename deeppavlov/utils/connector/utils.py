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
from pathlib import Path

from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.utils.server.server import SERVER_CONFIG_FILENAME, get_server_params

log = getLogger(__name__)


def get_connector_params(connector_name: str, model_config: Path) -> dict:
    server_config_path = Path(get_settings_path(), SERVER_CONFIG_FILENAME).resolve()
    connector_defaults = read_json(server_config_path)['connector_defaults']

    if connector_name not in connector_defaults:
        e = ValueError(f'There is no {connector_name} key at {server_config_path}')
        log.error(e)
        raise e

    config = get_server_params(server_config_path, model_config)
    config['conversation_lifetime'] = connector_defaults['conversation_lifetime']
    config['next_utter_msg'] = connector_defaults['next_utter_msg']
    config.update(connector_defaults[connector_name])

    return config
