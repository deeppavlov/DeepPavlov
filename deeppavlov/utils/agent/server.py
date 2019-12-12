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

import asyncio
import logging
from pathlib import Path
from typing import Optional, Union

from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.utils.agent.rabbitmq import RabbitMQServiceGateway

CONNECTOR_CONFIG_FILENAME = 'server_config.json'


def start_rabbit_service(model_config: Union[str, Path],
                         service_name: Optional[str] = None,
                         agent_namespace: Optional[str] = None,
                         batch_size: Optional[int] = None,
                         utterance_lifetime_sec: Optional[int] = None,
                         rabbit_host: Optional[str] = None,
                         rabbit_port: Optional[int] = None,
                         rabbit_login: Optional[str] = None,
                         rabbit_password: Optional[str] = None,
                         rabbit_virtualhost: Optional[str] = None) -> None:
    """Launches DeepPavlov model receiving utterances and sending responses via RabbitMQ message broker.

    Args:
        model_config: Path to DeepPavlov model to be launched.
        service_name: Service name set in DeepPavlov Agent config. Used to format RabbitMQ exchanges, queues and routing
            keys names.
        agent_namespace: Service processes messages only from agents with the same namespace value.
        batch_size: Limits the maximum number of utterances to be processed by service at one inference.
        utterance_lifetime_sec: RabbitMQ message expiration time in seconds.
        rabbit_host: RabbitMQ server host name.
        rabbit_port: RabbitMQ server port number.
        rabbit_login: RabbitMQ server administrator username.
        rabbit_password: RabbitMQ server administrator password.
        rabbit_virtualhost: RabbitMQ server virtualhost name.

    """
    service_config_path = get_settings_path() / CONNECTOR_CONFIG_FILENAME
    service_config: dict = read_json(service_config_path)['agent-rabbit']

    service_name = service_name or service_config['service_name']
    agent_namespace = agent_namespace or service_config['agent_namespace']
    batch_size = batch_size or service_config['batch_size']
    utterance_lifetime_sec = utterance_lifetime_sec or service_config['utterance_lifetime_sec']
    rabbit_host = rabbit_host or service_config['rabbit_host']
    rabbit_port = rabbit_port or service_config['rabbit_port']
    rabbit_login = rabbit_login or service_config['rabbit_login']
    rabbit_password = rabbit_password or service_config['rabbit_password']
    rabbit_virtualhost = rabbit_virtualhost or service_config['rabbit_virtualhost']

    loop = asyncio.get_event_loop()

    gateway = RabbitMQServiceGateway(
        model_config=model_config,
        service_name=service_name,
        agent_namespace=agent_namespace,
        batch_size=batch_size,
        utterance_lifetime_sec=utterance_lifetime_sec,
        rabbit_host=rabbit_host,
        rabbit_port=rabbit_port,
        rabbit_login=rabbit_login,
        rabbit_password=rabbit_password,
        rabbit_virtualhost=rabbit_virtualhost,
        loop=loop
    )

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        gateway.disconnect()
        loop.stop()
        loop.close()
        logging.shutdown()
