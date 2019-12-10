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

import asyncio
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Any, Dict, TypeVar, Union

import aio_pika
from aio_pika import Connection, Channel, Exchange, Queue, IncomingMessage, Message

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.core.data.utils import jsonify_data
from deeppavlov.utils.connector import DialogLogger
from deeppavlov.utils.server import get_server_params

dialog_logger = DialogLogger(logger_name='agent_rabbit')
log = logging.getLogger(__name__)

CONNECTOR_CONFIG_FILENAME = 'server_config.json'

AGENT_IN_EXCHANGE_NAME_TEMPLATE = '{agent_namespace}_e_in'
AGENT_OUT_EXCHANGE_NAME_TEMPLATE = '{agent_namespace}_e_out'
AGENT_ROUTING_KEY_TEMPLATE = 'agent.{agent_name}'

SERVICE_QUEUE_NAME_TEMPLATE = '{agent_namespace}_q_service_{service_name}'
SERVICE_ROUTING_KEY_TEMPLATE = 'service.{service_name}.any'


class MessageBase:

    def __init__(self, msg_type: str, agent_name: str):
        self.msg_type = msg_type
        self.agent_name = agent_name

    @classmethod
    def from_json(cls, message_json):
        return cls(**message_json)

    def to_json(self) -> dict:
        return self.__dict__


TMessageBase = TypeVar('TMessageBase', bound=MessageBase)


class ServiceTaskMessage(MessageBase):
    agent_name: str
    payload: Dict

    def __init__(self, agent_name: str, payload: Dict) -> None:
        super().__init__('service_task', agent_name)
        self.payload = payload


class ServiceResponseMessage(MessageBase):
    agent_name: str
    response: Any
    task_id: str

    def __init__(self, task_id: str, agent_name: str, response: Any) -> None:
        super().__init__('service_response', agent_name)
        self.task_id = task_id
        self.response = response


_message_wrappers_map = {
    'service_task': ServiceTaskMessage,
    'service_response': ServiceResponseMessage
}


def get_transport_message(message_json: dict) -> TMessageBase:
    message_type = message_json.pop('msg_type')

    if message_type not in _message_wrappers_map:
        raise ValueError(f'Unknown transport message type: {message_type}')

    message_wrapper_class: TMessageBase = _message_wrappers_map[message_type]

    return message_wrapper_class.from_json(message_json)


class RabbitMQServiceGateway:
    _model: Chainer
    _service_name: str
    _batch_size: int
    _incoming_messages_buffer: List[IncomingMessage]
    _add_to_buffer_lock: asyncio.Lock
    _infer_lock: asyncio.Lock
    _loop: asyncio.AbstractEventLoop
    _agent_in_exchange: Exchange
    _agent_out_exchange: Exchange
    _connection: Connection
    _agent_in_channel: Channel
    _agent_out_channel: Channel
    _in_queue: Optional[Queue]
    _utterance_lifetime_sec: int

    def __init__(self,
                 model_config: Union[str, Path],
                 service_name: str,
                 agent_name: str,
                 agent_namespace: str,
                 batch_size: int,
                 utterance_lifetime_sec: int,
                 rabbit_host: str,
                 rabbit_port: int,
                 rabbit_login: str,
                 rabbit_password: str,
                 rabbit_virtualhost: str) -> None:
        server_params = get_server_params(model_config)

        self._model_args_names = server_params['model_args_names']

        self._model = build_model(model_config)
        self._in_queue = None
        self._utterance_lifetime_sec = utterance_lifetime_sec
        self._loop = asyncio.get_event_loop()
        self._service_name = service_name
        self._batch_size = batch_size
        self._agent_namespace = agent_namespace
        self._agent_name = agent_name
        self._incoming_messages_buffer = []
        self._add_to_buffer_lock = asyncio.Lock()
        self._infer_lock = asyncio.Lock()
        self._rabbit_host = rabbit_host
        self._rabbit_port = rabbit_port
        self._rabbit_login = rabbit_login
        self._rabbit_password = rabbit_password
        self._rabbit_virtualhost = rabbit_virtualhost

        self._loop.run_until_complete(self._connect())
        self._loop.run_until_complete(self._setup_queues())
        self._loop.run_until_complete(self._in_queue.consume(callback=self._on_message_callback))
        log.info(f'Service in queue started consuming')

    async def send_to_service(self, payloads: List[Dict]) -> List[Any]:
        batch = defaultdict(list)
        for payload in payloads:
            for key, value in payload.items():
                batch[key].extend(value)
        responses_batch = self.interact(batch)

        return responses_batch

    async def _connect(self) -> None:

        log.info('Starting RabbitMQ connection...')

        while True:
            try:
                self._connection = await aio_pika.connect_robust(loop=self._loop,
                                                                 host=self._rabbit_host,
                                                                 port=self._rabbit_port,
                                                                 login=self._rabbit_login,
                                                                 password=self._rabbit_password,
                                                                 virtualhost=self._rabbit_virtualhost)
                log.info('RabbitMQ connected')
                break
            except ConnectionError:
                reconnect_timeout = 5
                log.error(f'RabbitMQ connection error, making another attempt in {reconnect_timeout} secs')
                time.sleep(reconnect_timeout)

        self._agent_in_channel = await self._connection.channel()
        agent_in_exchange_name = AGENT_IN_EXCHANGE_NAME_TEMPLATE.format(agent_namespace=self._agent_namespace)
        self._agent_in_exchange = await self._agent_in_channel.declare_exchange(name=agent_in_exchange_name,
                                                                                type=aio_pika.ExchangeType.TOPIC)
        log.info(f'Declared agent in exchange: {agent_in_exchange_name}')

        self._agent_out_channel = await self._connection.channel()
        agent_out_exchange_name = AGENT_OUT_EXCHANGE_NAME_TEMPLATE.format(agent_namespace=self._agent_namespace)
        self._agent_out_exchange = await self._agent_in_channel.declare_exchange(name=agent_out_exchange_name,
                                                                                 type=aio_pika.ExchangeType.TOPIC)
        log.info(f'Declared agent out exchange: {agent_out_exchange_name}')

    def disconnect(self):
        self._connection.close()

    async def _setup_queues(self) -> None:
        in_queue_name = SERVICE_QUEUE_NAME_TEMPLATE.format(agent_namespace=self._agent_namespace,
                                                           service_name=self._service_name)

        self._in_queue = await self._agent_out_channel.declare_queue(name=in_queue_name, durable=True)
        log.info(f'Declared service in queue: {in_queue_name}')

        service_routing_key = SERVICE_ROUTING_KEY_TEMPLATE.format(service_name=self._service_name)
        await self._in_queue.bind(exchange=self._agent_out_exchange, routing_key=service_routing_key)
        log.info(f'Queue: {in_queue_name} bound to routing key: {service_routing_key}')

        await self._agent_out_channel.set_qos(prefetch_count=self._batch_size * 2)

    async def _on_message_callback(self, message: IncomingMessage) -> None:
        await self._add_to_buffer_lock.acquire()
        self._incoming_messages_buffer.append(message)
        log.debug('Incoming message received')

        if len(self._incoming_messages_buffer) < self._batch_size:
            self._add_to_buffer_lock.release()

        await self._infer_lock.acquire()
        try:
            messages_batch = self._incoming_messages_buffer

            if messages_batch:
                self._incoming_messages_buffer = []

                if self._add_to_buffer_lock.locked():
                    self._add_to_buffer_lock.release()
                tasks_batch: List[ServiceTaskMessage] = [get_transport_message(json.loads(message.body,
                                                                                          encoding='utf-8'))
                                                         for message in messages_batch]

                # TODO: Think about proper infer errors and aknowledge handling
                processed_ok = await self._process_tasks(tasks_batch)

                if processed_ok:
                    for message in messages_batch:
                        await message.ack()
                else:
                    for message in messages_batch:
                        await message.reject()

            elif self._add_to_buffer_lock.locked():
                self._add_to_buffer_lock.release()
        finally:
            self._infer_lock.release()

    async def _process_tasks(self, tasks_batch: List[ServiceTaskMessage]) -> bool:
        task_uuids_batch, payloads = \
            zip(*[(task.payload['task_id'], task.payload['payload']) for task in tasks_batch])

        log.debug(f'Prepared for infering tasks {str(task_uuids_batch)}')

        try:
            responses_batch = await asyncio.wait_for(self.send_to_service(payloads),
                                                     self._utterance_lifetime_sec)

            results_replies = []

            for i, response in enumerate(responses_batch):
                results_replies.append(
                    self._send_results(tasks_batch[i], response)
                )

            await asyncio.gather(*results_replies)
            log.debug(f'Processed tasks {str(task_uuids_batch)}')
            return True
        except asyncio.TimeoutError:
            return False

    async def _send_results(self, task: ServiceTaskMessage, response: Dict) -> None:
        result = ServiceResponseMessage(agent_name=task.agent_name,
                                        task_id=task.payload["task_id"],
                                        response=response)

        message = Message(body=json.dumps(result.to_json()).encode('utf-8'),
                          delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                          expiration=self._utterance_lifetime_sec)

        routing_key = AGENT_ROUTING_KEY_TEMPLATE.format(agent_name=task.agent_name)
        await self._agent_in_exchange.publish(message=message, routing_key=routing_key)
        log.debug(f'Sent response for task {str(task.payload["task_id"])} with routing key {routing_key}')

    def interact(self, payload: Dict[str, Optional[List]]) -> List:
        model_args = payload.values()
        for key in payload.keys():
            if key not in self._model_args_names:
                raise ValueError(f'There is no key {key} in model args names of request')
        dialog_logger.log_in(payload)
        error_msg = None
        lengths = {len(model_arg) for model_arg in model_args if model_arg is not None}

        if not lengths:
            error_msg = 'got empty request'
        elif 0 in lengths:
            error_msg = 'got empty array as model argument'
        elif len(lengths) > 1:
            error_msg = 'got several different batch sizes'

        if error_msg is not None:
            log.error(error_msg)
            raise ValueError(error_msg)

        batch_size = next(iter(lengths))
        model_args = [arg or [None] * batch_size for arg in model_args]

        prediction = self._model(*model_args)
        if len(self._model.out_params) == 1:
            prediction = [prediction]
        prediction = list(zip(*prediction))
        result = jsonify_data(prediction)
        dialog_logger.log_out(result)
        return result


def start_rabbit_service(model_config: Path,
                         service_name: Optional[str] = None,
                         agent_name: Optional[str] = None,
                         agent_namespace: Optional[str] = None,
                         batch_size: Optional[int] = None,
                         utterance_lifetime_sec: Optional[int] = None,
                         rabbit_host: Optional[str] = None,
                         rabbit_port: Optional[int] = None,
                         rabbit_login: Optional[str] = None,
                         rabbit_password: Optional[str] = None,
                         rabbit_virtualhost: Optional[str] = None) -> None:
    service_config_path = get_settings_path() / CONNECTOR_CONFIG_FILENAME
    service_config: dict = read_json(service_config_path)['agent-rabbit']

    service_name = service_name or service_config['service_name']
    agent_name = agent_name or service_config['agent_name']
    agent_namespace = agent_namespace or service_config['agent_namespace']
    batch_size = batch_size or service_config['batch_size']
    utterance_lifetime_sec = utterance_lifetime_sec or service_config['utterance_lifetime_sec']
    rabbit_host = rabbit_host or service_config['rabbit_host']
    rabbit_port = rabbit_port or service_config['rabbit_port']
    rabbit_login = rabbit_login or service_config['rabbit_login']
    rabbit_password = rabbit_password or service_config['rabbit_password']
    rabbit_virtualhost = rabbit_virtualhost or service_config['rabbit_virtualhost']

    gateway = RabbitMQServiceGateway(
        model_config=model_config,
        service_name=service_name,
        agent_name=agent_name,
        agent_namespace=agent_namespace,
        batch_size=batch_size,
        utterance_lifetime_sec=utterance_lifetime_sec,
        rabbit_host=rabbit_host,
        rabbit_port=rabbit_port,
        rabbit_login=rabbit_login,
        rabbit_password=rabbit_password,
        rabbit_virtualhost=rabbit_virtualhost
    )

    loop = asyncio.get_event_loop()

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise e
    finally:
        gateway.disconnect()
        loop.stop()
        loop.close()
        logging.shutdown()
