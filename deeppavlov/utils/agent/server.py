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
from typing import List, Optional, Callable
from uuid import uuid4

import aio_pika
from aio_pika import Connection, Channel, Exchange, Queue, IncomingMessage, Message

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.data.utils import jsonify_data
from deeppavlov.utils.connector import DialogLogger
from deeppavlov.utils.server import get_server_params

dialog_logger = DialogLogger(logger_name='rest_api')
log = logging.getLogger(__name__)

AGENT_IN_EXCHANGE_NAME_TEMPLATE = '{agent_namespace}_e_in'
AGENT_OUT_EXCHANGE_NAME_TEMPLATE = '{agent_namespace}_e_out'
AGENT_QUEUE_NAME_TEMPLATE = '{agent_namespace}_q_agent_{agent_name}'
AGENT_ROUTING_KEY_TEMPLATE = 'agent.{agent_name}'

SERVICE_QUEUE_NAME_TEMPLATE = '{agent_namespace}_q_service_{service_name}'
SERVICE_ROUTING_KEY_TEMPLATE = 'service.{service_name}.any'
SERVICE_INSTANCE_ROUTING_KEY_TEMPLATE = 'service.{service_name}.instance.{instance_id}'

CHANNEL_QUEUE_NAME_TEMPLATE = '{agent_namespace}_{agent_name}_q_channel_{channel_id}'
CHANNEL_ROUTING_KEY_TEMPLATE = 'agent.{agent_name}.channel.{channel_id}.any'


from typing import TypeVar, Any, Dict


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
    service_instance_id: str
    response: Any
    task_id: str

    def __init__(self, task_id: str, agent_name: str, service_instance_id: str, response: Any) -> None:
        super().__init__('service_response', agent_name)
        self.task_id = task_id
        self.service_instance_id = service_instance_id
        self.response = response


class ToChannelMessage(MessageBase):
    agent_name: str
    channel_id: str
    user_id: str
    response: str

    def __init__(self, agent_name: str, channel_id: str, user_id: str, response: str) -> None:
        super().__init__('to_channel_message', agent_name)
        self.channel_id = channel_id
        self.user_id = user_id
        self.response = response


class FromChannelMessage(MessageBase):
    agent_name: str
    channel_id: str
    user_id: str
    utterance: str
    reset_dialog: bool

    def __init__(self, agent_name: str, channel_id: str, user_id: str, utterance: str, reset_dialog: bool) -> None:
        super().__init__('from_channel_message', agent_name)
        self.channel_id = channel_id
        self.user_id = user_id
        self.utterance = utterance
        self.reset_dialog = reset_dialog


_message_wrappers_map = {
    'service_task': ServiceTaskMessage,
    'service_response': ServiceResponseMessage,
    'to_channel_message': ToChannelMessage,
    'from_channel_message': FromChannelMessage
}


def get_transport_message(message_json: dict) -> TMessageBase:
    message_type = message_json.pop('msg_type')

    if message_type not in _message_wrappers_map:
        raise ValueError(f'Unknown transport message type: {message_type}')

    message_wrapper_class: TMessageBase = _message_wrappers_map[message_type]

    return message_wrapper_class.from_json(message_json)


class ServiceGatewayBase:
    _to_service_callback: Callable

    def __init__(self, to_service_callback: Callable, *args, **kwargs) -> None:
        super(ServiceGatewayBase, self).__init__(*args, **kwargs)
        self._to_service_callback = to_service_callback


class RabbitMQTransportBase:
    _config: dict
    _loop: asyncio.AbstractEventLoop
    _agent_in_exchange: Exchange
    _agent_out_exchange: Exchange
    _connection: Connection
    _agent_in_channel: Channel
    _agent_out_channel: Channel
    _in_queue: Optional[Queue]
    _utterance_lifetime_sec: int

    def __init__(self, config: dict, *args, **kwargs):
        super(RabbitMQTransportBase, self).__init__(*args, **kwargs)
        self._config = config
        self._in_queue = None
        self._utterance_lifetime_sec = config['utterance_lifetime_sec']

    async def _connect(self) -> None:
        agent_namespace = self._config['agent_namespace']

        host = self._config['transport']['AMQP']['host']
        port = self._config['transport']['AMQP']['port']
        login = self._config['transport']['AMQP']['login']
        password = self._config['transport']['AMQP']['password']
        virtualhost = self._config['transport']['AMQP']['virtualhost']

        log.info('Starting RabbitMQ connection...')

        while True:
            try:
                self._connection = await aio_pika.connect_robust(loop=self._loop, host=host, port=port, login=login,
                                                                 password=password, virtualhost=virtualhost)

                log.info('RabbitMQ connected')
                break
            except ConnectionError:
                reconnect_timeout = 5
                log.error(f'RabbitMQ connection error, making another attempt in {reconnect_timeout} secs')
                time.sleep(reconnect_timeout)

        self._agent_in_channel = await self._connection.channel()
        agent_in_exchange_name = AGENT_IN_EXCHANGE_NAME_TEMPLATE.format(agent_namespace=agent_namespace)
        self._agent_in_exchange = await self._agent_in_channel.declare_exchange(name=agent_in_exchange_name,
                                                                                type=aio_pika.ExchangeType.TOPIC)
        log.info(f'Declared agent in exchange: {agent_in_exchange_name}')

        self._agent_out_channel = await self._connection.channel()
        agent_out_exchange_name = AGENT_OUT_EXCHANGE_NAME_TEMPLATE.format(agent_namespace=agent_namespace)
        self._agent_out_exchange = await self._agent_in_channel.declare_exchange(name=agent_out_exchange_name,
                                                                                 type=aio_pika.ExchangeType.TOPIC)
        log.info(f'Declared agent out exchange: {agent_out_exchange_name}')

    def disconnect(self):
        self._connection.close()

    async def _setup_queues(self) -> None:
        raise NotImplementedError

    async def _on_message_callback(self, message: IncomingMessage) -> None:
        raise NotImplementedError


class RabbitMQServiceGateway(RabbitMQTransportBase, ServiceGatewayBase):
    _service_name: str
    _instance_id: str
    _batch_size: int
    _incoming_messages_buffer: List[IncomingMessage]
    _add_to_buffer_lock: asyncio.Lock
    _infer_lock: asyncio.Lock

    def __init__(self, config: dict, to_service_callback: Callable) -> None:
        super(RabbitMQServiceGateway, self).__init__(config=config, to_service_callback=to_service_callback)
        self._loop = asyncio.get_event_loop()
        self._service_name = self._config['service']['name']
        self._instance_id = self._config['service'].get('instance_id', None) or f'{self._service_name}{str(uuid4())}'
        self._batch_size = self._config['service'].get('batch_size', 1)

        self._incoming_messages_buffer = []
        self._add_to_buffer_lock = asyncio.Lock()
        self._infer_lock = asyncio.Lock()

        self._loop.run_until_complete(self._connect())
        self._loop.run_until_complete(self._setup_queues())
        self._loop.run_until_complete(self._in_queue.consume(callback=self._on_message_callback))
        log.info(f'Service in queue started consuming')

    async def _setup_queues(self) -> None:
        agent_namespace = self._config['agent_namespace']

        in_queue_name = SERVICE_QUEUE_NAME_TEMPLATE.format(agent_namespace=agent_namespace,
                                                           service_name=self._service_name)

        self._in_queue = await self._agent_out_channel.declare_queue(name=in_queue_name, durable=True)
        log.info(f'Declared service in queue: {in_queue_name}')

        # TODO think if we can remove this workaround for bot annotators
        service_names = self._config['service'].get('names', []) or [self._service_name]
        for service_name in service_names:
            any_instance_routing_key = SERVICE_ROUTING_KEY_TEMPLATE.format(service_name=service_name)
            await self._in_queue.bind(exchange=self._agent_out_exchange, routing_key=any_instance_routing_key)
            log.info(f'Queue: {in_queue_name} bound to routing key: {any_instance_routing_key}')

            this_instance_routing_key = SERVICE_INSTANCE_ROUTING_KEY_TEMPLATE.format(service_name=service_name,
                                                                                     instance_id=self._instance_id)

            await self._in_queue.bind(exchange=self._agent_out_exchange, routing_key=this_instance_routing_key)
            log.info(f'Queue: {in_queue_name} bound to routing key: {this_instance_routing_key}')

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
            responses_batch = await asyncio.wait_for(self._to_service_callback(payloads),
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
                                        service_instance_id=self._instance_id,
                                        task_id=task.payload["task_id"],
                                        response=response)

        message = Message(body=json.dumps(result.to_json()).encode('utf-8'),
                          delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                          expiration=self._utterance_lifetime_sec)

        routing_key = AGENT_ROUTING_KEY_TEMPLATE.format(agent_name=task.agent_name)
        await self._agent_in_exchange.publish(message=message, routing_key=routing_key)
        log.debug(f'Sent response for task {str(task.payload["task_id"])} with routing key {routing_key}')



def interact(model: Chainer, payload: Dict[str, Optional[List]], model_args_names) -> List:
    model_args = payload.values()
    for key in payload.keys():
        if key not in model_args_names:
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

    prediction = model(*model_args)
    if len(model.out_params) == 1:
        prediction = [prediction]
    prediction = list(zip(*prediction))
    result = jsonify_data(prediction)
    dialog_logger.log_out(result)
    return result


def start_rabbit_service(model_config: Path) -> None:
    server_params = get_server_params(model_config)

    model_args_names = server_params['model_args_names']

    model = build_model(model_config)

    async def send_to_service(payloads: List[Dict]) -> List[Any]:
        batch = defaultdict(list)
        for payload in payloads:
            for key, value in payload.items():
                batch[key].extend(value)
        responses_batch = interact(model, batch, model_args_names)

        return responses_batch

    gateway_config = {
        'agent_namespace': 'deeppavlov_agent',
        'agent_name': 'dp_agent',
        'utterance_lifetime_sec': 120,
        'channels': {},
        'transport': {
            'type': 'AMQP',
            'AMQP': {
                'host': '127.0.0.1',
                'port': 5672,
                'login': 'guest',
                'password': 'guest',
                'virtualhost': '/'
            }
        },
        'service':{
            'name': 'ner'
        }
    }

    gateway = RabbitMQServiceGateway(config=gateway_config, to_service_callback=send_to_service)

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
