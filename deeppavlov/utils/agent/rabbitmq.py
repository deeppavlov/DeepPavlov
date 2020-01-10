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
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from traceback import format_exc
from typing import Any, Dict, List, Optional, Union

import aio_pika
from aio_pika import Connection, Channel, Exchange, Queue, IncomingMessage, Message

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.data.utils import jsonify_data
from deeppavlov.utils.agent.messages import ServiceTaskMessage, ServiceResponseMessage, ServiceErrorMessage
from deeppavlov.utils.agent.messages import get_service_task_message
from deeppavlov.utils.connector import DialogLogger
from deeppavlov.utils.server import get_server_params

dialog_logger = DialogLogger(logger_name='agent_rabbit')
log = logging.getLogger(__name__)

AGENT_IN_EXCHANGE_NAME_TEMPLATE = '{agent_namespace}_e_in'
AGENT_OUT_EXCHANGE_NAME_TEMPLATE = '{agent_namespace}_e_out'
AGENT_ROUTING_KEY_TEMPLATE = 'agent.{agent_name}'

SERVICE_QUEUE_NAME_TEMPLATE = '{agent_namespace}_q_service_{service_name}'
SERVICE_ROUTING_KEY_TEMPLATE = 'service.{service_name}'


class RabbitMQServiceGateway:
    """Class object connects to the RabbitMQ broker to process requests from the DeepPavlov Agent."""
    _add_to_buffer_lock: asyncio.Lock
    _infer_lock: asyncio.Lock
    _model: Chainer
    _model_args_names: List[str]
    _incoming_messages_buffer: List[IncomingMessage]
    _batch_size: int
    _utterance_lifetime_sec: int
    _in_queue: Optional[Queue]
    _connection: Connection
    _agent_in_exchange: Exchange
    _agent_out_exchange: Exchange
    _agent_in_channel: Channel
    _agent_out_channel: Channel

    def __init__(self,
                 model_config: Union[str, Path],
                 service_name: str,
                 agent_namespace: str,
                 batch_size: int,
                 utterance_lifetime_sec: int,
                 rabbit_host: str,
                 rabbit_port: int,
                 rabbit_login: str,
                 rabbit_password: str,
                 rabbit_virtualhost: str,
                 loop: asyncio.AbstractEventLoop) -> None:
        self._add_to_buffer_lock = asyncio.Lock()
        self._infer_lock = asyncio.Lock()
        server_params = get_server_params(model_config)
        self._model_args_names = server_params['model_args_names']
        self._model = build_model(model_config)
        self._in_queue = None
        self._utterance_lifetime_sec = utterance_lifetime_sec
        self._batch_size = batch_size
        self._incoming_messages_buffer = []

        loop.run_until_complete(self._connect(loop=loop, host=rabbit_host, port=rabbit_port, login=rabbit_login,
                                              password=rabbit_password, virtualhost=rabbit_virtualhost,
                                              agent_namespace=agent_namespace))
        loop.run_until_complete(self._setup_queues(service_name, agent_namespace))
        loop.run_until_complete(self._in_queue.consume(callback=self._on_message_callback))

        log.info(f'Service in queue started consuming')

    async def _connect(self,
                       loop: asyncio.AbstractEventLoop,
                       host: str,
                       port: int,
                       login: str,
                       password: str,
                       virtualhost: str,
                       agent_namespace: str) -> None:
        """Connects to RabbitMQ message broker and initiates agent in and out channels and exchanges."""
        log.info('Starting RabbitMQ connection...')

        while True:
            try:
                self._connection = await aio_pika.connect_robust(loop=loop,
                                                                 host=host,
                                                                 port=port,
                                                                 login=login,
                                                                 password=password,
                                                                 virtualhost=virtualhost)
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

    async def _setup_queues(self, service_name: str, agent_namespace: str) -> None:
        """Setups input queue to get messages from DeepPavlov Agent."""
        in_queue_name = SERVICE_QUEUE_NAME_TEMPLATE.format(agent_namespace=agent_namespace,
                                                           service_name=service_name)

        self._in_queue = await self._agent_out_channel.declare_queue(name=in_queue_name, durable=True)
        log.info(f'Declared service in queue: {in_queue_name}')

        service_routing_key = SERVICE_ROUTING_KEY_TEMPLATE.format(service_name=service_name)
        await self._in_queue.bind(exchange=self._agent_out_exchange, routing_key=service_routing_key)
        log.info(f'Queue: {in_queue_name} bound to routing key: {service_routing_key}')

        await self._agent_out_channel.set_qos(prefetch_count=self._batch_size * 2)

    async def _on_message_callback(self, message: IncomingMessage) -> None:
        """Processes messages from the input queue.

        Collects incoming messages to buffer, sends tasks batches for further processing. Depending on the success of
        the processing result sends negative or positive acknowledgements to the input messages.

        """
        await self._add_to_buffer_lock.acquire()
        self._incoming_messages_buffer.append(message)
        log.debug('Incoming message received')

        if len(self._incoming_messages_buffer) < self._batch_size:
            self._add_to_buffer_lock.release()

        await self._infer_lock.acquire()
        try:
            messages_batch = self._incoming_messages_buffer
            valid_messages_batch: List[IncomingMessage] = []
            tasks_batch: List[ServiceTaskMessage] = []

            if messages_batch:
                self._incoming_messages_buffer = []

                if self._add_to_buffer_lock.locked():
                    self._add_to_buffer_lock.release()

                for message in messages_batch:
                    try:
                        task = get_service_task_message(json.loads(message.body, encoding='utf-8'))
                        tasks_batch.append(task)
                        valid_messages_batch.append(message)
                    except Exception as e:
                        log.error(f'Failed to get ServiceTaskMessage from the incoming message: {repr(e)}')
                        await message.reject()

            elif self._add_to_buffer_lock.locked():
                self._add_to_buffer_lock.release()

            if tasks_batch:
                try:
                    await self._process_tasks(tasks_batch)
                except Exception as e:
                    task_ids = [task.payload["task_id"] for task in tasks_batch]
                    log.error(f'got exception {repr(e)} while processing tasks {", ".join(task_ids)}')
                    formatted_exception = format_exc()
                    error_replies = [self._send_results(task, formatted_exception) for task in tasks_batch]
                    await asyncio.gather(*error_replies)
                    for message in valid_messages_batch:
                        await message.reject()
                else:
                    for message in valid_messages_batch:
                        await message.ack()
        finally:
            self._infer_lock.release()

    async def _process_tasks(self, tasks_batch: List[ServiceTaskMessage]) -> None:
        """Gets from tasks batch payloads to infer model, processes them and creates tasks to send results."""
        task_uuids_batch, payloads = \
            zip(*[(task.payload['task_id'], task.payload['payload']) for task in tasks_batch])

        log.debug(f'Prepared to infer tasks {", ".join(task_uuids_batch)}')

        responses_batch = await asyncio.wait_for(self._interact(payloads),
                                                 self._utterance_lifetime_sec)

        results_replies = [self._send_results(task, response) for task, response in zip(tasks_batch, responses_batch)]
        await asyncio.gather(*results_replies)

        log.debug(f'Processed tasks {", ".join(task_uuids_batch)}')

    async def _interact(self, payloads: List[Dict]) -> List[Any]:
        """Infers model with the batch."""
        batch = defaultdict(list)

        for payload in payloads:
            for arg_name in self._model_args_names:
                batch[arg_name].extend(payload.get(arg_name, [None]))

        dialog_logger.log_in(batch)

        prediction = self._model(*batch.values())
        if len(self._model.out_params) == 1:
            prediction = [prediction]
        prediction = list(zip(*prediction))
        result = jsonify_data(prediction)

        dialog_logger.log_out(result)

        return result

    async def _send_results(self, task: ServiceTaskMessage, response: Union[Dict, str]) -> None:
        """Sends responses batch to the DeepPavlov Agent using agent input exchange.

        Args:
            task: Task message from DeepPavlov Agent.
            response: DeepPavlov model response (dict type) if infer was successful otherwise string representation of
                raised error

        """
        if isinstance(response, dict):
            result = ServiceResponseMessage(agent_name=task.agent_name,
                                            task_id=task.payload["task_id"],
                                            response=response)
        else:
            result = ServiceErrorMessage(agent_name=task.agent_name,
                                         task_id=task.payload["task_id"],
                                         formatted_exc=response)

        message = Message(body=json.dumps(result.to_json()).encode('utf-8'),
                          delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                          expiration=self._utterance_lifetime_sec)

        routing_key = AGENT_ROUTING_KEY_TEMPLATE.format(agent_name=task.agent_name)
        await self._agent_in_exchange.publish(message=message, routing_key=routing_key)
        log.debug(f'Sent response for task {str(task.payload["task_id"])} with routing key {routing_key}')
