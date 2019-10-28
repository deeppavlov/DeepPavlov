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
import socket
from logging import getLogger
from pathlib import Path
from struct import pack, unpack, error
from typing import Dict, List, Optional, Tuple, Union, Any

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.core.data.utils import jsonify_data
from deeppavlov.utils.connector import DialogLogger
from deeppavlov.utils.server import get_server_params

SOCKET_CONFIG_FILENAME = 'socket_config.json'
HEADER_FORMAT = '<I'

log = getLogger(__name__)
dialog_logger = DialogLogger(logger_name='socket_api')


def encode(data: Any) -> bytes:
    json_data = jsonify_data(data)
    bytes_data = json.dumps(json_data).encode()
    response = pack(HEADER_FORMAT, len(bytes_data)) + bytes_data
    return response


class SocketServer:
    """Creates socket server that sends the received data to the DeepPavlov model and returns model response.

    The server receives dictionary serialized to JSON formatted bytes array and sends it to the model. The dictionary
    keys should match model arguments names, the values should be lists or tuples of inferenced values.

    Example:
        {“context”:[“Elon Musk launched his cherry Tesla roadster to the Mars orbit”]}

    Socket server returns dictionary {'status': status, 'payload': payload} serialized to a JSON formatted byte array,
    where:
        status (str): 'OK' if the model successfully processed the data, else - error message.
        payload: (Optional[List[Tuple]]): The model result if no error has occurred, otherwise None

    """
    _launch_msg: str
    _loop: asyncio.AbstractEventLoop
    _model: Chainer
    _params: Dict
    _socket: socket.socket

    def __init__(self,
                 model_config: Path,
                 socket_type: str,
                 port: Optional[int] = None,
                 socket_file: Optional[Union[str, Path]] = None) -> None:
        """Initialize socket server.

        Args:
            model_config: Path to the config file.
            socket_type: Socket family. "TCP" for the AF_INET socket, "UNIX" for the AF_UNIX.
            port: Port number for the AF_INET address family. If parameter is not defined, the port number from the
                model_config is used.
            socket_file: Path to the file to which server of the AF_UNIX address family connects. If parameter
                is not defined, the path from the model_config is used.

        """
        self._loop = asyncio.get_event_loop()
        socket_config_path = get_settings_path() / SOCKET_CONFIG_FILENAME
        self._params = get_server_params(model_config, socket_config_path)
        socket_type = socket_type or self._params['socket_type']

        if socket_type == 'TCP':
            host = self._params['host']
            port = port or self._params['port']
            self._launch_msg = f'{self._params["binding_message"]} http://{host}:{port}'
            self._loop.create_task(asyncio.start_server(self._handle_client, host, port))
        elif socket_type == 'UNIX':
            socket_file = socket_file or self._params['unix_socket_file']
            socket_path = Path(socket_file).resolve()
            if socket_path.exists():
                socket_path.unlink()
            self._launch_msg = f'{self._params["binding_message"]} {socket_file}'
            self._loop.create_task(asyncio.start_unix_server(self._handle_client, socket_file))
        else:
            raise ValueError(f'socket type "{socket_type}" is not supported')

        self._model = build_model(model_config)

    def start(self) -> None:
        """Binds the socket to the address and enables the server to accept connections"""
        log.info(self._launch_msg)
        try:
            self._loop.run_forever()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            log.error(f'got exception {e} while running server')
        finally:
            self._loop.close()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        addr = writer.get_extra_info('peername')
        log.info(f'handling connection from {addr}')
        while True:
            header = await reader.read(4)
            if not header:
                log.info(f'closing connection from {addr}')
                writer.close()
                break
            elif len(header) != 4:
                error_msg = f'header "{header}" lengths less than 4 bytes'
                log.error(error_msg)
                response = self._response(error_msg)
            else:
                data_len = unpack(HEADER_FORMAT, header)[0]
                request_body = await reader.read(data_len)
                try:
                    data = json.loads(request_body)
                    response = await self.interact(data)
                except ValueError:
                    error_msg = f'request "{request_body}" type is not json'
                    log.error(error_msg)
                    response = self._response(error_msg)
            writer.write(response)
            await writer.drain()

    async def interact(self, data) -> bytes:
        dialog_logger.log_in(data)
        model_args = []
        for param_name in self._params['model_args_names']:
            param_value = data.get(param_name)
            if param_value is None or (isinstance(param_value, list) and len(param_value) > 0):
                model_args.append(param_value)
            else:
                error_msg = f"nonempty array expected but got '{param_name}'={repr(param_value)}"
                log.error(error_msg)
                return self._response(error_msg)
        lengths = {len(i) for i in model_args if i is not None}

        if not lengths:
            error_msg = 'got empty request'
            log.error(error_msg)
            return self._response(error_msg)
        elif len(lengths) > 1:
            error_msg = f'got several different batch sizes: {lengths}'
            log.error(error_msg)
            return self._response(error_msg)

        batch_size = list(lengths)[0]
        model_args = [arg or [None] * batch_size for arg in model_args]

        # in case when some parameters were not described in model_args
        model_args += [[None] * batch_size for _ in range(len(self._model.in_x) - len(model_args))]

        prediction = await self._loop.run_in_executor(None, self._model, *model_args)
        if len(self._model.out_params) == 1:
            prediction = [prediction]
        prediction = list(zip(*prediction))
        dialog_logger.log_out(prediction)
        return self._response(payload=prediction)

    @staticmethod
    def _response(status: str = 'OK', payload: Optional[List[Tuple]] = None) -> bytes:
        """Puts arguments into dict and serialize it to JSON formatted byte array.

        Args:
            status: Response status. 'OK' if no error has occurred, otherwise error message.
            payload: DeepPavlov model result if no error has occurred, otherwise None.

        Returns:
            dict({'status': status, 'payload': payload}) serialized to a JSON formatted byte array.

        """
        return encode({'status': status, 'payload': payload})


def start_socket_server(model_config: Path, socket_type: str, port: Optional[int],
                        socket_file: Optional[Union[str, Path]]) -> None:
    server = SocketServer(model_config, socket_type, port, socket_file)
    server.start()
