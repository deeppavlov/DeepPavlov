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
from typing import Dict, List, Optional, Tuple, Union

from deeppavlov.core.agent.dialog_logger import DialogLogger
from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.core.data.utils import jsonify_data
from deeppavlov.utils.server.server import get_server_params

SOCKET_CONFIG_FILENAME = 'socket_config.json'


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
    _address_family: socket.AddressFamily
    _bind_address: Union[Tuple[str, int], str]
    _launch_msg: str
    _loop: asyncio.AbstractEventLoop
    _model: Chainer
    _params: Dict
    _socket: socket.socket
    _socket_type: str

    def __init__(self, model_config: Path, socket_type: str, port: Optional[int] = None,
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
        socket_config_path = get_settings_path() / SOCKET_CONFIG_FILENAME
        self._params = get_server_params(socket_config_path, model_config)
        self._socket_type = socket_type or self._params['socket_type']

        if self._socket_type == 'TCP':
            host = self._params['host']
            port = port or self._params['port']
            self._address_family = socket.AF_INET
            self._launch_msg = f'{self._params["binding_message"]} http://{host}:{port}'
            self._bind_address = (host, port)
        elif self._socket_type == 'UNIX':
            self._address_family = socket.AF_UNIX
            bind_address = socket_file or self._params['unix_socket_file']
            bind_address = Path(bind_address).resolve()
            if bind_address.exists():
                bind_address.unlink()
            self._bind_address = str(bind_address)
            self._launch_msg = f'{self._params["binding_message"]} {self._bind_address}'
        else:
            raise ValueError(f'socket type "{self._socket_type}" is not supported')

        self._dialog_logger = DialogLogger(agent_name='dp_api')
        self._log = getLogger(__name__)
        self._loop = asyncio.get_event_loop()
        self._model = build_model(model_config)
        self._socket = socket.socket(self._address_family, socket.SOCK_STREAM)

        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.setblocking(False)

    def start(self) -> None:
        """Binds the socket to the address and enables the server to accept connections"""
        self._socket.bind(self._bind_address)
        self._socket.listen()
        self._log.info(self._launch_msg)
        try:
            self._loop.run_until_complete(self._server())
        except Exception as e:
            self._log.error(f'got exception {e} while running server')
        finally:
            self._loop.close()
            self._socket.close()

    async def _server(self) -> None:
        while True:
            conn, addr = await self._loop.sock_accept(self._socket)
            self._loop.create_task(self._handle_connection(conn, addr))

    async def _handle_connection(self, conn: socket.socket, addr: Tuple) -> None:
        self._log.info(f'handling connection from {addr}')
        conn.setblocking(False)
        recv_data = b''
        try:
            while True:
                chunk = await self._loop.run_in_executor(None, conn.recv, self._params['bufsize'])
                if chunk:
                    recv_data += chunk
                else:
                    break
        except BlockingIOError:
            pass
        try:
            data = json.loads(recv_data)
        except ValueError:
            await self._wrap_error(conn, f'request "{recv_data}" type is not json')
            return
        self._dialog_logger.log_in(data)
        model_args = []
        for param_name in self._params['model_args_names']:
            param_value = data.get(param_name)
            if param_value is None or (isinstance(param_value, list) and len(param_value) > 0):
                model_args.append(param_value)
            else:
                await self._wrap_error(conn, f"nonempty array expected but got '{param_name}'={repr(param_value)}")
                return
        lengths = {len(i) for i in model_args if i is not None}

        if not lengths:
            await self._wrap_error(conn, 'got empty request')
            return
        elif len(lengths) > 1:
            await self._wrap_error(conn, f'got several different batch sizes: {lengths}')
            return
        batch_size = list(lengths)[0]
        model_args = [arg or [None] * batch_size for arg in model_args]

        # in case when some parameters were not described in model_args
        model_args += [[None] * batch_size for _ in range(len(self._model.in_x) - len(model_args))]

        prediction = await self._loop.run_in_executor(None, self._model, *model_args)
        if len(self._model.out_params) == 1:
            prediction = [prediction]
        prediction = list(zip(*prediction))
        result = await self._response('OK', prediction)
        self._dialog_logger.log_out(result)
        await self._loop.sock_sendall(conn, result)

    async def _wrap_error(self, conn: socket.socket, error: str) -> None:
        self._log.error(error)
        await self._loop.sock_sendall(conn, await self._response(error, None))

    @staticmethod
    async def _response(status: str, payload: Optional[List[Tuple]]) -> bytes:
        """Puts arguments into dict and serialize it to JSON formatted byte array.

        Args:
            status: Response status. 'OK' if no error has occurred, otherwise error message.
            payload: DeepPavlov model result if no error has occurred, otherwise None.

        Returns:
            dict({'status': status, 'payload': payload}) serialized to a JSON formatted byte array.

        """
        resp_dict = jsonify_data({'status': status, 'payload': payload})
        resp_str = json.dumps(resp_dict)
        return resp_str.encode('utf-8')


def start_socket_server(model_config: Path, socket_type: str, port: Optional[int],
                        socket_file: Optional[Union[str, Path]]) -> None:
    server = SocketServer(model_config, socket_type, port, socket_file)
    server.start()
