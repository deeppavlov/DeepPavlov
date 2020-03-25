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
from logging import getLogger
from pathlib import Path
from struct import pack, unpack
from typing import Any, List, Optional, Tuple, Union

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.data.utils import jsonify_data
from deeppavlov.utils.connector import DialogLogger
from deeppavlov.utils.server import get_server_params

HEADER_FORMAT = '<I'

log = getLogger(__name__)
dialog_logger = DialogLogger(logger_name='socket_api')


def encode(data: Any) -> bytes:
    """Сonverts data to the socket server input formatted bytes array.

    Serializes ``data`` to the JSON formatted bytes array and adds 4 bytes to the beginning of the array - packed
    to bytes length of the JSON formatted bytes array. Header format is "<I"
    (see https://docs.python.org/3/library/struct.html#struct-format-strings)

    Args:
        data: Object to pact to the bytes array.

    Raises:
        TypeError: If data is not JSON-serializable object.

    Examples:
        >>> from deeppavlov.utils.socket import encode
        >>> encode({'a':1})
        b'\x08\x00\x00\x00{"a": 1}
        >>> encode([42])
        b'\x04\x00\x00\x00[42]'

    """
    json_data = jsonify_data(data)
    bytes_data = json.dumps(json_data).encode()
    response = pack(HEADER_FORMAT, len(bytes_data)) + bytes_data
    return response


class SocketServer:
    """Creates socket server that sends the received data to the DeepPavlov model and returns model response.

    The server receives bytes array consists of the `header` and the `body`. The `header` is the first 4 bytes
    of the array - `body` length in bytes represented by a packed unsigned int (byte order is little-endian).
    `body` is dictionary serialized to JSON formatted bytes array that server sends to the model. The dictionary
    keys should match model arguments names, the values should be lists or tuples of inferenced values.

    Socket server request creation example:
        >>> from deeppavlov.utils.socket import encode
        >>> request = encode({"context":["Elon Musk launched his cherry Tesla roadster to the Mars orbit"]})
        >>> request
        b'I\x00\x00\x00{"x": ["Elon Musk launched his cherry Tesla roadster to the Mars orbit"]}'

    Socket server response, like the request, consists of the header and the body. Response body is dictionary
    {'status': status, 'payload': payload} serialized to a JSON formatted byte array, where:
        status (str): 'OK' if the model successfully processed the data, else - error message.
        payload: (Optional[List[Tuple]]): The model result if no error has occurred, otherwise None.

    """
    _launch_msg: str
    _loop: asyncio.AbstractEventLoop
    _model: Chainer
    _model_args_names: List

    def __init__(self,
                 model_config: Path,
                 socket_type: str,
                 port: Optional[int] = None,
                 socket_file: Optional[Union[str, Path]] = None) -> None:
        """Initializes socket server.

        Args:
            model_config: Path to the config file.
            socket_type: Socket family. "TCP" for the AF_INET socket server, "UNIX" for UNIX Domain Socket server.
            port: Port number for the AF_INET address family. If parameter is not defined, the port number from the
                utils/settings/server_config.json is used.
            socket_file: Path to the file to which UNIX Domain Socket server connects. If parameter is not defined,
                the path from the utils/settings/server_config.json is used.

        Raises:
            ValueError: If ``socket_type`` parameter is neither "TCP" nor "UNIX".

        """
        server_params = get_server_params(model_config)
        socket_type = socket_type or server_params['socket_type']
        self._loop = asyncio.get_event_loop()

        if socket_type == 'TCP':
            host = server_params['host']
            port = port or server_params['port']
            self._launch_msg = f'{server_params["socket_launch_message"]} http://{host}:{port}'
            self._loop.create_task(asyncio.start_server(self._handle_client, host, port))
        elif socket_type == 'UNIX':
            socket_file = socket_file or server_params['unix_socket_file']
            socket_path = Path(socket_file).resolve()
            if socket_path.exists():
                socket_path.unlink()
            self._launch_msg = f'{server_params["socket_launch_message"]} {socket_file}'
            self._loop.create_task(asyncio.start_unix_server(self._handle_client, socket_file))
        else:
            raise ValueError(f'socket type "{socket_type}" is not supported')

        self._model = build_model(model_config)
        self._model_args_names = server_params['model_args_names']

    def start(self) -> None:
        """Launches socket server"""
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
        """Handles connection from a client.

        Validates requests, sends request body to DeepPavlov model, sends responses to client.

        """
        addr = writer.get_extra_info('peername')
        log.info(f'handling connection from {addr}')
        while True:
            header = await reader.read(4)
            if not header:
                log.info(f'closing connection from {addr}')
                writer.close()
                break
            elif len(header) != 4:
                error_msg = f'header "{header}" length less than 4 bytes'
                log.error(error_msg)
                response = self._response(error_msg)
            else:
                data_len = unpack(HEADER_FORMAT, header)[0]
                request_body = await reader.read(data_len)
                try:
                    data = json.loads(request_body)
                    response = await self._interact(data)
                except ValueError:
                    error_msg = f'request "{request_body}" type is not json'
                    log.error(error_msg)
                    response = self._response(error_msg)
            writer.write(response)
            await writer.drain()

    async def _interact(self, data: dict) -> bytes:
        dialog_logger.log_in(data)
        model_args = []
        for param_name in self._model_args_names:
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
        """Puts arguments into dict and serialize it to JSON formatted byte array with header.

        Args:
            status: Response status. 'OK' if no error has occurred, otherwise error message.
            payload: DeepPavlov model result if no error has occurred, otherwise None.

        Returns:
            dict({'status': status, 'payload': payload}) serialized to a JSON formatted byte array starting with the
                4-byte header - the length of serialized dict in bytes.

        """
        return encode({'status': status, 'payload': payload})


def start_socket_server(model_config: Path, socket_type: str, port: Optional[int],
                        socket_file: Optional[Union[str, Path]]) -> None:
    server = SocketServer(model_config, socket_type, port, socket_file)
    server.start()
