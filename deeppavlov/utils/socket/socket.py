import asyncio
import json
import socket
from logging import getLogger
from pathlib import Path
from typing import Dict, Optional, Tuple

from deeppavlov.core.agent.dialog_logger import DialogLogger
from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path

SOCKET_CONFIG_FILENAME = 'socket_config.json'


class SocketServer:
    _host: str
    _loop: asyncio.AbstractEventLoop
    _model: Chainer
    _params: Dict
    _port: int
    _socket: socket.socket

    def __init__(self, model_config: Path, port: Optional[int] = None):
        self._params = get_socket_params()
        self._dialog_logger = DialogLogger(agent_name='dp_api')
        self._host = self._params['host']
        self._log = getLogger(__name__)
        self._loop = asyncio.get_event_loop()
        self._model = build_model(model_config)
        self._port = port or self._params['port']
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.setblocking(False)

    def start(self) -> None:
        self._socket.bind((self._host, self._port))
        self._socket.listen()
        self._log.info(f'launching socket at http://{self._host}:{self._port}')
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
            await self._wrap_error(conn, 'request type is not json')
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
            await self._wrap_error(conn, 'got several different batch sizes')
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
        await self._loop.sock_sendall(conn, await self._response(error, ''))

    @staticmethod
    async def _response(status: str, payload) -> bytes:
        resp_dict = {'status': status, 'payload': payload}
        resp_str = json.dumps(resp_dict)
        return resp_str.encode('utf-8')


def get_socket_params() -> Dict:
    socket_config_path = get_settings_path() / SOCKET_CONFIG_FILENAME
    socket_params = read_json(socket_config_path)
    return socket_params


def start_socket_server(model_config: Path, port: Optional[int] = None) -> None:
    server = SocketServer(model_config, port)
    server.start()
