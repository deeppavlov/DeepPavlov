import asyncio
import json
import socket
from logging import getLogger
from pathlib import Path
from typing import Optional, List

from deeppavlov.core.agent.dialog_logger import DialogLogger
from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path

SOCKET_CONFIG_FILENAME = 'socket_config.json'

log = getLogger(__name__)
loop = asyncio.get_event_loop()
dialog_logger = DialogLogger(agent_name='dp_api')


async def handle_connection(conn: socket.socket, addr, model: Chainer, params_names: List[str], bufsize: int):
    async def response(status, payload):
        resp_dict = {'status': status, 'payload': payload}
        resp_str = json.dumps(resp_dict)
        return resp_str.encode('utf-8')

    async def wrap_error(conn, error):
        log.error(error)
        await loop.sock_sendall(conn, await response(error, ''))

    log.info(f'handling connection from {addr}')
    conn.setblocking(False)
    recv_data = b''
    try:
        while True:
            chunk = await loop.run_in_executor(None, conn.recv, bufsize)
            if chunk:
                recv_data += chunk
            else:
                break
    except BlockingIOError:
        pass
    try:
        data = json.loads(recv_data)
    except ValueError:
        await wrap_error(conn, 'request type is not json')
        return
    dialog_logger.log_in(data)
    model_args = []
    for param_name in params_names:
        param_value = data.get(param_name)
        if param_value is None or (isinstance(param_value, list) and len(param_value) > 0):
            model_args.append(param_value)
        else:
            await wrap_error(conn, f"nonempty array expected but got '{param_name}'={repr(param_value)}")
            return
    lengths = {len(i) for i in model_args if i is not None}

    if not lengths:
        await wrap_error(conn, 'got empty request')
        return
    elif len(lengths) > 1:
        await wrap_error(conn, 'got several different batch sizes')
        return
    batch_size = list(lengths)[0]
    model_args = [arg or [None] * batch_size for arg in model_args]

    # in case when some parameters were not described in model_args
    model_args += [[None] * batch_size for _ in range(len(model.in_x) - len(model_args))]

    prediction = await loop.run_in_executor(None, model, *model_args)
    if len(model.out_params) == 1:
        prediction = [prediction]
    prediction = list(zip(*prediction))
    result = await response('OK', prediction)
    dialog_logger.log_out(result)
    await loop.sock_sendall(conn, result)


async def server(sock: socket.socket, model: Chainer, params_names: List[str], bufsize: int) -> None:
    while True:
        conn, addr = await loop.sock_accept(sock)
        loop.create_task(handle_connection(conn, addr, model, params_names, bufsize))


def get_socket_params() -> dict:
    socket_config_path = get_settings_path() / SOCKET_CONFIG_FILENAME
    socket_params = read_json(socket_config_path)
    return socket_params


def start_model_socket(model_config: Path, port: Optional[int] = None) -> None:
    socket_params = get_socket_params()

    model_args_names = socket_params['model_args_names']
    host = socket_params['host']
    port = port or socket_params['port']
    bufsize = socket_params['bufsize']

    model = build_model(model_config)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setblocking(False)
    s.bind((host, port))
    s.listen()
    log.info(f'socket http://{host}:{port} has successfully launched')
    try:
        loop.run_until_complete(server(s, model, model_args_names, bufsize))
    except Exception as e:
        log.error(f'got exception {e} while running server')
    finally:
        loop.close()
        s.close()
