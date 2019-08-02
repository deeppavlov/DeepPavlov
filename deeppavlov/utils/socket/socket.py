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


async def handle_connection(conn: socket.socket, addr, model: Chainer, params_names: List[str]):
    conn.setblocking(False)
    incoming_data = b''
    recv_data = conn.recv(1024)
    incoming_data += recv_data
    try:
        data = json.loads(incoming_data)
    except ValueError:
        log.error('request type is not json')
        conn.sendall(b'[]')
        return
    dialog_logger.log_in(data)
    model_args = []
    for param_name in params_names:
        param_value = data.get(param_name)
        if param_value is None or (isinstance(param_value, list) and len(param_value) > 0):
            model_args.append(param_value)
        else:
            log.error(f"nonempty array expected but got '{param_name}'={repr(param_value)}")
            conn.sendall(b'[]')
            return
    lengths = {len(i) for i in model_args if i is not None}

    if not lengths:
        log.error('got empty request')
        conn.sendall(b'[]')
        return
    elif len(lengths) > 1:
        log.error('got several different batch sizes')
        conn.sendall(b'[]')
        return
    batch_size = list(lengths)[0]
    model_args = [arg or [None] * batch_size for arg in model_args]

    # in case when some parameters were not described in model_args
    model_args += [[None] * batch_size for _ in range(len(model.in_x) - len(model_args))]

    prediction = await loop.run_in_executor(None, model, *model_args)
    if len(model.out_params) == 1:
        prediction = [prediction]
    prediction = list(zip(*prediction))
    result = json.dumps(prediction)
    dialog_logger.log_out(result)
    conn.sendall(result.encode('utf-8'))


async def process_connections(server: socket.socket, model: Chainer, params_names: List[str]) -> None:
    while True:
        conn, addr = await loop.run_in_executor(None, server.accept)
        loop.create_task(handle_connection(conn, addr, model, params_names))


def start_model_socket(model_config: Path, port: Optional[int] = None) -> None:
    socket_config_path = get_settings_path() / SOCKET_CONFIG_FILENAME
    socket_params = read_json(socket_config_path)
    model_args_names = socket_params['model_args_names']

    host = socket_params['host']
    port = port or socket_params['port']

    model = build_model(model_config)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen()
    loop.run_until_complete(process_connections(server, model, model_args_names))