import io
import json
import logging
import os
import pickle
import shutil
import signal
import socket
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from struct import unpack
from time import sleep
from typing import Optional, Union
from urllib.parse import urljoin

import pexpect
import pexpect.popen_spawn
import pytest
import requests

import deeppavlov
from deeppavlov import build_model
from deeppavlov.core.commands.utils import parse_config
from deeppavlov.core.data.utils import get_all_elems_from_json
from deeppavlov.download import deep_download
from deeppavlov.utils.server import get_server_params
from deeppavlov.utils.socket import encode

tests_dir = Path(__file__).parent
test_configs_path = tests_dir / "deeppavlov" / "configs"
src_dir = Path(deeppavlov.__path__[0]) / "configs"
test_src_dir = tests_dir / "test_configs"
download_path = tests_dir / "download"

cache_dir: Optional[Path] = None
if not os.getenv('DP_PYTEST_NO_CACHE'):
    cache_dir = tests_dir / 'download_cache'

api_port = os.getenv('DP_PYTEST_API_PORT')
if api_port is not None:
    api_port = int(api_port)

TEST_MODES = ['IP',  # test_inferring_pretrained_model
              'TI',  # test_consecutive_training_and_inferring
              'SR',  # test_serialization
              ]

ALL_MODES = ('IP', 'TI', 'SR')

ONE_ARGUMENT_INFER_CHECK = ('Dummy text', None)
TWO_ARGUMENTS_INFER_CHECK = ('Dummy text', 'Dummy text', None)
FOUR_ARGUMENTS_INFER_CHECK = ('Dummy text', 'Dummy text', 'Dummy text', 'Dummy_text', None)

# Mapping from model name to config-model_dir-ispretrained and corresponding queries-response list.
PARAMS = {
    "relation_extraction": {
        ("relation_extraction/re_docred.json", "relation_extraction", ('IP',)):
            [
                (
                    [["Barack", "Obama", "is", "married", "to", "Michelle", "Obama", ",", "born", "Michelle",
                      "Robinson", "."]],
                    [[[(0, 2)], [(5, 7), (9, 11)]]],
                    [["PER", "PER"]],
                    (
                        ['P26'],
                        ['spouse']
                    )
                )
            ],
        ("relation_extraction/re_rured.json", "relation_extraction", ('IP',)):
            [
                (
                    [["Илон", "Маск", "живет", "в", "Сиэттле", "."]],
                    [[[(0, 2)], [(4, 6)]]],
                    [["PERSON", "CITY"]],
                    (
                        ['P551'],
                        ['место жительства']
                    )
                ),
            ]
    }
}

MARKS = {"gpu_only": ["squad"], "slow": ["error_model", "go_bot", "squad"]}  # marks defined in pytest.ini

TEST_GRID = []
for model in PARAMS.keys():
    for conf_file, model_dir, mode in PARAMS[model].keys():
        marks = []
        for mark in MARKS.keys():
            if model in MARKS[mark]:
                marks.append(eval("pytest.mark." + mark))
        grid_unit = pytest.param(model, conf_file, model_dir, mode, marks=marks)
        TEST_GRID.append(grid_unit)


def _override_with_test_values(item: Union[dict, list]) -> None:
    if isinstance(item, dict):
        keys = [k for k in item.keys() if k.startswith('pytest_')]
        for k in keys:
            item[k[len('pytest_'):]] = item.pop(k)
        item = item.values()

    for child in item:
        if isinstance(child, (dict, list)):
            _override_with_test_values(child)


def download_config(config_path):
    src_file = src_dir / config_path
    if not src_file.is_file():
        src_file = test_src_dir / config_path

    if not src_file.is_file():
        raise RuntimeError('No config file {}'.format(config_path))

    with src_file.open(encoding='utf8') as fin:
        config: dict = json.load(fin)

    # Download referenced config files
    config_references = get_all_elems_from_json(parse_config(config), 'config_path')
    for config_ref in config_references:
        splitted = config_ref.split("/")
        first_subdir_index = splitted.index("configs") + 1
        m_name = config_ref.split('/')[first_subdir_index]
        config_ref = '/'.join(config_ref.split('/')[first_subdir_index:])

        test_configs_path.joinpath(m_name).mkdir(exist_ok=True)
        if not test_configs_path.joinpath(config_ref).exists():
            download_config(config_ref)

    # Update config for testing
    config.setdefault('train', {}).setdefault('pytest_epochs', 1)
    config['train'].setdefault('pytest_max_batches', 2)
    config['train'].setdefault('pytest_max_test_batches', 2)
    _override_with_test_values(config)

    config_path = test_configs_path / config_path
    config_path.parent.mkdir(exist_ok=True, parents=True)
    with config_path.open("w", encoding='utf8') as fout:
        json.dump(config, fout)


def install_config(config_path):
    logfile = io.BytesIO(b'')
    p = pexpect.popen_spawn.PopenSpawn(sys.executable + " -m deeppavlov install " + str(config_path), timeout=None,
                                       logfile=logfile)
    p.readlines()
    if p.wait() != 0:
        raise RuntimeError('Installing process of {} returned non-zero exit code: \n{}'
                           .format(config_path, logfile.getvalue().decode()))


def setup_module():
    shutil.rmtree(str(test_configs_path), ignore_errors=True)
    shutil.rmtree(str(download_path), ignore_errors=True)
    test_configs_path.mkdir(parents=True)

    for m_name, conf_dict in PARAMS.items():
        test_configs_path.joinpath(m_name).mkdir(exist_ok=True, parents=True)
        for (config_path, _, _), _ in conf_dict.items():
            download_config(config_path)

    os.environ['DP_ROOT_PATH'] = str(download_path)
    os.environ['DP_CONFIGS_PATH'] = str(test_configs_path)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['DP_CACHE_DIR'] = str(cache_dir.resolve())


def teardown_module():
    shutil.rmtree(str(test_configs_path.parent), ignore_errors=True)
    shutil.rmtree(str(download_path), ignore_errors=True)

    if cache_dir:
        shutil.rmtree(str(cache_dir), ignore_errors=True)


def _serialize(config):
    chainer = build_model(config, download=True)
    return chainer.serialize()


def _infer(config, inputs, download=False):
    chainer = build_model(config, download=download)
    if inputs:
        prediction = chainer(*inputs)
        if len(chainer.out_params) == 1:
            prediction = [prediction]
    else:
        prediction = []
    return prediction


def _deserialize(config, raw_bytes, examples):
    chainer = build_model(config, serialized=raw_bytes)
    for *query, expected_response in examples:
        query = [[q] for q in query]
        actual_response = chainer(*query)
        if expected_response is not None:
            if actual_response is not None and len(actual_response) > 0:
                actual_response = actual_response[0]
            assert expected_response == str(actual_response), \
                f"Error in interacting with {model_dir} ({conf_file}): {query}"


@pytest.mark.parametrize("model,conf_file,model_dir,mode", TEST_GRID, scope='class')
class TestQuickStart(object):
    @staticmethod
    def infer(config_path, qr_list=None, check_outputs=True):

        *inputs, expected_outputs = zip(*qr_list) if qr_list else ([],)
        with ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(_infer, config_path, inputs)
        outputs = list(zip(*f.result()))

        if check_outputs:
            errors = ';'.join([f'expected `{expected}` got `{output}`'
                               for output, expected in zip(outputs, expected_outputs)
                               if expected is not None and expected != output])
            if errors:
                raise RuntimeError(f'Unexpected results for {config_path}: {errors}')

    @staticmethod
    def infer_api(config_path):
        server_params = get_server_params(config_path)

        url_base = 'http://{}:{}'.format(server_params['host'], api_port or server_params['port'])
        url = urljoin(url_base.replace('http://0.0.0.0:', 'http://127.0.0.1:'), server_params['model_endpoint'])

        post_headers = {'Accept': 'application/json'}

        logfile = io.BytesIO(b'')
        args = [sys.executable, "-m", "deeppavlov", "riseapi", str(config_path)]
        if api_port:
            args += ['-p', str(api_port)]
        p = pexpect.popen_spawn.PopenSpawn(' '.join(args),
                                           timeout=None, logfile=logfile)
        try:
            p.expect(url_base)

            get_url = urljoin(url_base.replace('http://0.0.0.0:', 'http://127.0.0.1:'), '/api')
            get_response = requests.get(get_url)
            response_code = get_response.status_code
            assert response_code == 200, f"GET /api request returned error code {response_code} with {config_path}"

            model_args_names = get_response.json()
            post_payload = dict()
            for arg_name in model_args_names:
                arg_value = ' '.join(['qwerty'] * 10)
                post_payload[arg_name] = [arg_value]

            post_response = requests.post(url, json=post_payload, headers=post_headers)
            response_code = post_response.status_code
            assert response_code == 200, f"POST request returned error code {response_code} with {config_path}"

        except pexpect.exceptions.EOF:
            raise RuntimeError('Got unexpected EOF: \n{}'.format(logfile.getvalue().decode()))

        finally:
            p.kill(signal.SIGTERM)
            p.wait()
            # if p.wait() != 0:
            #     raise RuntimeError('Error in shutting down API server: \n{}'.format(logfile.getvalue().decode()))

    @staticmethod
    def infer_socket(config_path, socket_type):
        socket_params = get_server_params(config_path)
        model_args_names = socket_params['model_args_names']

        host = socket_params['host']
        host = host.replace('0.0.0.0', '127.0.0.1')
        port = api_port or socket_params['port']

        socket_payload = {}
        for arg_name in model_args_names:
            arg_value = ' '.join(['qwerty'] * 10)
            socket_payload[arg_name] = [arg_value]

        logfile = io.BytesIO(b'')
        args = [sys.executable, "-m", "deeppavlov", "risesocket", str(config_path), '--socket-type', socket_type]
        if socket_type == 'TCP':
            args += ['-p', str(port)]
            address_family = socket.AF_INET
            connect_arg = (host, port)
        else:
            address_family = socket.AF_UNIX
            connect_arg = socket_params['unix_socket_file']
        p = pexpect.popen_spawn.PopenSpawn(' '.join(args),
                                           timeout=None, logfile=logfile)
        try:
            p.expect(socket_params['socket_launch_message'])
            with socket.socket(address_family, socket.SOCK_STREAM) as s:
                try:
                    s.connect(connect_arg)
                except ConnectionRefusedError:
                    sleep(1)
                    s.connect(connect_arg)
                s.sendall(encode(socket_payload))
                s.settimeout(120)
                header = s.recv(4)
                body_len = unpack('<I', header)[0]
                data = bytearray()
                while len(data) < body_len:
                    chunk = s.recv(body_len - len(data))
                    if not chunk:
                        raise ValueError(f'header does not match body\nheader: {body_len}\nbody length: {len(data)}'
                                         f'data: {data}')
                    data.extend(chunk)
            try:
                resp = json.loads(data)
            except json.decoder.JSONDecodeError:
                raise ValueError(f"Can't decode model response {data}")
            assert resp['status'] == 'OK', f"{socket_type} socket request returned status: {resp['status']}" \
                                           f" with {config_path}\n{logfile.getvalue().decode()}"

        except pexpect.exceptions.EOF:
            raise RuntimeError(f'Got unexpected EOF: \n{logfile.getvalue().decode()}')

        except json.JSONDecodeError:
            raise ValueError(f'Got JSON not serializable response from model: "{data}"\n{logfile.getvalue().decode()}')

        finally:
            p.kill(signal.SIGTERM)
            p.wait()

    def test_inferring_pretrained_model(self, model, conf_file, model_dir, mode):
        if 'IP' in mode:
            config_file_path = str(test_configs_path.joinpath(conf_file))
            install_config(config_file_path)
            deep_download(config_file_path)

            self.infer(test_configs_path / conf_file, PARAMS[model][(conf_file, model_dir, mode)])
        else:
            pytest.skip("Unsupported mode: {}".format(mode))

    def test_inferring_pretrained_model_api(self, model, conf_file, model_dir, mode):
        if 'IP' in mode:
            self.infer_api(test_configs_path / conf_file)
        else:
            pytest.skip("Unsupported mode: {}".format(mode))

    def test_inferring_pretrained_model_socket(self, model, conf_file, model_dir, mode):
        if 'IP' in mode:
            self.infer_socket(test_configs_path / conf_file, 'TCP')

            if 'TI' not in mode:
                shutil.rmtree(str(download_path), ignore_errors=True)
        else:
            pytest.skip(f"Unsupported mode: {mode}")

    def test_serialization(self, model, conf_file, model_dir, mode):
        if 'SR' not in mode:
            return pytest.skip("Unsupported mode: {}".format(mode))

        config_file_path = test_configs_path / conf_file

        with ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(_serialize, config_file_path)
        raw_bytes = f.result()

        serialized: list = pickle.loads(raw_bytes)
        if not any(serialized):
            pytest.skip("Serialization not supported: {}".format(conf_file))
            return
        serialized.clear()

        with ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(_deserialize, config_file_path, raw_bytes, PARAMS[model][(conf_file, model_dir, mode)])

        exc = f.exception()
        if exc is not None:
            raise exc

    def test_consecutive_training_and_inferring(self, model, conf_file, model_dir, mode):
        if 'TI' in mode:
            c = test_configs_path / conf_file
            model_path = download_path / model_dir

            if 'IP' not in mode:
                config_path = str(test_configs_path.joinpath(conf_file))
                install_config(config_path)
                deep_download(config_path)
            shutil.rmtree(str(model_path), ignore_errors=True)

            logfile = io.BytesIO(b'')
            p = pexpect.popen_spawn.PopenSpawn(sys.executable + " -m deeppavlov train " + str(c), timeout=None,
                                               logfile=logfile)
            p.readlines()
            if p.wait() != 0:
                raise RuntimeError('Training process of {} returned non-zero exit code: \n{}'
                                   .format(model_dir, logfile.getvalue().decode()))
            self.infer(c, PARAMS[model][(conf_file, model_dir, mode)], check_outputs=False)

            shutil.rmtree(str(download_path), ignore_errors=True)
        else:
            pytest.skip("Unsupported mode: {}".format(mode))
