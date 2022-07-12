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

import argparse
from logging import getLogger

from deeppavlov.core.commands.infer import interact_model, predict_on_stream
from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.common.cross_validation import calc_cv_score
from deeppavlov.core.common.file import find_config
from deeppavlov.download import deep_download
from deeppavlov.utils.agent import start_rabbit_service
from deeppavlov.utils.pip_wrapper import install_from_config
from deeppavlov.utils.server import start_model_server
from deeppavlov.utils.socket import start_socket_server

log = getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("mode", help="select a mode, train or interact", type=str,
                    choices={'train', 'evaluate', 'interact', 'predict', 'riseapi', 'risesocket', 'agent-rabbit',
                             'download', 'install', 'crossval'})
parser.add_argument("config_path", help="path to a pipeline json config", type=str)

parser.add_argument("-e", "--start-epoch-num", dest="start_epoch_num", default=None,
                    help="Start epoch number", type=int)
parser.add_argument("--recursive", action="store_true", help="Train nested configs")

parser.add_argument("-b", "--batch-size", dest="batch_size", default=None, help="inference batch size", type=int)
parser.add_argument("-f", "--input-file", dest="file_path", default=None, help="Path to the input file", type=str)
parser.add_argument("-d", "--download", action="store_true", help="download model components")

parser.add_argument("--folds", help="number of folds", type=int, default=5)

parser.add_argument("--https", action="store_true", default=None, help="run model in https mode")
parser.add_argument("--key", default=None, help="ssl key", type=str)
parser.add_argument("--cert", default=None, help="ssl certificate", type=str)

parser.add_argument("-p", "--port", default=None, help="api port", type=int)

parser.add_argument("--socket-type", default="TCP", type=str, choices={"TCP", "UNIX"})
parser.add_argument("--socket-file", default="/tmp/deeppavlov_socket.s", type=str)

parser.add_argument("-sn", "--service-name", default=None, help="service name for agent-rabbit mode", type=str)
parser.add_argument("-an", "--agent-namespace", default=None, help="dp-agent namespace name", type=str)
parser.add_argument("-ul", "--utterance-lifetime", default=None, help="message expiration in seconds", type=int)
parser.add_argument("-rh", "--rabbit-host", default=None, help="RabbitMQ server host", type=str)
parser.add_argument("-rp", "--rabbit-port", default=None, help="RabbitMQ server port", type=int)
parser.add_argument("-rl", "--rabbit-login", default=None, help="RabbitMQ server login", type=str)
parser.add_argument("-rpwd", "--rabbit-password", default=None, help="RabbitMQ server password", type=str)
parser.add_argument("-rvh", "--rabbit-virtualhost", default=None, help="RabbitMQ server virtualhost", type=str)


def main():
    args = parser.parse_args()
    pipeline_config_path = find_config(args.config_path)

    if args.download or args.mode == 'download':
        deep_download(pipeline_config_path)

    if args.mode == 'train':
        train_evaluate_model_from_config(pipeline_config_path,
                                         recursive=args.recursive,
                                         start_epoch_num=args.start_epoch_num)
    elif args.mode == 'evaluate':
        train_evaluate_model_from_config(pipeline_config_path, to_train=False, start_epoch_num=args.start_epoch_num)
    elif args.mode == 'interact':
        interact_model(pipeline_config_path)
    elif args.mode == 'riseapi':
        start_model_server(pipeline_config_path, args.https, args.key, args.cert, port=args.port)
    elif args.mode == 'risesocket':
        start_socket_server(pipeline_config_path, args.socket_type, port=args.port, socket_file=args.socket_file)
    elif args.mode == 'agent-rabbit':
        start_rabbit_service(model_config=pipeline_config_path,
                             service_name=args.service_name,
                             agent_namespace=args.agent_namespace,
                             batch_size=args.batch_size,
                             utterance_lifetime_sec=args.utterance_lifetime,
                             rabbit_host=args.rabbit_host,
                             rabbit_port=args.rabbit_port,
                             rabbit_login=args.rabbit_login,
                             rabbit_password=args.rabbit_password,
                             rabbit_virtualhost=args.rabbit_virtualhost)
    elif args.mode == 'predict':
        predict_on_stream(pipeline_config_path, args.batch_size, args.file_path)
    elif args.mode == 'install':
        install_from_config(pipeline_config_path)
    elif args.mode == 'crossval':
        if args.folds < 2:
            log.error('Minimum number of Folds is 2')
        else:
            calc_cv_score(pipeline_config_path, n_folds=args.folds, is_loo=False)


if __name__ == "__main__":
    main()
