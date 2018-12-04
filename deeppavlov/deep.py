"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse

from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.commands.infer import interact_model, predict_on_stream
from deeppavlov.core.common.file import find_config
from deeppavlov.core.common.log import get_logger
from deeppavlov.download import deep_download
from deeppavlov.core.common.cross_validation import calc_cv_score
from utils.alice import start_alice_server
from utils.telegram_utils.telegram_ui import interact_model_by_telegram
from utils.server_utils.server import start_model_server
from utils.ms_bot_framework_utils.server import run_ms_bf_default_agent
from utils.pip_wrapper import install_from_config


log = get_logger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("mode", help="select a mode, train or interact", type=str,
                    choices={'train', 'evaluate', 'interact', 'predict', 'interactbot', 'interactmsbot',
                             'riseapi', 'download', 'install', 'crossval'})
parser.add_argument("config_path", help="path to a pipeline json config", type=str)

parser.add_argument("-e", "--start-epoch-num", dest="start_epoch_num", default=0, help="Start epoch number", type=int)
parser.add_argument("--recursive", action="store_true", help="Train nested configs")

parser.add_argument("-b", "--batch-size", dest="batch_size", default=1, help="inference batch size", type=int)
parser.add_argument("-f", "--input-file", dest="file_path", default=None, help="Path to the input file", type=str)
parser.add_argument("-d", "--download", action="store_true", help="download model components")

parser.add_argument("--folds", help="number of folds", type=int, default=5)

parser.add_argument("-t", "--token", default=None,  help="telegram bot token", type=str)
parser.add_argument("-i", "--ms-id", default=None, help="microsoft bot framework app id", type=str)
parser.add_argument("-s", "--ms-secret", default=None, help="microsoft bot framework app secret", type=str)

parser.add_argument("--multi-instance", action="store_true", help="allow rising of several instances of the model")
parser.add_argument("--stateful", action="store_true", help="interact with a stateful model")

parser.add_argument("--https", action="store_true", help="run model in https mode")
parser.add_argument("--key", default=None, help="ssl key", type=str)
parser.add_argument("--cert", default=None, help="ssl certificate", type=str)

parser.add_argument("--api-mode", help="rest api mode: 'basic' with batches or 'alice' for  Yandex.Dialogs format",
                    type=str, default='basic', choices={'basic', 'alice'})


def main():
    args = parser.parse_args()
    pipeline_config_path = find_config(args.config_path)

    if args.download or args.mode == 'download':
        deep_download(pipeline_config_path)

    multi_instance = args.multi_instance
    stateful = args.stateful

    start_epoch_num = args.start_epoch_num

    if args.mode == 'train':
        train_evaluate_model_from_config(pipeline_config_path, recursive=args.recursive, 
                                         start_epoch_num=start_epoch_num)
    elif args.mode == 'evaluate':
        train_evaluate_model_from_config(pipeline_config_path, to_train=False, to_validate=False,
                                         start_epoch_num=start_epoch_num)
    elif args.mode == 'interact':
        interact_model(pipeline_config_path)
    elif args.mode == 'interactbot':
        token = args.token
        interact_model_by_telegram(pipeline_config_path, token)
    elif args.mode == 'interactmsbot':
        ms_id = args.ms_id
        ms_secret = args.ms_secret
        run_ms_bf_default_agent(model_config=pipeline_config_path,
                                app_id=ms_id,
                                app_secret=ms_secret,
                                multi_instance=multi_instance,
                                stateful=stateful)
    elif args.mode == 'riseapi':
        alice = args.api_mode == 'alice'
        https = args.https
        ssl_key = args.key
        ssl_cert = args.cert
        if alice:
            start_alice_server(pipeline_config_path, https, ssl_key, ssl_cert)
        else:
            start_model_server(pipeline_config_path, https, ssl_key, ssl_cert)
    elif args.mode == 'predict':
        predict_on_stream(pipeline_config_path, args.batch_size, args.file_path)
    elif args.mode == 'install':
        install_from_config(pipeline_config_path)
    elif args.mode == 'crossval':
        if args.folds < 2:
            log.error('Minimum number of Folds is 2')
        else:
            n_folds = args.folds
            calc_cv_score(pipeline_config_path, n_folds=n_folds, is_loo=False)


if __name__ == "__main__":
    main()
