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
from pathlib import Path
import sys
import os

p = (Path(__file__) / ".." / "..").resolve()
sys.path.append(str(p))

from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.commands.infer import interact_model, predict_on_stream
from deeppavlov.core.common.log import get_logger
from deeppavlov.download import deep_download
from utils.telegram_utils.telegram_ui import interact_model_by_telegram
from utils.server_utils.server import start_model_server
from utils.pip_wrapper import install_from_config


log = get_logger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("mode", help="select a mode, train or interact", type=str,
                    choices={'train', 'evaluate', 'interact', 'predict', 'interactbot', 'riseapi', 'download',
                             'install'})
parser.add_argument("config_path", help="path to a pipeline json config", type=str)
parser.add_argument("-t", "--token", help="telegram bot token", type=str)
parser.add_argument("-b", "--batch-size", dest="batch_size", default=1, help="inference batch size", type=int)
parser.add_argument("-f", "--input-file", dest="file_path", default=None, help="Path to the input file", type=str)
parser.add_argument("-d", "--download", action="store_true", help="download model components")


def find_config(pipeline_config_path: str):
    if not Path(pipeline_config_path).is_file():
        configs = [c for c in Path(__file__).parent.glob(f'configs/**/{pipeline_config_path}.json')
                   if str(c.with_suffix('')).endswith(pipeline_config_path)]  # a simple way to not allow * and ?
        if configs:
            log.info(f"Interpreting '{pipeline_config_path}' as '{configs[0]}'")
            pipeline_config_path = str(configs[0])
    return pipeline_config_path


def main():
    args = parser.parse_args()
    pipeline_config_path = find_config(args.config_path)
    if args.download or args.mode == 'download':
        deep_download(['-c', pipeline_config_path])
    token = args.token or os.getenv('TELEGRAM_TOKEN')

    if args.mode == 'train':
        train_evaluate_model_from_config(pipeline_config_path)
    elif args.mode == 'evaluate':
        train_evaluate_model_from_config(pipeline_config_path, to_train=False, to_validate=False)
    elif args.mode == 'interact':
        interact_model(pipeline_config_path)
    elif args.mode == 'interactbot':
        if not token:
            log.error('Token required: initiate -t param or TELEGRAM_BOT env var with Telegram bot token')
        else:
            interact_model_by_telegram(pipeline_config_path, token)
    elif args.mode == 'riseapi':
        start_model_server(pipeline_config_path)
    elif args.mode == 'predict':
        predict_on_stream(pipeline_config_path, args.batch_size, args.file_path)
    elif args.mode == 'install':
        install_from_config(pipeline_config_path)


if __name__ == "__main__":
    main()
