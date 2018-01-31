"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed inder the Apache License, Version 2.0 (the "License");
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

from deeppavlov.core.commands.utils import set_usr_dir, get_usr_dir
from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.infer import interact_model
from telegram_utils.telegram_ui import interact_model_by_telegram

parser = argparse.ArgumentParser()

parser.add_argument("mode", help="select a mode, train or interact", type=str,
                    choices={'train', 'interact', 'interactbot'})
parser.add_argument("config_path", help="path to a pipeline json config", type=str)
parser.add_argument("-t", "--token", help="telegram bot token", type=str)


def main():
    args = parser.parse_args()
    pipeline_config_path = args.config_path
    set_usr_dir(pipeline_config_path)

    token = args.token or os.getenv('TELEGRAM_TOKEN')

    try:
        if args.mode == 'train':
            train_model_from_config(pipeline_config_path)
        elif args.mode == 'interact':
            interact_model(pipeline_config_path)
        elif args.mode == 'interactbot':
            if not token:
                print(
                    'Token required: initiate -t parm or TELEGRAM_BOT env var with Telegram bot token')
            else:
                interact_model_by_telegram(pipeline_config_path, token)
    finally:
        usr_dir = get_usr_dir()
        if not list(usr_dir.iterdir()):
            usr_dir.rmdir()


if __name__ == "__main__":
    main()
