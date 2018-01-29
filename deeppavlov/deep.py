import argparse
from pathlib import Path
import sys

p = (Path(__file__) / ".." / "..").resolve()
sys.path.append(str(p))

from deeppavlov.core.commands.utils import set_usr_dir, get_usr_dir
from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.infer import interact_model
from utils.telegram_ui import interact_model_by_telegram

parser = argparse.ArgumentParser()

parser.add_argument("mode", help="select a mode, train or interact", type=str,
                    choices={'train', 'interact'})
parser.add_argument("config_path", help="path to a pipeline json config", type=str)
parser.add_argument("--tgbot", help="telegram bot token", type=str, default=None)


def main():
    args = parser.parse_args()
    pipeline_config_path = args.config_path
    set_usr_dir(pipeline_config_path)
    try:
        if args.mode == 'train':
            train_model_from_config(pipeline_config_path)
        elif args.mode == 'interact':
            token = args.tgbot
            if token is None:
                interact_model(pipeline_config_path)
            else:
                interact_model_by_telegram(pipeline_config_path, token)
    finally:
        usr_dir = get_usr_dir()
        if not list(usr_dir.iterdir()):
            usr_dir.rmdir()


if __name__ == "__main__":
    main()
